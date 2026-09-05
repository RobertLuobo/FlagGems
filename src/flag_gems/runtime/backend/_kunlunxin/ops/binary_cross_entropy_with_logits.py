import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import pointwise_dynamic

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Kunlunxin backend-local binary_cross_entropy_with_logits.
#
# Perf evidence (harness/solution/performance/binary_cross_entropy_with_logits_xpu3_20260817.md):
#  1. pointwise_dynamic 1D codegen uses a 512-lane tile: measured ~22 GB/s.
#     A single-tile-per-CTA copy at BLOCK=16384 reaches ~1.6 TB/s (torch
#     native copy: 1.85 TB/s).  Per-CTA dynamic `range` loops collapse to
#     ~4 GB/s (gm2lm cannot pipeline loads across dynamic loop iterations);
#     `tl.static_range` unrolling keeps the DMA path fast (~11 ms for the
#     full elementwise+reduce of 2^28 fp16, vs ~840 ms with the dynamic
#     loop).  So: static-unrolled kernels, BLOCK=16384, U=4-16.
#  2. The elementwise formula evaluates log(1+exp(-|x|)) once per element
#     (one exp + one log) instead of computing both branches; the other
#     branch is recovered exactly by identity log(1+e^x) = x + log(1+e^-x).
#     (2-exp + 2-log variant measured identical: the memory path, not the
#     transcendental count, is the wall on this backend.)
#  3. reduction=mean/sum: no zero-pad / second-buffer schemes are used.
#     Measured (2026-09-05, use_gems environment, CUDA-event timing): the
#     zero-padding + `_copy_from` + host `mid.sum() - pad*log(2)` pipeline
#     costs ~0.4 ms fixed (2 zeros 0.11 + 2 copies 0.06 + sum/sub/div/cast
#     caught by the generic registered kernels 0.35), while the native op is
#     a single ~0.02 ms kernel.  Instead a masked split reduction is used and
#     the whole mean/sum costs ONE launch for N<=16384 (scalar kernel) or TWO
#     launches beyond: an unmasked full-block stage-1 (grid = N//16384 CTAs)
#     plus a single-CTA kernel that folds the fp32 partials and adds the
#     partial tail loss over [Nbase, N) directly, then applies *1/N + cast.
#
#     CRITICAL backend hazard (isolated, 2026-09-05): a partial-masked tail
#     CTA inside a *reduction* tl.sum miscompiles once the grid exceeds ~12
#     CTAs -- `idx < N` collapses to all-true, the OOB lanes read past the
#     tensor and their garbage loss is accumulated (N=5,924,352 measured
#     +5.3e3 relative 1.1e-3 error; N=96,000/grid<=12 correct; the flat
#     pointwise kernels and all grid<=12 cases are unaffected).  Hence the
#     two-stage shape above: stage-1 CTAs are never partially masked and the
#     tail is handled by a separate single-CTA launch, which was verified
#     correct for tail in {1, ..., 16383}.
#
#     Notes on the masked-load hazards found during this work (all verified
#     in isolation with minimal kernels):
#      a. `tl.sum(tl.where(m, v, 0.0))` (predicate fused into the reduction)
#         returns 0 for N < BLOCK on this backend; the accumulation form
#         `acc; acc += tl.where(m, v, 0.0); tl.sum(acc)` is correct.
#      b. bf16 masked loads return 0 for the *valid* lanes when only a single
#         element is masked in of a 16384-lane tile (N=1).  fp16/fp32 are
#         unaffected.  Consequently N <= _SMALL_N (2048) keeps the generic
#         pointwise path, and a bf16 N % 16384 == 1 (tail==1) also falls back
#         to it; no benchmark/test shape below 2048 exists besides the
#         ()/(1,) functional cases.
#      c. `tl.load(..., other=0.0)` applied to bf16 tiles also zeroes the
#         valid lanes; the masked without-`other` + where form is used.
#  4. The pointwise tensor is never materialized for mean/sum.
# ---------------------------------------------------------------------------

# tl.sum safety cap on this backend: 8192 (no buffer) / 32768 (with
# buffer_size_limit=2048).  BLOCK=16384 + buffer_size_limit fulfils it.
_BULK_BLOCK = 16384
_SMALL_N = 2048
_FLAT_BLOCK = 2048
_FLAT_U = 8


def _flat_cfg(N):
    grid = max(1, triton.cdiv(N, _FLAT_BLOCK * _FLAT_U))
    return _FLAT_BLOCK, _FLAT_U, grid


def _need_mask(N, blk, u):
    return (N % (blk * u)) != 0


# ---------------------------- elementwise helpers ---------------------------


@triton.jit
def _bce_loss(xv, yv):
    # stable single-exp: log(1+e^-|x|) + max(x,0) - x*y
    return tl.log(1.0 + tl.exp(-tl.abs(xv))) + tl.maximum(xv, 0.0) - xv * yv


@triton.jit
def _bce_pos_weight_loss(xv, yv, pv):
    log_e = tl.log(1.0 + tl.exp(-tl.abs(xv)))
    neg_log = tl.where(xv >= 0, log_e, -xv + log_e)  # log(1+e^-x)
    pos_log = tl.where(xv >= 0, xv + log_e, log_e)  # log(1+e^x)
    x_pos = yv * pv * neg_log + (1.0 - yv) * (xv + neg_log)
    x_neg = yv * (-pv * xv + pv * pos_log) + (1.0 - yv) * pos_log
    return tl.where(xv >= 0.0, x_pos, x_neg)


# ----------- unmasked full-block stage-1 kernels (N>16384) ------------------
# grid * BLOCK * U covers exactly the first `full` blocks of N (N % (BLOCK*U)
# elements handled by the fused finalize+tail kernel below).  No mask at all:
# masked tails in a >12-CTA reduction miscompile on this backend (the last
# CTA's lanes read past N and the mask/where are dropped), see
# harness/solution/performance/binary_cross_entropy_with_logits_xpu3_*.md.


@triton.jit
def _bce_reduce_full_kernel(x, y, mid, BLOCK: tl.constexpr, U: tl.constexpr):
    pid = tl.program_id(0)
    base = pid * BLOCK * U
    acc = tl.zeros([BLOCK], dtype=tl.float32)
    for i in tl.static_range(U):
        idx = base + i * BLOCK + tl.arange(0, BLOCK)
        xv = tl.load(x + idx).to(tl.float32)
        yv = tl.load(y + idx).to(tl.float32)
        acc += _bce_loss(xv, yv)
    tl.store(mid + pid, tl.sum(acc))


@triton.jit
def _bce_weight_reduce_full_kernel(x, y, w, mid, BLOCK: tl.constexpr, U: tl.constexpr):
    pid = tl.program_id(0)
    base = pid * BLOCK * U
    acc = tl.zeros([BLOCK], dtype=tl.float32)
    for i in tl.static_range(U):
        idx = base + i * BLOCK + tl.arange(0, BLOCK)
        xv = tl.load(x + idx).to(tl.float32)
        yv = tl.load(y + idx).to(tl.float32)
        wv = tl.load(w + idx).to(tl.float32)
        acc += _bce_loss(xv, yv) * wv
    tl.store(mid + pid, tl.sum(acc))


@triton.jit
def _bce_pos_weight_reduce_full_kernel(
    x, y, pw, mid, BLOCK: tl.constexpr, U: tl.constexpr
):
    pid = tl.program_id(0)
    base = pid * BLOCK * U
    acc = tl.zeros([BLOCK], dtype=tl.float32)
    for i in tl.static_range(U):
        idx = base + i * BLOCK + tl.arange(0, BLOCK)
        xv = tl.load(x + idx).to(tl.float32)
        yv = tl.load(y + idx).to(tl.float32)
        pv = tl.load(pw + idx).to(tl.float32)
        acc += _bce_pos_weight_loss(xv, yv, pv)
    tl.store(mid + pid, tl.sum(acc))


@triton.jit
def _bce_weight_pos_weight_reduce_full_kernel(
    x, y, w, pw, mid, BLOCK: tl.constexpr, U: tl.constexpr
):
    pid = tl.program_id(0)
    base = pid * BLOCK * U
    acc = tl.zeros([BLOCK], dtype=tl.float32)
    for i in tl.static_range(U):
        idx = base + i * BLOCK + tl.arange(0, BLOCK)
        xv = tl.load(x + idx).to(tl.float32)
        yv = tl.load(y + idx).to(tl.float32)
        wv = tl.load(w + idx).to(tl.float32)
        pv = tl.load(pw + idx).to(tl.float32)
        acc += _bce_pos_weight_loss(xv, yv, pv) * wv
    tl.store(mid + pid, tl.sum(acc))


# ------------- single-launch scalar kernels (N<=16384, mean/sum) ------------
# One CTA covers the whole input; the result (with *1/N and dtype cast) is
# written straight to the output tensor so mean/sum costs exactly one launch.


@triton.jit
def _bce_scalar_kernel(x, y, out, N, inv_n, BLOCK: tl.constexpr, U: tl.constexpr):
    acc = tl.zeros([BLOCK], dtype=tl.float32)
    for i in tl.static_range(U):
        idx = i * BLOCK + tl.arange(0, BLOCK)
        m = idx < N
        xv = tl.load(x + idx, mask=m).to(tl.float32)
        yv = tl.load(y + idx, mask=m).to(tl.float32)
        acc += tl.where(m, _bce_loss(xv, yv), 0.0)
    tl.store(out, (tl.sum(acc) * inv_n).to(out.dtype.element_ty))


@triton.jit
def _bce_weight_scalar_kernel(
    x, y, w, out, N, inv_n, BLOCK: tl.constexpr, U: tl.constexpr
):
    acc = tl.zeros([BLOCK], dtype=tl.float32)
    for i in tl.static_range(U):
        idx = i * BLOCK + tl.arange(0, BLOCK)
        m = idx < N
        xv = tl.load(x + idx, mask=m).to(tl.float32)
        yv = tl.load(y + idx, mask=m).to(tl.float32)
        wv = tl.load(w + idx, mask=m).to(tl.float32)
        acc += tl.where(m, _bce_loss(xv, yv) * wv, 0.0)
    tl.store(out, (tl.sum(acc) * inv_n).to(out.dtype.element_ty))


@triton.jit
def _bce_pos_weight_scalar_kernel(
    x, y, pw, out, N, inv_n, BLOCK: tl.constexpr, U: tl.constexpr
):
    acc = tl.zeros([BLOCK], dtype=tl.float32)
    for i in tl.static_range(U):
        idx = i * BLOCK + tl.arange(0, BLOCK)
        m = idx < N
        xv = tl.load(x + idx, mask=m).to(tl.float32)
        yv = tl.load(y + idx, mask=m).to(tl.float32)
        pv = tl.load(pw + idx, mask=m).to(tl.float32)
        acc += tl.where(m, _bce_pos_weight_loss(xv, yv, pv), 0.0)
    tl.store(out, (tl.sum(acc) * inv_n).to(out.dtype.element_ty))


@triton.jit
def _bce_weight_pos_weight_scalar_kernel(
    x, y, w, pw, out, N, inv_n, BLOCK: tl.constexpr, U: tl.constexpr
):
    acc = tl.zeros([BLOCK], dtype=tl.float32)
    for i in tl.static_range(U):
        idx = i * BLOCK + tl.arange(0, BLOCK)
        m = idx < N
        xv = tl.load(x + idx, mask=m).to(tl.float32)
        yv = tl.load(y + idx, mask=m).to(tl.float32)
        wv = tl.load(w + idx, mask=m).to(tl.float32)
        pv = tl.load(pw + idx, mask=m).to(tl.float32)
        acc += tl.where(m, _bce_pos_weight_loss(xv, yv, pv) * wv, 0.0)
    tl.store(out, (tl.sum(acc) * inv_n).to(out.dtype.element_ty))


# ------------- stage-1b tail kernels (grid=1, masked) -----------------------
# Covers the partial tail [Nbase, N) (N - Nbase < BLOCK) of an N>16384 input.
# Kept as a *separate* single-CTA launch: an earlier merged fold+tail kernel
# (two masked sections in one kernel) miscompiled on this backend even when
# the tail section was entirely masked off.


@triton.jit
def _bce_tail_kernel(x, y, mid, Nbase, N, BLOCK: tl.constexpr, U: tl.constexpr):
    acc = tl.zeros([BLOCK], dtype=tl.float32)
    for i in tl.static_range(U):
        idx = Nbase + i * BLOCK + tl.arange(0, BLOCK)
        m = idx < N
        xv = tl.load(x + idx, mask=m).to(tl.float32)
        yv = tl.load(y + idx, mask=m).to(tl.float32)
        acc += tl.where(m, _bce_loss(xv, yv), 0.0)
    tl.store(mid, tl.sum(acc))


@triton.jit
def _bce_weight_tail_kernel(
    x, y, w, mid, Nbase, N, BLOCK: tl.constexpr, U: tl.constexpr
):
    acc = tl.zeros([BLOCK], dtype=tl.float32)
    for i in tl.static_range(U):
        idx = Nbase + i * BLOCK + tl.arange(0, BLOCK)
        m = idx < N
        xv = tl.load(x + idx, mask=m).to(tl.float32)
        yv = tl.load(y + idx, mask=m).to(tl.float32)
        wv = tl.load(w + idx, mask=m).to(tl.float32)
        acc += tl.where(m, _bce_loss(xv, yv) * wv, 0.0)
    tl.store(mid, tl.sum(acc))


@triton.jit
def _bce_pos_weight_tail_kernel(
    x, y, pw, mid, Nbase, N, BLOCK: tl.constexpr, U: tl.constexpr
):
    acc = tl.zeros([BLOCK], dtype=tl.float32)
    for i in tl.static_range(U):
        idx = Nbase + i * BLOCK + tl.arange(0, BLOCK)
        m = idx < N
        xv = tl.load(x + idx, mask=m).to(tl.float32)
        yv = tl.load(y + idx, mask=m).to(tl.float32)
        pv = tl.load(pw + idx, mask=m).to(tl.float32)
        acc += tl.where(m, _bce_pos_weight_loss(xv, yv, pv), 0.0)
    tl.store(mid, tl.sum(acc))


@triton.jit
def _bce_weight_pos_weight_tail_kernel(
    x, y, w, pw, mid, Nbase, N, BLOCK: tl.constexpr, U: tl.constexpr
):
    acc = tl.zeros([BLOCK], dtype=tl.float32)
    for i in tl.static_range(U):
        idx = Nbase + i * BLOCK + tl.arange(0, BLOCK)
        m = idx < N
        xv = tl.load(x + idx, mask=m).to(tl.float32)
        yv = tl.load(y + idx, mask=m).to(tl.float32)
        wv = tl.load(w + idx, mask=m).to(tl.float32)
        pv = tl.load(pw + idx, mask=m).to(tl.float32)
        acc += tl.where(m, _bce_pos_weight_loss(xv, yv, pv) * wv, 0.0)
    tl.store(mid, tl.sum(acc))


# ------------- stage-2 fold kernel (grid=1, fp32 partials) ------------------


@triton.jit
def _bce_finalize_kernel(mid, out, G, inv_n, BLOCK: tl.constexpr, U: tl.constexpr):
    acc = tl.zeros([BLOCK], dtype=tl.float32)
    for i in tl.static_range(U):
        idx = i * BLOCK + tl.arange(0, BLOCK)
        m = idx < G
        v = tl.load(mid + idx, mask=m, other=0.0)
        acc += tl.where(m, v, 0.0)
    tl.store(out, (tl.sum(acc) * inv_n).to(out.dtype.element_ty))


# -------------------- flat pointwise kernels (reduction=0) -------------------
@triton.jit
def _bce_flat_kernel(
    x, y, out, N, BLOCK: tl.constexpr, U: tl.constexpr, NEED_MASK: tl.constexpr
):
    pid = tl.program_id(0)
    base = pid * BLOCK * U
    for i in tl.static_range(U):
        idx = base + i * BLOCK + tl.arange(0, BLOCK)
        if NEED_MASK:
            m = idx < N
            xv = tl.load(x + idx, mask=m, other=0.0).to(tl.float32)
            yv = tl.load(y + idx, mask=m, other=0.0).to(tl.float32)
            tl.store(out + idx, _bce_loss(xv, yv).to(out.dtype.element_ty), mask=m)
        else:
            xv = tl.load(x + idx).to(tl.float32)
            yv = tl.load(y + idx).to(tl.float32)
            tl.store(out + idx, _bce_loss(xv, yv).to(out.dtype.element_ty))


@triton.jit
def _bce_weight_flat_kernel(
    x, y, w, out, N, BLOCK: tl.constexpr, U: tl.constexpr, NEED_MASK: tl.constexpr
):
    pid = tl.program_id(0)
    base = pid * BLOCK * U
    for i in tl.static_range(U):
        idx = base + i * BLOCK + tl.arange(0, BLOCK)
        if NEED_MASK:
            m = idx < N
            xv = tl.load(x + idx, mask=m, other=0.0).to(tl.float32)
            yv = tl.load(y + idx, mask=m, other=0.0).to(tl.float32)
            wv = tl.load(w + idx, mask=m, other=0.0).to(tl.float32)
            tl.store(
                out + idx, (_bce_loss(xv, yv) * wv).to(out.dtype.element_ty), mask=m
            )
        else:
            xv = tl.load(x + idx).to(tl.float32)
            yv = tl.load(y + idx).to(tl.float32)
            wv = tl.load(w + idx).to(tl.float32)
            tl.store(out + idx, (_bce_loss(xv, yv) * wv).to(out.dtype.element_ty))


@triton.jit
def _bce_pos_weight_flat_kernel(
    x, y, pw, out, N, BLOCK: tl.constexpr, U: tl.constexpr, NEED_MASK: tl.constexpr
):
    pid = tl.program_id(0)
    base = pid * BLOCK * U
    for i in tl.static_range(U):
        idx = base + i * BLOCK + tl.arange(0, BLOCK)
        if NEED_MASK:
            m = idx < N
            xv = tl.load(x + idx, mask=m, other=0.0).to(tl.float32)
            yv = tl.load(y + idx, mask=m, other=0.0).to(tl.float32)
            pv = tl.load(pw + idx, mask=m, other=0.0).to(tl.float32)
            tl.store(
                out + idx,
                _bce_pos_weight_loss(xv, yv, pv).to(out.dtype.element_ty),
                mask=m,
            )
        else:
            xv = tl.load(x + idx).to(tl.float32)
            yv = tl.load(y + idx).to(tl.float32)
            pv = tl.load(pw + idx).to(tl.float32)
            tl.store(
                out + idx, _bce_pos_weight_loss(xv, yv, pv).to(out.dtype.element_ty)
            )


@triton.jit
def _bce_weight_pos_weight_flat_kernel(
    x,
    y,
    w,
    pw,
    out,
    N,
    BLOCK: tl.constexpr,
    U: tl.constexpr,
    NEED_MASK: tl.constexpr,
):
    pid = tl.program_id(0)
    base = pid * BLOCK * U
    for i in tl.static_range(U):
        idx = base + i * BLOCK + tl.arange(0, BLOCK)
        if NEED_MASK:
            m = idx < N
            xv = tl.load(x + idx, mask=m, other=0.0).to(tl.float32)
            yv = tl.load(y + idx, mask=m, other=0.0).to(tl.float32)
            wv = tl.load(w + idx, mask=m, other=0.0).to(tl.float32)
            pv = tl.load(pw + idx, mask=m, other=0.0).to(tl.float32)
            tl.store(
                out + idx,
                (_bce_pos_weight_loss(xv, yv, pv) * wv).to(out.dtype.element_ty),
                mask=m,
            )
        else:
            xv = tl.load(x + idx).to(tl.float32)
            yv = tl.load(y + idx).to(tl.float32)
            wv = tl.load(w + idx).to(tl.float32)
            pv = tl.load(pw + idx).to(tl.float32)
            tl.store(
                out + idx,
                (_bce_pos_weight_loss(xv, yv, pv) * wv).to(out.dtype.element_ty),
            )


# ------------------ pointwise_dynamic fallback (non-contiguous) -------------


@pointwise_dynamic(is_tensor=[True, True], promotion_methods=[(0, 1, "DEFAULT")])
@triton.jit
def _bce_kernel(x, y):
    x_f32 = x.to(tl.float32)
    y_f32 = y.to(tl.float32)
    return tl.where(
        x_f32 >= 0,
        x_f32 - x_f32 * y_f32 + tl.log(1.0 + tl.exp(-x_f32)),
        tl.log(1.0 + tl.exp(x_f32)) - x_f32 * y_f32,
    )


@pointwise_dynamic(is_tensor=[True, True, True], promotion_methods=[(0, 1, "DEFAULT")])
@triton.jit
def _bce_weight_kernel(x, y, weight):
    x_f32 = x.to(tl.float32)
    y_f32 = y.to(tl.float32)
    w_f32 = weight.to(tl.float32)
    loss = tl.where(
        x_f32 >= 0,
        x_f32 - x_f32 * y_f32 + tl.log(1.0 + tl.exp(-x_f32)),
        tl.log(1.0 + tl.exp(x_f32)) - x_f32 * y_f32,
    )
    return loss * w_f32


@pointwise_dynamic(is_tensor=[True, True, True], promotion_methods=[(0, 1, "DEFAULT")])
@triton.jit
def _bce_pos_weight_kernel(x, y, pos_weight):
    x_f32 = x.to(tl.float32)
    y_f32 = y.to(tl.float32)
    pw_f32 = pos_weight.to(tl.float32)
    log_1p_exp_neg_x = tl.log(1.0 + tl.exp(-x_f32))
    log_1p_exp_x = tl.log(1.0 + tl.exp(x_f32))
    x_pos = y_f32 * pw_f32 * log_1p_exp_neg_x + (1.0 - y_f32) * (
        x_f32 + log_1p_exp_neg_x
    )
    x_neg = (
        y_f32 * (-pw_f32 * x_f32 + pw_f32 * log_1p_exp_x) + (1.0 - y_f32) * log_1p_exp_x
    )
    return tl.where(x_f32 >= 0, x_pos, x_neg)


@pointwise_dynamic(
    is_tensor=[True, True, True, True], promotion_methods=[(0, 1, "DEFAULT")]
)
@triton.jit
def _bce_weight_pos_weight_kernel(x, y, weight, pos_weight):
    x_f32 = x.to(tl.float32)
    y_f32 = y.to(tl.float32)
    w_f32 = weight.to(tl.float32)
    pw_f32 = pos_weight.to(tl.float32)
    log_1p_exp_neg_x = tl.log(1.0 + tl.exp(-x_f32))
    log_1p_exp_x = tl.log(1.0 + tl.exp(x_f32))
    x_pos = y_f32 * pw_f32 * log_1p_exp_neg_x + (1.0 - y_f32) * (
        x_f32 + log_1p_exp_neg_x
    )
    x_neg = (
        y_f32 * (-pw_f32 * x_f32 + pw_f32 * log_1p_exp_x) + (1.0 - y_f32) * log_1p_exp_x
    )
    loss = tl.where(x_f32 >= 0, x_pos, x_neg)
    return loss * w_f32


# --------------------------------- wrapper ----------------------------------

# Non-contiguous inputs and tiny tensors (N <= _SMALL_N, where the masked-load
# of a bf16 single element miscompiles) go through the generic pointwise
# path.  Contiguous inputs with N > _SMALL_N use the masked split reduction
# (no padding, no extra copies); N <= _BULK_BLOCK runs in a single launch.


def binary_cross_entropy_with_logits(
    self, target, weight=None, pos_weight=None, reduction=1
):
    logger.debug("GEMS_KUNLUNXIN BINARY_CROSS_ENTROPY_WITH_LOGITS")
    has_w = weight is not None
    has_pw = pos_weight is not None
    wargs = []
    if has_w:
        wargs.append(weight)
    if has_pw:
        wargs.append(pos_weight)

    use_flat = (
        self.is_contiguous()
        and target.is_contiguous()
        and all(wgt.is_contiguous() for wgt in wargs)
    )
    N = self.numel()

    if (
        not use_flat
        or N == 0
        or N <= _SMALL_N
        or (self.dtype == torch.bfloat16 and N % _BULK_BLOCK == 1)
    ):
        # fallback: pointwise_dynamic handles arbitrary layouts/strides;
        # tiny N (<= _SMALL_N) keeps the numerically-safe generic path;
        # N == 0 returns the reduction identity below.
        if has_w and has_pw:
            out = _bce_weight_pos_weight_kernel(self, target, weight, pos_weight)
        elif has_w:
            out = _bce_weight_kernel(self, target, weight)
        elif has_pw:
            out = _bce_pos_weight_kernel(self, target, pos_weight)
        else:
            out = _bce_kernel(self, target)

        if reduction == 2:
            if N == 0:
                return torch.zeros((), dtype=self.dtype, device=self.device)
            return out.to(torch.float32).reshape(-1).sum().to(self.dtype)
        if reduction == 1:
            if N == 0:
                return torch.full(
                    (), float("nan"), dtype=self.dtype, device=self.device
                )
            return (out.to(torch.float32).reshape(-1).sum() / N).to(self.dtype)
        return out

    # ---- fast path: contiguous inputs ----
    if reduction == 0:
        blk, u, grid = _flat_cfg(N)
        need_mask = _need_mask(N, blk, u)
        out = torch.empty_like(self)
        with torch_device_fn.device(self.device):
            if has_w and has_pw:
                _bce_weight_pos_weight_flat_kernel[(grid,)](
                    self,
                    target,
                    weight,
                    pos_weight,
                    out,
                    N,
                    BLOCK=blk,
                    U=u,
                    NEED_MASK=need_mask,
                )
            elif has_w:
                _bce_weight_flat_kernel[(grid,)](
                    self,
                    target,
                    weight,
                    out,
                    N,
                    BLOCK=blk,
                    U=u,
                    NEED_MASK=need_mask,
                )
            elif has_pw:
                _bce_pos_weight_flat_kernel[(grid,)](
                    self,
                    target,
                    pos_weight,
                    out,
                    N,
                    BLOCK=blk,
                    U=u,
                    NEED_MASK=need_mask,
                )
            else:
                _bce_flat_kernel[(grid,)](
                    self,
                    target,
                    out,
                    N,
                    BLOCK=blk,
                    U=u,
                    NEED_MASK=need_mask,
                )
        return out

    # mean (1) / sum (2): masked single- or two-stage reduction.
    inv_n = 1.0 / N if reduction == 1 else 1.0
    out = torch.empty((), dtype=self.dtype, device=self.device)
    with torch_device_fn.device(self.device):
        if N <= _BULK_BLOCK:
            # one launch: whole input in a single CTA (BLOCK=16384, U=1)
            if has_w and has_pw:
                _bce_weight_pos_weight_scalar_kernel[(1,)](
                    self,
                    target,
                    weight,
                    pos_weight,
                    out,
                    N,
                    inv_n,
                    BLOCK=_BULK_BLOCK,
                    U=1,
                    buffer_size_limit=2048,
                )
            elif has_w:
                _bce_weight_scalar_kernel[(1,)](
                    self,
                    target,
                    weight,
                    out,
                    N,
                    inv_n,
                    BLOCK=_BULK_BLOCK,
                    U=1,
                    buffer_size_limit=2048,
                )
            elif has_pw:
                _bce_pos_weight_scalar_kernel[(1,)](
                    self,
                    target,
                    pos_weight,
                    out,
                    N,
                    inv_n,
                    BLOCK=_BULK_BLOCK,
                    U=1,
                    buffer_size_limit=2048,
                )
            else:
                _bce_scalar_kernel[(1,)](
                    self,
                    target,
                    out,
                    N,
                    inv_n,
                    BLOCK=_BULK_BLOCK,
                    U=1,
                    buffer_size_limit=2048,
                )
            return out

        full = N // _BULK_BLOCK
        tail = N - full * _BULK_BLOCK
        nbase = full * _BULK_BLOCK
        grid = full + (1 if tail else 0)
        mid = torch.empty((grid,), dtype=torch.float32, device=self.device)
        if full:
            if has_w and has_pw:
                _bce_weight_pos_weight_reduce_full_kernel[(full,)](
                    self,
                    target,
                    weight,
                    pos_weight,
                    mid,
                    BLOCK=_BULK_BLOCK,
                    U=1,
                    buffer_size_limit=2048,
                )
            elif has_w:
                _bce_weight_reduce_full_kernel[(full,)](
                    self,
                    target,
                    weight,
                    mid,
                    BLOCK=_BULK_BLOCK,
                    U=1,
                    buffer_size_limit=2048,
                )
            elif has_pw:
                _bce_pos_weight_reduce_full_kernel[(full,)](
                    self,
                    target,
                    pos_weight,
                    mid,
                    BLOCK=_BULK_BLOCK,
                    U=1,
                    buffer_size_limit=2048,
                )
            else:
                _bce_reduce_full_kernel[(full,)](
                    self,
                    target,
                    mid,
                    BLOCK=_BULK_BLOCK,
                    U=1,
                    buffer_size_limit=2048,
                )
        if tail:
            if has_w and has_pw:
                _bce_weight_pos_weight_tail_kernel[(1,)](
                    self,
                    target,
                    weight,
                    pos_weight,
                    mid[full:],
                    nbase,
                    N,
                    BLOCK=_BULK_BLOCK,
                    U=1,
                    buffer_size_limit=2048,
                )
            elif has_w:
                _bce_weight_tail_kernel[(1,)](
                    self,
                    target,
                    weight,
                    mid[full:],
                    nbase,
                    N,
                    BLOCK=_BULK_BLOCK,
                    U=1,
                    buffer_size_limit=2048,
                )
            elif has_pw:
                _bce_pos_weight_tail_kernel[(1,)](
                    self,
                    target,
                    pos_weight,
                    mid[full:],
                    nbase,
                    N,
                    BLOCK=_BULK_BLOCK,
                    U=1,
                    buffer_size_limit=2048,
                )
            else:
                _bce_tail_kernel[(1,)](
                    self,
                    target,
                    mid[full:],
                    nbase,
                    N,
                    BLOCK=_BULK_BLOCK,
                    U=1,
                    buffer_size_limit=2048,
                )
        u2 = (grid + _BULK_BLOCK - 1) // _BULK_BLOCK
        _bce_finalize_kernel[(1,)](
            mid,
            out,
            grid,
            inv_n,
            BLOCK=_BULK_BLOCK,
            U=u2,
            buffer_size_limit=2048,
        )
    return out
