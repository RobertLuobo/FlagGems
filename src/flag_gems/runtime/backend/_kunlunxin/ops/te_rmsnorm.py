# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn


@triton.jit
def _te_rmsnorm_bwd_dx_1d_kernel(
    dx_ptr,
    dz_ptr,
    x_ptr,
    gamma_ptr,
    rsigma_ptr,
    N,
    zero_centered_gamma: tl.constexpr,
    C: tl.constexpr,
    NEED_TAIL: tl.constexpr,
):
    # 1D per-row dX (grid=(M,)): every load/store is a [C] block (block DMA),
    # no 2D [R, C] tile.  The 2D-tile variant saturates at ~35-125GB/s on this
    # backend (see the layernorm backward note: 2D tiles 20-60x slower than
    # 1D row blocks); the per-row layout keeps the row-sum (c1) as a [C]
    # elementwise accumulator + one 1D tl.sum (1D reduces are legal on XPU;
    # only 2D axis=0 reduces and [C, R] transposed loads are not).
    # Needs two passes (c1 before dx), so x/dz are re-read in pass 2.
    pid = tl.program_id(0)
    row = pid * N
    rsigma = tl.load(rsigma_ptr + pid).to(tl.float32)
    acc = tl.zeros([C], dtype=tl.float32)

    for off in range(0, N, C):
        cols = off + tl.arange(0, C)
        if NEED_TAIL:
            cmask = cols < N
        x = tl.load(x_ptr + row + cols).to(tl.float32)
        dz = tl.load(dz_ptr + row + cols).to(tl.float32)
        gamma = tl.load(gamma_ptr + cols).to(tl.float32)
        if zero_centered_gamma:
            gamma = gamma + 1.0
        if NEED_TAIL:
            acc += tl.where(cmask, x * rsigma * dz * gamma, 0.0)
        else:
            acc += x * rsigma * dz * gamma

    c1 = tl.sum(acc) / N

    for off in range(0, N, C):
        cols = off + tl.arange(0, C)
        if NEED_TAIL:
            cmask = cols < N
        x = tl.load(x_ptr + row + cols).to(tl.float32)
        dz = tl.load(dz_ptr + row + cols).to(tl.float32)
        gamma = tl.load(gamma_ptr + cols).to(tl.float32)
        if zero_centered_gamma:
            gamma = gamma + 1.0
        x_hat = x * rsigma
        dx = rsigma * (dz * gamma - x_hat * c1)
        if NEED_TAIL:
            dx = tl.where(cmask, dx, 0.0)
            tl.store(dx_ptr + row + cols, dx, mask=cmask)
        else:
            tl.store(dx_ptr + row + cols, dx)


@triton.jit
def _te_rmsnorm_bwd_dgamma_kernel(
    dgamma_partial_ptr,
    dz_ptr,
    x_ptr,
    rsigma_ptr,
    M,
    N,
    BM: tl.constexpr,
    C: tl.constexpr,
):
    # Validated XPU pattern for dW/dgamma (verbatim structure of
    # _kunlunxin/ops/layernorm.py::weight_bias_backward_1d_kernel): 1D [C]
    # column-vector loads with an M-loop, grid = (cdiv(N, C), cdiv(M, BM)).
    # 2D tiles are avoided entirely: a [R, C] tile needs an axis=0 reduce
    # (rejected by the XPU legalizer) and a [C, R] transposed load miscompiles.
    n0 = tl.program_id(0) * C
    mi = tl.program_id(1)
    m0 = mi * BM
    cols = tl.arange(0, C)
    cmask = (n0 + cols) < N
    acc = tl.zeros([C], dtype=tl.float32)
    for r in range(0, BM):
        m = m0 + r
        base = m * N + n0
        x = tl.load(x_ptr + base + cols, mask=cmask, other=0.0).to(tl.float32)
        dz = tl.load(dz_ptr + base + cols, mask=cmask, other=0.0).to(tl.float32)
        rsigma = tl.load(rsigma_ptr + m).to(tl.float32)
        acc += tl.where(cmask, dz * x * rsigma, 0.0)
    tl.store(dgamma_partial_ptr + mi * N + n0 + cols, acc, mask=cmask)


@triton.jit
def _te_rmsnorm_bwd_dgamma_reduce_kernel(
    dgamma_ptr,
    dgamma_partial_ptr,
    P,
    N,
    C: tl.constexpr,
):
    n0 = tl.program_id(0) * C
    cols = n0 + tl.arange(0, C)
    cmask = cols < N
    acc = tl.zeros([C], dtype=tl.float32)
    for i in range(0, P):
        acc += tl.load(
            dgamma_partial_ptr + i * N + cols, mask=cmask, other=0.0
        ).to(tl.float32)
    tl.store(dgamma_ptr + cols, acc, mask=cmask)


def _dgamma_bm_size(M):
    # rows per dgamma program: largest divisor of M <= _DGM_BM_MAX (so the
    # M-loop never masks OOB row reads under the M % BM == 0 contract); min 1.
    block = min(M, _DGM_BM_MAX)
    while block > 1 and M % block != 0:
        block //= 2
    return max(1, block)


_DGM_BM_MAX = 128


def _ln_bwd_col_size(N):
    # chunk width for the 1D backward kernels: largest power of 2 <= min(N, 8192)
    # (tl.arange must stay pow2; > 8192 lanes is a 1D defect on this backend).
    cap = min(N, 8192)
    return 1 << (cap.bit_length() - 1)


# --- te_rmsnorm_fwd (TE-aligned rmsnorm forward), XPU-local fast paths ---------
# The generic ``flag_gems.ops.te_rmsnorm`` fwd kernels cannot be used on XPU:
#   * ``rmsnorm_fwd_kernel`` launches one block of ``next_power_of_2(N)`` lanes;
#     for N in [16384, 32768] (the fused path up to 65536 // itemsize) that
#     exceeds the 8192-lane limit and the XPU backend miscompiles it (measured
#     max|err| 4.9 fp32 at (16, 16384) / 15.2 fp16 at (1024, 32768)).
#   * the vendor ``rms_norm`` tile2d kernel rounds ``(x * rrms)`` to the output
#     dtype *before* multiplying by w (double rounding); on fp16/bf16 that
#     exceeds the test rtol (fp16 1e-3 / bf16 0.016, measured 3.9e-3 / 3.1e-2),
#     and its [TILE_M, 8192] tile (admitted by the rms_norm 65536-element
#     budget) crashes the XPU vectorizer (llvm::cast<ClusterLayoutAttr> assert).
# The two kernels below mirror the validated vendor rms_norm structure (2D
# unmasked row-tile for N <= 4096; per-row chunked two-pass otherwise) but
# compute entirely in fp32 and round once at the store, matching the torch /
# TE reference exactly.
_FWD_TILE_N_MAX = 4096  # XPU vectorize miscompiles 2D tiles wider than 4096 cols
_FWD_TILE_ELEMS = 65536  # [TILE_M, N] fp32 tile element budget (proven on XPU)
_FWD_ROW_BLOCK = 64 * 128  # 8192: max lanes per 1D block (the >8192-lane defect)


@triton.jit
def _te_rmsnorm_fwd_tile2d_kernel(
    Y,  # output
    INV_RMS,  # per-row inverse rms
    X,  # input
    W,  # weight
    eps: tl.constexpr,
    TILE_M: tl.constexpr,  # rows per program (M % TILE_M == 0 guaranteed)
    N: tl.constexpr,  # number of columns (normalized dim), used as tile width
):
    # Unmasked [TILE_M, N] row-tile (M % TILE_M == 0), single-rounding:
    # y = (x * rrms * w) computed in fp32, rounded once at the store.
    pid = tl.program_id(0)

    n_off = tl.arange(0, N)
    w = tl.load(W + n_off).to(tl.float32)

    m_off = pid * TILE_M + tl.arange(0, TILE_M)
    offs = m_off[:, None] * N + n_off[None, :]

    x = tl.load(X + offs).to(tl.float32)

    var = tl.sum(x * x, axis=1) / N
    rrms = 1.0 / tl.sqrt(var + eps)

    y = (x * rrms[:, None] * w[None, :]).to(Y.dtype.element_ty)
    tl.store(Y + offs, y)
    tl.store(INV_RMS + m_off, rrms)


@triton.jit
def _te_rmsnorm_fwd_row_kernel(
    Y,  # output
    INV_RMS,  # per-row inverse rms
    X,  # input (one row per program)
    W,  # weight
    N,  # number of columns (normalized dim)
    eps,
    BLOCK: tl.constexpr,
    NEED_MASK: tl.constexpr,  # whether N is not a multiple of BLOCK
):
    # Per-row two-pass chunked kernel (N > _FWD_TILE_N_MAX or no TILE_M
    # candidate).  [BLOCK]-lane 1D chunks only (BLOCK <= 8192): the 2D
    # [*, BLOCK] tile form is what miscompiles on XPU for BLOCK > 4096.
    pid = tl.program_id(0)
    X += pid * N
    Y += pid * N

    # Pass 1: sum of squares (fp32 accumulate), chunks never exceed BLOCK.
    sum_sq = tl.zeros([BLOCK], dtype=tl.float32)
    for off in range(0, N, BLOCK):
        cols = off + tl.arange(0, BLOCK)
        if NEED_MASK:
            mask = cols < N
            x = tl.load(X + cols, mask=mask, other=0.0).to(tl.float32)
        else:
            x = tl.load(X + cols).to(tl.float32)
        sum_sq += x * x
    var = tl.sum(sum_sq, axis=0) / N
    rrms = 1.0 / tl.sqrt(var + eps)
    tl.store(INV_RMS + pid, rrms)

    # Pass 2: normalize and scale, single rounding at the store.
    for off in range(0, N, BLOCK):
        cols = off + tl.arange(0, BLOCK)
        if NEED_MASK:
            mask = cols < N
            x = tl.load(X + cols, mask=mask, other=0.0).to(tl.float32)
            w = tl.load(W + cols, mask=mask, other=0.0).to(tl.float32)
            y = (x * rrms * w).to(Y.dtype.element_ty)
            tl.store(Y + cols, y, mask=mask)
        else:
            x = tl.load(X + cols).to(tl.float32)
            w = tl.load(W + cols).to(tl.float32)
            y = (x * rrms * w).to(Y.dtype.element_ty)
            tl.store(Y + cols, y)


def _fwd_tile_m(N, M):
    """TILE_M for the unmasked 2D tile kernel, or None if not applicable.

    Mirrors rms_norm_forward's tile selection (TILE_M=32 preference; the
    [TILE_M, N] fp32 tile must stay within _FWD_TILE_ELEMS) but refuses tiles
    wider than _FWD_TILE_N_MAX columns: the [8, 8192] tile that the rms_norm
    65536-element budget admits when N == 8192 crashes the XPU vectorizer.
    """
    if N > _FWD_TILE_N_MAX:
        return None
    if N <= 256:
        for cand in (32, 16):
            if M % cand == 0:
                return cand
        return None
    tm = 32
    while tm * N > _FWD_TILE_ELEMS:
        tm //= 2
    while tm >= 2:
        if M % tm == 0:
            return tm
        tm //= 2
    return None


def te_rmsnorm_fwd(
    input: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    ln_out: torch.Tensor = None,
    quantizer=None,
    otype: torch.dtype = None,
    sm_margin: int = 0,
    zero_centered_gamma: bool = False,
):
    """TE-aligned RMSNorm forward (XPU backend local): y = x/sqrt(mean(x^2)+eps) * w,
    returns (y, None, rsigma).  See the module comment above for why the generic
    and rms_norm-reuse paths are not usable on this backend.
    """
    del sm_margin
    if zero_centered_gamma:
        # Reference adds in fp32 (weight.to(f32) + 1.0); adding in fp16/bf16
        # loses the low bits (1.0 + 0.05 rounds to 1.0 in bf16, ULP = 2^-8),
        # which exceeds the fp16/bf16 test rtol.
        weight = weight.to(torch.float32) + 1.0
    N = input.shape[-1]
    x = input.contiguous()
    M = x.numel() // N
    w = weight.contiguous()
    # empty_strided: `torch.empty` is intercepted by the gems empty op and
    # recompiled per call on this XPU (~95-100ms/call, see rms_norm fix).
    y = torch.empty_strided(x.size(), x.stride(), dtype=x.dtype, device=x.device)
    rsigma = torch.empty_strided((M,), (1,), dtype=torch.float32, device=x.device)

    with torch_device_fn.device(x.device):
        tile_m = _fwd_tile_m(N, M)
        if tile_m is not None:
            _te_rmsnorm_fwd_tile2d_kernel[(M // tile_m,)](
                y, rsigma, x, w, eps, tile_m, N
            )
        else:
            need_mask = (N % _FWD_ROW_BLOCK) != 0
            _te_rmsnorm_fwd_row_kernel[(M,)](
                y, rsigma, x, w, N, eps, _FWD_ROW_BLOCK, need_mask
            )

    if otype is not None and y.dtype != otype:
        y = y.to(otype)
    if ln_out is not None:
        ln_out.copy_(y)
        return ln_out, None, rsigma
    return y, None, rsigma


def te_rmsnorm_bwd(
    dz: torch.Tensor,
    x: torch.Tensor,
    rsigma: torch.Tensor,
    gamma: torch.Tensor,
    sm_margin: int = 0,
    zero_centered_gamma: bool = False,
):
    del sm_margin
    original_shape = x.shape
    N = gamma.shape[0]
    x_2d = x.reshape(-1, N).contiguous()
    dz_2d = dz.reshape(-1, N).contiguous()
    rsigma = rsigma.contiguous()
    M = x_2d.shape[0]
    dx = torch.empty_like(x_2d)
    dgamma = torch.empty_like(gamma)

    # dX: per-row 1D two-pass kernel (grid=(M,)).  Every load/store is a [C]
    # 1D block (block DMA); the 2D [R, C] tiled variant saturates at ~35-125
    # GB/s on this backend while per-row 1D reaches the raw memory ceiling
    # (~700 GB/s, measured on the layernorm backward rewrite).  C is capped at
    # 8192 lanes (the >8192-lane 1D defect, see _FWD_ROW_BLOCK) and the
    # N % C tail is masked.
    bc = _ln_bwd_col_size(N)

    # dgamma: two-stage (partial column sums, then 1D reduce) so the M-loop is
    # split across cdiv(M, BM) programs; single-program M-loop would serialize
    # all rows (measured ~2 orders of magnitude slower at (1024, 2048)).
    bm = _dgamma_bm_size(M)
    p = M // bm
    dgamma_partial = torch.empty((p, N), dtype=torch.float32, device=x.device)

    with torch_device_fn.device(x.device):
        _te_rmsnorm_bwd_dx_1d_kernel[(M,)](
            dx,
            dz_2d,
            x_2d,
            gamma,
            rsigma,
            N,
            zero_centered_gamma=zero_centered_gamma,
            C=bc,
            NEED_TAIL=(N % bc != 0),
            num_warps=4,
            isCloseUnrollControl=True,
            isCloseVectorization=True,
        )
        _te_rmsnorm_bwd_dgamma_kernel[(triton.cdiv(N, bc), p)](
            dgamma_partial,
            dz_2d,
            x_2d,
            rsigma,
            M,
            N,
            BM=bm,
            C=bc,
            num_warps=4,
            isCloseUnrollControl=True,
        )
        _te_rmsnorm_bwd_dgamma_reduce_kernel[(triton.cdiv(N, bc),)](
            dgamma,
            dgamma_partial,
            p,
            N,
            C=bc,
            num_warps=4,
            isCloseUnrollControl=True,
        )

    return dx.reshape(original_shape), dgamma


def _patch_generic_wrapper():
    """Route direct calls to the generic wrapper (flag_gems.ops.te_rmsnorm
    module) to this backend override.

    Tests and benchmarks import ``te_rmsnorm_bwd`` from
    ``flag_gems.ops.te_rmsnorm`` (bypassing the top-level ``flag_gems``
    registry that SpecOpRegistrar patches), so the generic Triton kernel
    would still be hit on XPU (it cannot compile there: the 2D
    ``tl.sum(..., axis=0)`` reduction in ``rmsnorm_bwd_dgamma_kernel`` is
    rejected by the XPU legalizer with
    ``axis must not be 0 for 2D+ shapes``).  Patching the module
    attribute at import time keeps the change backend-local: the generic
    module source is untouched and other vendor backends are unaffected
    (this module is only imported for the kunlunxin backend).
    """
    try:
        import sys

        _generic_module = sys.modules.get("flag_gems.ops.te_rmsnorm")
        if _generic_module is not None:
            if hasattr(_generic_module, "te_rmsnorm_bwd"):
                _generic_module.te_rmsnorm_bwd = te_rmsnorm_bwd
            if hasattr(_generic_module, "te_rmsnorm_fwd"):
                _generic_module.te_rmsnorm_fwd = te_rmsnorm_fwd
        # ``from .te_rmsnorm import ...`` in the ops package __init__ binds the
        # *generic* functions as the package attributes, so
        # ``flag_gems.ops.te_rmsnorm_{bwd,fwd}`` must be re-bound to this
        # backend implementation as well.
        import flag_gems.ops as _ops

        if hasattr(_ops, "te_rmsnorm_bwd"):
            _ops.te_rmsnorm_bwd = te_rmsnorm_bwd
        if hasattr(_ops, "te_rmsnorm_fwd"):
            _ops.te_rmsnorm_fwd = te_rmsnorm_fwd
    except ImportError:
        pass


_patch_generic_wrapper()


__all__ = ["te_rmsnorm_bwd", "te_rmsnorm_fwd"]