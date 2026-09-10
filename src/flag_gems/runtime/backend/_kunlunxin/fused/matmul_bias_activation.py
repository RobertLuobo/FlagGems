# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Kunlunxin(XPU) backend override for the fused matmul+bias+ReLU operator.
#
# The generic `flag_gems.fused.matmul_bias_activation` kernel (BLOCK_K=32,
# 1D bias broadcast `bias[None, :]` + tl.where epilogue) fails to lower on
# XPU inside `ConvertTritonSDNNToLLVM` (compile error, all shapes/dtypes
# fail), and even where it compiles the 1D-bias broadcast epilogue is ~100x
# slower than a 2D bias tile load (120ms vs 0.88ms on 4096^2 fp16, measured).
#
# This file reuses the structure proven in `_kunlunxin/ops/mm.py` (same
# backend, 2026-09-02, XPU 3/4): an *aligned* fast kernel (unmasked loads,
# tl.max_contiguous/tl.multiple_of hints, host-padded safe path for ragged
# shapes) plus a *safe* kernel that relies on host-side K-/C-padding so that
# no load/store address can leave its allocation (TritonXPU mis-lowers
# masked loads/stores whose addresses leave the allocation).
#
# Single-pass design (measured, XPU 6, 2026-08-13..09-09 probes):
#   * epilogue = 2D bias tile load + add + `tl.maximum(acc, 0.0)`.  A ReLU
#     written as tl.where(acc>0,acc,0) on the fp32 dot result compiles but is
#     40-100x slower (serialised layout path, 33-122ms); tl.maximum is fused
#     by `tritonsdnn-fuse-relu-activation` (0.875ms on 4096^2 fp16, only
#     ~0.23ms over the mm-only kernel).
#   * 1D-bias broadcast (`bias[None, :]` with inner stride 0) is ~100x slower
#     than a full 2D-tile load; the bias is therefore broadcast to the (M, N)
#     output shape on the host (a stride-(0, 1) view, no copy) and loaded as
#     a 2D block.
#   * BLOCK_SIZE_M/N = 128 for M/N <= 512 (launch-bound shapes), else 256;
#     BLOCK_SIZE_K = 128 for fp32 (BK=256 on fp32 silently returns garbage,
#     maxerr ~150) and for small shapes, 256 otherwise; warps 4/8; stages 2
#     (3 for bf16); GROUP_M=8 L2 swizzle.
#   * bf16 with K >= 2048 and M, N >= 128 runs with XMLIR_MATMUL_FAST_MODE=1
#     (the vendor mm lowering toggle, see ops/mm.py `_set_matmul_fast_mode`:
#     4096^3 bf16 1.32 -> 0.81ms, small bf16 shapes regress so it is gated).
#
# Ragged shapes (M % blk_m != 0 or N % blk_n != 0 or K % blk_k != 0, or a
# non-row-major / non-contiguous input or output) take the safe path: the
# host fully zero-pads A -> (cdiv(M,blk_m)*blk_m, cdiv(K,blk_k)*blk_k),
# B -> (cdiv(K,blk_k)*blk_k, cdiv(N,blk_n)*blk_n) and the bias ->
# (cdiv(M,blk_m)*blk_m, cdiv(N,blk_n)*blk_n), runs the kernel on the padded
# extents (so the `% M`/`% N` wraps are the identity and every 2D load/store
# is a full-rank in-bounds block: the bf16 vectorizer mis-compiles collapsed
# 2D loads whose rows/cols overlap) and copies the (M, N) view back through
# the native `_copy_from` engine.  The masked-load alternative is rejected:
# TritonXPU does not honour a masked load whose addresses leave the
# allocation.

import logging
import os

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import broadcastable_to, libentry
from flag_gems.utils import triton_lang_extension as ext

logger = logging.getLogger(__name__)

_FAST_MODE_ENV = "XMLIR_MATMUL_FAST_MODE"

# Same opt-in as ops/mm.py / ops/addmm.py: the generate_configs autotune path
# is not numerically safe on this backend (grossly wrong results on a large
# subset of generated tiles; see ops/addmm.py).  The aligned kernel below uses
# a fixed _block_* heuristic set instead; KLX_USE_AUTOTUNE=1 additionally
# disables the aligned path (identical to ops/mm.py).
KLX_USE_AUTOTUNE = os.environ.get("KLX_USE_AUTOTUNE", "0") == "1"


def _block_m(M):
    return 128 if M <= 512 else 256


def _block_n(N):
    return 128 if N <= 512 else 256


def _block_k(M, N, dtype):
    # fp32 BK=256 returns silently wrong values on this backend (maxerr ~150
    # on 4096^3) and has been measured to collapse (4.7ms vs 1.46ms at
    # BK=128); small shapes are launch-bound and prefer the 128-tile.  Only
    # large fp16/bf16 GEMMs benefit from BK=256 (fp16 4096^3: 1.01ms at
    # BK=128 vs 0.68ms at BK=256).
    if dtype == torch.float32 or (M <= 512 and N <= 512):
        return 128
    return 256


def _block_warps(M, N):
    return 4 if (M <= 512 and N <= 512) else 8


# --------------------------------------------------------------------------
# Aligned fast path: every (BLOCK_M, BLOCK_N, BLOCK_K) tile is complete, so
# the contiguity hints are truthful (they enable the backend block loads,
# ~1.2-1.7x on small tiles), no mask is needed and no address can leave the
# (M, N) / (M, K) / (K, N) allocations.  Both kernels are launched with the
# exact _block_* tiles the wrapper uses for the alignment gate / padding so
# the two can never disagree (a shape-dependent autotune/heuristic BLOCK
# would silently leave the padded allocations; see ops/mm.py).
# --------------------------------------------------------------------------
@libentry()
@triton.jit
def _matmul_bias_activation_aligned_kernel(
    a_ptr,
    w_ptr,
    b_ptr,
    c_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_wk,
    stride_wn,
    stride_bm,
    stride_bn,
    stride_cm,
    stride_cn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid = ext.program_id(0)
    grid_m = tl.cdiv(M, BLOCK_SIZE_M)
    grid_n = tl.cdiv(N, BLOCK_SIZE_N)
    # re-order program ID for better L2 reuse along the N dimension
    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // (group_size)

    rm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    rn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    ram = tl.max_contiguous(tl.multiple_of(rm, BLOCK_SIZE_M), BLOCK_SIZE_M)
    rbn = tl.max_contiguous(tl.multiple_of(rn, BLOCK_SIZE_N), BLOCK_SIZE_N)
    rk = tl.arange(0, BLOCK_SIZE_K)

    a_ptrs = a_ptr + (ram[:, None] * stride_am + rk[None, :] * stride_ak)
    w_ptrs = w_ptr + (rk[:, None] * stride_wk + rbn[None, :] * stride_wn)

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for _k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(a_ptrs)
        w = tl.load(w_ptrs)
        accumulator += tl.dot(a, w, allow_tf32=False)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        w_ptrs += BLOCK_SIZE_K * stride_wk

    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    # 2D bias tile load (the host passes a stride-(0, 1) broadcast view);
    # a 1D `bias[None, :]` broadcast epilogue is ~100x slower on this
    # backend, and the tl.where form of ReLU 40-100x slower still.
    b_ptrs = b_ptr + stride_bm * offs_cm[:, None] + stride_bn * offs_cn[None, :]
    bias = tl.load(b_ptrs)
    accumulator += bias
    accumulator = tl.maximum(accumulator, 0.0)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    tl.store(c_ptrs, accumulator.to(c_ptr.dtype.element_ty))


# --------------------------------------------------------------------------
# Safe path for ragged / non-row-major shapes.  The host hands in K-padded
# A (M, kp), B (kp, N) with kp % BLOCK_SIZE_K == 0 and a C buffer of
# cdiv(M, BLOCK_SIZE_M)*BLOCK_SIZE_M x cdiv(N, BLOCK_SIZE_N)*BLOCK_SIZE_N;
# loads wrap rows/cols through % (so the partial last tile stays inside the
# real extents) and no load/store mask is used: TritonXPU mis-lowers masked
# loads/stores whose addresses leave the allocation (see ops/mm.py).
# --------------------------------------------------------------------------
@libentry()
@triton.jit
def _matmul_bias_activation_safe_kernel(
    a_ptr,
    w_ptr,
    b_ptr,
    c_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_wk,
    stride_wn,
    stride_bm,
    stride_bn,
    stride_cm,
    stride_cn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid = ext.program_id(0)
    grid_m = tl.cdiv(M, BLOCK_SIZE_M)
    grid_n = tl.cdiv(N, BLOCK_SIZE_N)
    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // (group_size)

    rm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    rn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    ram = rm % M
    rbn = rn % N
    rk = tl.arange(0, BLOCK_SIZE_K)

    a_ptrs = a_ptr + (ram[:, None] * stride_am + rk[None, :] * stride_ak)
    w_ptrs = w_ptr + (rk[:, None] * stride_wk + rbn[None, :] * stride_wn)

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for _k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(a_ptrs)
        w = tl.load(w_ptrs)
        accumulator += tl.dot(a, w, allow_tf32=False)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        w_ptrs += BLOCK_SIZE_K * stride_wk

    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    # Bias is the (M, N) stride-(0, 1) broadcast view of an (N,)-storage
    # tensor; the column wrap keeps the largest column index at N-1 so the
    # unmasked 2D load stays inside the bias storage even though the C tile
    # is padded to N_pad.
    b_ptrs = b_ptr + stride_bm * offs_cm[:, None] + stride_bn * (offs_cn % N)[None, :]
    bias = tl.load(b_ptrs)
    accumulator += bias
    accumulator = tl.maximum(accumulator, 0.0)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    tl.store(c_ptrs, accumulator.to(c_ptr.dtype.element_ty))


def _set_matmul_fast_mode(a_dtype, M, N, K):
    """bf16 large-K GEMMs run faster under XMLIR_MATMUL_FAST_MODE=1 (the
    mangling of ops/mm.py `_set_matmul_fast_mode`): 4096^3 1.32 -> 0.81ms.
    Small bf16 shapes regress and fp16/fp32 are unaffected, so the flag is
    gated on bf16 only with K >= 2048 and M, N >= 128.  Returns the saved
    value (or None when the flag was not set)."""
    if a_dtype == torch.bfloat16 and K >= 2048 and M >= 128 and N >= 128:
        saved = os.environ.get(_FAST_MODE_ENV)
        os.environ[_FAST_MODE_ENV] = "1"
        return saved
    return None


def _restore_matmul_fast_mode(saved):
    if saved is None:
        os.environ.pop(_FAST_MODE_ENV, None)
    else:
        os.environ[_FAST_MODE_ENV] = saved


def _num_stages(dtype):
    # bf16 prefers pipelining depth 3 (measured: 1024^3 0.874 -> 0.843ms at
    # stages=3 with fast-mode), fp16/fp32 stay at the backend default 2.
    return 3 if dtype == torch.bfloat16 else 2


def _pad_full(a, b, bias, M, K, N, mp, np_, kp, device, dtype):
    """Materialize A, B and the bias into fully tile-padded buffers.

    The safe kernel issues unmasked (BLOCK_M, BLOCK_K) / (BLOCK_K, BLOCK_N)
    2D loads; when a real dimension is smaller than a tile (M=1 or N=1) the
    ``% M`` / ``% N`` wraps collapse whole tiles onto a single row/column and
    the bf16 vectorizer mis-compiles the collapsed load (GEP 2x scale reads
    past the allocation: measured illegal memory access on a (1, 1, 32) bf16
    fused call).  Padding A to (mp, kp), B to (kp, np_) and the bias to
    (mp, np_) keeps every 2D access full-rank and in-bounds by construction;
    the extra rows/cols hold zeros, and only c[0:M, 0:N] is copied back, so
    the padding cannot leak into the result.  Copies go through the native
    ``_copy_from`` engine (gems does not override it); ``x.contiguous()``
    must not be used because the registered ``_to_copy`` override
    mis-handles strided sources (see ops/mm.py).  ``narrow`` (registered
    zero-copy as_strided view) is used instead of a python ``[:, :K]``
    slice, which dispatches slice.Tensor without ``step`` under use_gems().
    """
    ap = torch.zeros((mp, kp), device=device, dtype=dtype)
    torch.ops.aten._copy_from(a, ap.narrow(0, 0, M).narrow(1, 0, K), False)
    bp = torch.zeros((kp, np_), device=device, dtype=dtype)
    torch.ops.aten._copy_from(b, bp.narrow(0, 0, K).narrow(1, 0, N), False)
    bias_pad = torch.zeros((mp, np_), device=device, dtype=dtype)
    torch.ops.aten._copy_from(bias, bias_pad.narrow(0, 0, M).narrow(1, 0, N), False)
    return ap, bp, bias_pad


def matmul_bias_activation(input, weight, bias):
    """
    Fused matmul + bias + ReLU activation.

    Args:
        input: Input tensor of shape (M, K)
        weight: Weight matrix of shape (K, N)
        bias: Bias vector of shape (N,) or (1, N)

    Returns:
        Output tensor of shape (M, N) with ReLU activation applied
    """
    logger.debug("GEMS_KUNLUNXIN MATMUL_BIAS_ACTIVATION")
    assert input.shape[1] == weight.shape[0], "Incompatible dimensions"
    assert broadcastable_to(
        bias.shape, (input.shape[0], weight.shape[1])
    ), "Incompatible input shape"
    M, K = input.shape
    _, N = weight.shape

    # NOTE: no ``x.contiguous()`` here - inside use_gems()/enable() a strided
    # input dispatches through the registered ``_to_copy`` override, whose
    # flat-1D kernel mis-handles non-contiguous strides (see ops/mm.py).
    # Both kernels take runtime strides; the safe path pads through the
    # native ``_copy_from`` engine, which is not overridden.
    if bias.dim() > 1:
        bias = bias.reshape(-1)
    out = torch.empty((M, N), device=input.device, dtype=input.dtype)
    # stride-(0, 1) broadcast view; loader sees a full 2D tile (no copy).
    bias = bias.broadcast_to(out.shape)

    blk_m = _block_m(M)
    blk_n = _block_n(N)
    blk_k = _block_k(M, N, input.dtype)
    stages = _num_stages(input.dtype)
    saved = _set_matmul_fast_mode(input.dtype, M, N, K)
    try:
        with torch_device_fn.device(input.device):
            grid = lambda META: (
                triton.cdiv(M, META["BLOCK_SIZE_M"])
                * triton.cdiv(N, META["BLOCK_SIZE_N"]),
            )
            if (
                not KLX_USE_AUTOTUNE
                and (input.stride(0), input.stride(1)) == (K, 1)
                and (weight.stride(0), weight.stride(1)) == (N, 1)
                and M % blk_m == 0
                and N % blk_n == 0
                and K % blk_k == 0
                and out.stride(1) == 1
            ):
                # Every tile is complete: no mask, truthful hints.  The
                # launched BLOCK_* are exactly the _block_* values used by
                # the gate above.
                _matmul_bias_activation_aligned_kernel[grid](
                    input,
                    weight,
                    bias,
                    out,
                    M,
                    N,
                    K,
                    input.stride(0),
                    input.stride(1),
                    weight.stride(0),
                    weight.stride(1),
                    bias.stride(0),
                    bias.stride(1),
                    out.stride(0),
                    out.stride(1),
                    BLOCK_SIZE_M=blk_m,
                    BLOCK_SIZE_N=blk_n,
                    BLOCK_SIZE_K=blk_k,
                    GROUP_M=8,
                    num_warps=_block_warps(M, N),
                    num_stages=stages,
                )
            else:
                kp = triton.cdiv(K, blk_k) * blk_k
                mp = triton.cdiv(M, blk_m) * blk_m
                np_ = triton.cdiv(N, blk_n) * blk_n
                a_pad, w_pad, bias_pad = _pad_full(
                    input,
                    weight,
                    bias,
                    M,
                    K,
                    N,
                    mp,
                    np_,
                    kp,
                    input.device,
                    input.dtype,
                )
                c = torch.empty((mp, np_), device=input.device, dtype=input.dtype)
                # Launched on the padded extents (M'=mp, N'=np_, K'=kp): the
                # % wraps are the identity and every 2D load/store stays
                # inside the fully padded allocations.
                _matmul_bias_activation_safe_kernel[grid](
                    a_pad,
                    w_pad,
                    bias_pad,
                    c,
                    mp,
                    np_,
                    kp,
                    a_pad.stride(0),
                    a_pad.stride(1),
                    w_pad.stride(0),
                    w_pad.stride(1),
                    bias_pad.stride(0),
                    bias_pad.stride(1),
                    c.stride(0),
                    c.stride(1),
                    BLOCK_SIZE_M=blk_m,
                    BLOCK_SIZE_N=blk_n,
                    BLOCK_SIZE_K=blk_k,
                    GROUP_M=8,
                    num_warps=_block_warps(M, N),
                    num_stages=stages,
                )
                # ``narrow`` instead of python ``[:M, :N]`` (dispatches
                # slice.Tensor without ``step`` under use_gems()); the copy
                # goes through the native ``_copy_from`` engine.
                torch.ops.aten._copy_from(c.narrow(0, 0, M).narrow(1, 0, N), out, False)
    finally:
        _restore_matmul_fast_mode(saved)
    return out
