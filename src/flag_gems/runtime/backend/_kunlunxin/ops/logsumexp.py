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

import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as ext

logger = logging.getLogger(__name__)

# Redispatch key used to reach PyTorch's native (vendor) logsumexp. On this XPU
# the vendor's fused logsumexp kernel beats any Triton path we can express for a
# middle-dim (K>1) reduction (see the module docstring / solution doc), so the
# K>1 branch defers to it instead of materializing a slow transpose copy.
_FALLBACK_KEYSET = torch._C.DispatchKeySet(
    torch._C.DispatchKey.CompositeImplicitAutograd
)

# Inner-dim (K==1) reduction tiers:
#  - N <= _MULTIROW_MAX_N:   one multirow tile kernel (N constexpr, block DMA,
#    order-preserving uint32-key max). The uint32 key turns the XPU fp32
#    wide-row `tl.max` serial chain (~25x slower than `tl.sum`) into a fast
#    integer reduction (~4x).
#  - N >  _MULTIROW_MAX_N:   two-kernel chunk-split (single data read, single
#    exp per element): fast partials z_c = sum(exp(a)) per [TILE_R, BN] chunk
#    tile (no max, no max-shift -- avoids the expensive [TILE_R, BN] subtract
#    tile that spills on this XPU), then a tiny per-row combine
#    out = log(sum_c z_c). The max-shift is free to skip for ordinary inputs
#    (|x| <= ~80), so the slow (m, z) kernels below are kept as the guarded
#    fallback: after the fast kernels, any row with an out-of-range element
#    shows up as +-inf and is recomputed by the exact path. The host guard
#    costs one device sync, which is an amortized no-op on the multi-ms
#    big-N path it guards (the N <= 4096 multirow path stays exact so the
#    guard never hurts the launch-bound small shapes).
_MULTIROW_MAX_N = 4096
_CHUNK_BN = 4096


@libentry()
@triton.jit
def logsumexp_kernel_multirow(
    output_ptr,
    input_ptr,
    M,
    N: tl.constexpr,
    TILE_M: tl.constexpr,
    NEED_MASK: tl.constexpr,
):
    """Reduce the innermost dim N for many rows per program.

    Order-preserving uint32 key trick: float32 bits -> key = bits ^
    (0x80000000 | (bits >> 31)) is strictly increasing (radix sort family), so
    `tl.max(key, axis=1)` finds the per-row max on the fast integer reduction
    path. Decode with `bits = key ^ (0x80000000 | ((key >> 31) ^ 1))`.

    N is a constexpr so ``tl.arange(0, N)`` spans exactly [0, N) and the
    ``[TILE_M, N]`` tile is one stride-1 contiguous block -> block DMA on XPU
    (a runtime N would fall back to discrete gathers). Row masking is only
    compiled in when NEED_MASK, i.e. M % TILE_M != 0.
    """
    pid = ext.program_id(0)
    m_offsets = pid * TILE_M + tl.arange(0, TILE_M)
    n_offsets = tl.arange(0, N)
    m_mask = m_offsets < M
    offsets = m_offsets[:, None] * N + n_offsets[None, :]
    if NEED_MASK:
        inp = tl.load(
            input_ptr + offsets, mask=m_mask[:, None], other=-float("inf")
        ).to(tl.float32)
    else:
        inp = tl.load(input_ptr + offsets).to(tl.float32)
    bits = inp.to(tl.uint32, bitcast=True)
    key = bits ^ (tl.full([TILE_M, N], 0x80000000, tl.uint32) | (bits >> 31))
    m_key = tl.max(key, axis=1)
    bits_m = m_key ^ (tl.full([TILE_M], 0x80000000, tl.uint32) | ((m_key >> 31) ^ 1))
    m = bits_m.to(tl.float32, bitcast=True)
    safe_m = tl.where(m == float("-inf"), 0.0, m)
    z = tl.sum(tl.exp(inp - safe_m[:, None]), axis=1)
    # keep native semantics for special values: NaN -> NaN, +inf -> +inf,
    # all-(-inf) rows -> -inf.
    res = tl.where(
        m == float("-inf"), m, tl.where(m == float("inf"), m, safe_m + tl.log(z))
    )
    tl.store(output_ptr + m_offsets, res, mask=m_mask)


@libentry()
@triton.jit
def logsumexp_kernel_partial(
    mrow_ptr,
    zrow_ptr,
    input_ptr,
    R,
    BN: tl.constexpr,
    TILE_R: tl.constexpr,
    NEED_MASK: tl.constexpr,
):
    """Per-chunk partial (max, sum-exp) for a big innermost dim.

    Input is the flattened [rows * C, BN] view of the full 4096-chunks (each
    chunk stride-1 contiguous; BN constexpr keeps block DMA). No column
    masking -- the caller routes any tail (N % BN != 0) through the per-row
    kernel instead (masked-column reductions miscompute on this backend).
    Partial (m_c, z_c) pairs are stored compactly per chunk row; the host pads
    each row to TILE_C with -inf/0 so the combine kernel reads mask-free.
    """
    pid = ext.program_id(0)
    r_offsets = pid * TILE_R + tl.arange(0, TILE_R)
    r_mask = r_offsets < R
    n_offsets = tl.arange(0, BN)
    offsets = r_offsets[:, None] * BN + n_offsets[None, :]
    if NEED_MASK:
        a = tl.load(input_ptr + offsets, mask=r_mask[:, None], other=-float("inf")).to(
            tl.float32
        )
    else:
        a = tl.load(input_ptr + offsets).to(tl.float32)
    bits = a.to(tl.uint32, bitcast=True)
    key = bits ^ (tl.full([TILE_R, BN], 0x80000000, tl.uint32) | (bits >> 31))
    m_key = tl.max(key, axis=1)
    bits_m = m_key ^ (tl.full([TILE_R], 0x80000000, tl.uint32) | ((m_key >> 31) ^ 1))
    m = bits_m.to(tl.float32, bitcast=True)
    safe_m = tl.where(m == float("-inf"), 0.0, m)
    z = tl.sum(tl.exp(a - safe_m[:, None]), axis=1)
    tl.store(mrow_ptr + r_offsets, m, mask=r_mask)
    tl.store(zrow_ptr + r_offsets, z, mask=r_mask)


@libentry()
@triton.jit
def logsumexp_kernel_combine(
    output_ptr,
    mrow_ptr,
    zrow_ptr,
    mtail_ptr,
    ztail_ptr,
    M,
    C_FULL: tl.constexpr,
    HAS_TAIL: tl.constexpr,
    TILE_C: tl.constexpr,
):
    """Combine the C_FULL per-chunk partials of one row plus (optionally) the
    tail partial at slot C_FULL: out = m + log(sum zc exp(mc - m))."""
    row = ext.program_id(0)
    c_offsets = tl.arange(0, TILE_C)
    mc = tl.load(mrow_ptr + row * TILE_C + c_offsets)
    zc = tl.load(zrow_ptr + row * TILE_C + c_offsets)
    if HAS_TAIL:
        m_t = tl.load(mtail_ptr + row)
        z_t = tl.load(ztail_ptr + row)
        is_tail = c_offsets == C_FULL
        mc = tl.where(is_tail, m_t, mc)
        zc = tl.where(is_tail, z_t, zc)
    m = tl.max(mc, axis=0)
    safe_m = tl.where(m == float("-inf"), 0.0, m)
    # exp(mc - safe_m) is 0 for the -inf pad chunks; zc is 0 there too.
    z = tl.sum(zc * tl.exp(mc - safe_m), axis=0)
    res = tl.where(
        m == float("-inf"), m, tl.where(m == float("inf"), m, safe_m + tl.log(z))
    )
    tl.store(output_ptr + row, res)


@libentry()
@triton.jit
def logsumexp_kernel_fast_partial(
    zrow_ptr,
    input_ptr,
    R,
    BN: tl.constexpr,
    TILE_R: tl.constexpr,
    NEED_MASK: tl.constexpr,
):
    """Fast per-chunk partial z_c = sum(exp(a)) for a big innermost dim.

    Unlike ``logsumexp_kernel_partial`` there is no max and no max-shift: the
    subtract on a [TILE_R, BN] tile is what spills registers on this XPU, and
    for |x| <= ~80 the unshifted sum cannot overflow (a [TILE_R, BN] chunk has
    at most BN=4096 elements, so z_c < 4096 * e^80 < 3.4e38). Out-of-range
    rows are caught by the host-side guard in ``_reduce_inner`` and rerun
    through the exact (m, z) path. -inf rows give z_c = 0 and +inf rows give
    z_c = +inf, both of which stay exact through the combine's log.
    """
    pid = ext.program_id(0)
    r_offsets = pid * TILE_R + tl.arange(0, TILE_R)
    r_mask = r_offsets < R
    n_offsets = tl.arange(0, BN)
    offsets = r_offsets[:, None] * BN + n_offsets[None, :]
    if NEED_MASK:
        a = tl.load(input_ptr + offsets, mask=r_mask[:, None], other=-float("inf")).to(
            tl.float32
        )
    else:
        a = tl.load(input_ptr + offsets).to(tl.float32)
    z = tl.sum(tl.exp(a), axis=1)
    tl.store(zrow_ptr + r_offsets, z, mask=r_mask)


@libentry()
@triton.jit
def logsumexp_kernel_fast_combine(output_ptr, zrow_ptr, M, TILE_C: tl.constexpr):
    """Fast per-row combine: out = log(sum_c z_c). The zrow buffer is padded
    with 0 slots (pad slots contribute 0 to the sum)."""
    row = ext.program_id(0)
    c_offsets = tl.arange(0, TILE_C)
    zc = tl.load(zrow_ptr + row * TILE_C + c_offsets)
    tl.store(output_ptr + row, tl.log(tl.sum(zc, axis=0)))


@libentry()
@triton.jit
def logsumexp_kernel_tail_partials(
    mrow_ptr,
    zrow_ptr,
    input_ptr,
    M,
    ROW_STRIDE,
    N,
    TILE_N: tl.constexpr,
):
    """Per-row (m, z) partials for a tail slice [M, N] strided by ROW_STRIDE.

    Tail widths are < _CHUNK_BN (<= 4096), so TILE_N is power-of-two and the
    loop body executes once; the single masked iteration is verified exact on
    this backend (unlike padded 2D-tile masked reductions). Emits compact
    per-row max m and max-shifted sum z for the combine kernel.
    """
    pid = ext.program_id(0)
    m = tl.full([TILE_N], value=float("-inf"), dtype=tl.float32)
    z = tl.full([TILE_N], value=0.0, dtype=tl.float32)
    input_ptr += pid * ROW_STRIDE

    for start_n in range(0, N, TILE_N):
        n_offsets = start_n + tl.arange(0, TILE_N)
        mask = n_offsets < N
        a = tl.load(input_ptr + n_offsets, mask=mask, other=-float("inf")).to(
            tl.float32
        )
        m_new = tl.maximum(m, a)
        all_neg_inf = m_new == float("-inf")
        z = tl.where(all_neg_inf, z, z * tl.exp(m - m_new) + tl.exp(a - m_new))
        m = m_new

    m_r = tl.max(m, axis=0)
    z_r = tl.sum(z * tl.exp(m - m_r), axis=0)
    # all-(-inf) tails must contribute z=0 to the combine (exp(-inf - -inf)
    # would be NaN), and all-(-inf) rows are resolved by the combine's -inf
    # guard.
    tl.store(mrow_ptr + pid, m_r)
    tl.store(zrow_ptr + pid, tl.where(m_r == float("-inf"), 0.0, z_r))


def _reduce_inner_small(inp, rows, N, out):
    """Single-tile multirow kernel for N <= _MULTIROW_MAX_N (exact).

    TILE_M=32 for N > 64: the [32, N] tile is the measured sweet spot on this
    XPU (16.3us vs 18.2us/14.1us for 64/16 on [256,256] f32; ~-2% on
    [1024,1024]; ~+2% on [4096,4096]) -- the previous 64/16 split was tuned
    for the N<=64 launch-bound tier only. The N <= 64 tier keeps 16 (a single
    small tile per program; 32 would waste partial rows)."""
    if N <= 64:
        TILE_M = 16
    else:
        TILE_M = 32
    need_mask = 1 if rows % TILE_M else 0
    grid = (triton.cdiv(rows, TILE_M), 1, 1)
    logsumexp_kernel_multirow[grid](
        out,
        inp,
        rows,
        N=N,
        TILE_M=TILE_M,
        NEED_MASK=need_mask,
        num_warps=8,
        buffer_size_limit=2048,
    )


def _reduce_tail_partials(mrow, zrow, inp, rows, row_stride, tail_n):
    """Reduce a [rows, tail_n] tail-view (strided by row_stride) into compact
    (m, z) partials via the per-row online kernel."""
    TILE_N = max(1, triton.next_power_of_2(tail_n))
    grid = (rows, 1, 1)
    logsumexp_kernel_tail_partials[grid](
        mrow,
        zrow,
        inp,
        rows,
        row_stride,
        tail_n,
        TILE_N=TILE_N,
        num_warps=4,
        buffer_size_limit=2048,
    )


def _reduce_inner_chunk_slow(inp, rows, N):
    """Exact chunk-split path (single data read, single exp per element):
    per-chunk (m_c, z_c) partials via the tile kernel, then a tiny per-row
    combine. Any tail (N % 4096 != 0) is reduced by the per-row online kernel
    over a tail-slice view (masked-tail reductions miscompute on this
    backend). Used as-is for N % 4096 != 0 and as the guarded fallback of the
    fast path below."""
    out = torch.empty((rows,), dtype=torch.float32, device=inp.device)
    BN = _CHUNK_BN
    C_full = N // BN
    TAIL = N - C_full * BN
    TILE_C = max(1, triton.next_power_of_2(C_full + (1 if TAIL else 0)))
    # partials compact per chunk; then per-row padded to TILE_C with
    # (-inf, 0) pad slots so the combine kernel reads mask-free.
    mrow = torch.empty((rows * C_full,), dtype=torch.float32, device=inp.device)
    zrow = torch.empty_like(mrow)
    if C_full:
        R = rows * C_full
        TILE_R = 32
        need_mask = 1 if R % TILE_R else 0
        full_view = torch.ops.aten.slice(inp, 1, 0, C_full * BN)
        # reshape may copy only when the slice is non-contiguous (tail
        # cases with N % BN != 0); the aligned path is a null-op view.
        flat = torch.ops.aten.reshape(full_view, (R, BN))
        grid = (triton.cdiv(R, TILE_R), 1, 1)
        logsumexp_kernel_partial[grid](
            mrow,
            zrow,
            flat,
            R,
            BN=BN,
            TILE_R=TILE_R,
            NEED_MASK=need_mask,
            num_warps=4,
            buffer_size_limit=2048,
        )
    if C_full and TILE_C != C_full:
        mrow = mrow.view(rows, C_full)
        zrow = zrow.view(rows, C_full)
        mp = torch.full(
            (rows, TILE_C), -float("inf"), dtype=torch.float32, device=inp.device
        )
        zp = torch.zeros((rows, TILE_C), dtype=torch.float32, device=inp.device)
        mp[:, :C_full] = mrow
        zp[:, :C_full] = zrow
        mrow = mp
        zrow = zp
    elif not C_full:
        mrow = torch.full(
            (rows, TILE_C), -float("inf"), dtype=torch.float32, device=inp.device
        )
        zrow = torch.zeros((rows, TILE_C), dtype=torch.float32, device=inp.device)
    if TAIL:
        # tail slice view: [rows, TAIL] strided by N (no copy)
        tail_view = torch.ops.aten.slice(inp, 1, C_full * BN, N)
        mtail = torch.empty((rows,), dtype=torch.float32, device=inp.device)
        ztail = torch.empty_like(mtail)
        _reduce_tail_partials(mtail, ztail, tail_view, rows, N, TAIL)
    else:
        # unused sentinel pointer for the HAS_TAIL=0 build
        mtail = torch.empty((1,), dtype=torch.float32, device=inp.device)
        ztail = torch.empty_like(mtail)
    logsumexp_kernel_combine[(rows, 1, 1)](
        out,
        mrow,
        zrow,
        mtail,
        ztail,
        rows,
        C_FULL=C_full,
        HAS_TAIL=1 if TAIL else 0,
        TILE_C=TILE_C,
        num_warps=4,
        buffer_size_limit=2048,
    )
    return out


def _reduce_inner_chunk_fast(inp, rows, N):
    """Fast chunk-split path for N % _CHUNK_BN == 0: unshifted per-chunk
    partials z_c = sum(exp(a)) + out = log(sum_c z_c). No per-element subtract
    tile (it spills on this XPU) and no per-element max. Exact for |x| <= ~80
    (a [TILE_R, BN] partial cannot overflow: z_c < BN * e^80 < 3.4e38) and for
    the +-inf/NaN specials; out-of-range rows surface as +-inf and are
    corrected by the host guard in ``_reduce_inner``."""
    BN = _CHUNK_BN
    C_full = N // BN
    TILE_C = max(1, triton.next_power_of_2(C_full))
    R = rows * C_full
    zrow = torch.empty((R,), dtype=torch.float32, device=inp.device)
    TILE_R = 4
    need_mask = 1 if R % TILE_R else 0
    flat = torch.ops.aten.reshape(inp, (R, BN))
    grid = (triton.cdiv(R, TILE_R), 1, 1)
    logsumexp_kernel_fast_partial[grid](
        zrow,
        flat,
        R,
        BN=BN,
        TILE_R=TILE_R,
        NEED_MASK=need_mask,
        num_warps=4,
        buffer_size_limit=2048,
    )
    # pad to TILE_C with 0 slots (they contribute 0 to the combine sum)
    zrow_p = torch.zeros((rows, TILE_C), dtype=torch.float32, device=inp.device)
    zrow_p[:, :C_full] = zrow.view(rows, C_full)
    out = torch.empty((rows,), dtype=torch.float32, device=inp.device)
    logsumexp_kernel_fast_combine[(rows, 1, 1)](
        out, zrow_p, rows, TILE_C=TILE_C, num_warps=4, buffer_size_limit=2048
    )
    return out


def _reduce_inner(inp, rows, N):
    """logsumexp over the innermost dim N of a contiguous [rows, N] tensor."""
    out = torch.empty((rows,), dtype=inp.dtype, device=inp.device)
    if N <= _MULTIROW_MAX_N:
        _reduce_inner_small(inp, rows, N, out)
    elif N % _CHUNK_BN == 0:
        # Fast aligned chunk path (no max-shift). The host guard costs one
        # device sync, amortized over the multi-ms big-N read; on overflow or
        # underflow (|x| > ~80) it reroutes to the exact path below.
        fast = _reduce_inner_chunk_fast(inp, rows, N)
        if torch.any(torch.isinf(fast)):
            fast.copy_(_reduce_inner_chunk_slow(inp, rows, N))
        out.copy_(fast)
    else:
        out.copy_(_reduce_inner_chunk_slow(inp, rows, N))
    return out


def _native_logsumexp(inp, dim, keepdim):
    """Reach PyTorch's native (vendor) logsumexp, bypassing the gems override."""
    return torch.ops.aten.logsumexp.default.redispatch(
        _FALLBACK_KEYSET, inp, dim, keepdim
    )


def logsumexp(inp, dim, keepdim=False):
    logger.debug("GEMS_KUNLUNXIN LOGSUMEXP")

    if isinstance(dim, (list, tuple)):
        if len(dim) == 0:
            # Empty dim list means no reduction, just return the input.
            return inp.clone()
        if len(dim) != 1:
            # Multi-dim reduction: the vendor's native kernel beats a sequence
            # of Triton reductions on this XPU.
            return _native_logsumexp(inp, list(dim), keepdim)
        dim = dim[0]

    assert dim >= -inp.ndim and dim < inp.ndim, "Invalid dim"
    dim = dim % inp.ndim

    N = inp.shape[dim]
    K = 1
    for i in range(dim + 1, inp.ndim):
        K *= inp.shape[i]

    # Middle-dim reduction (K > 1) or a size-1 reduction: defer to the native
    # vendor kernel. A Triton middle reduction on XPU is a dead end -- a physical
    # transpose+contiguous can't reach the vendor's fast copy once gems overrides
    # copy_, and a direct strided/discrete reduction either overflows uni_sram or
    # mis-computes (2D axis=0 reduce here). N==1 is a trivial identity that
    # the native kernel does faster than a gems copy.
    if K > 1 or N == 1:
        return _native_logsumexp(inp, [dim], keepdim)

    # K == 1: innermost-dim reduction -> fast contiguous Triton kernels.
    M = 1
    for i in range(dim):
        M *= inp.shape[i]
    inp = inp.contiguous()
    shape = list(inp.shape)
    shape[dim] = 1

    with torch_device_fn.device(inp.device):
        out = _reduce_inner(inp, M, N).view(shape)

    if not keepdim:
        out = out.squeeze(dim=dim)
    return out
