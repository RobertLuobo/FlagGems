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

# Tile budget (in elements) for the multirow constexpr-N kernel. A [TILE_M, N]
# fp32 tile stays within this budget so it fits XPU uni_sram. N values above
# this fall back to the per-row 1D-loop kernel.
_MULTIROW_BUDGET = 32768

# Redispatch key used to reach PyTorch's native (vendor) logsumexp. On this XPU
# the vendor's fused logsumexp kernel beats any Triton path we can express for a
# middle-dim (K>1) reduction (see the module docstring / solution doc), so the
# K>1 branch defers to it instead of materializing a slow transpose copy.
_FALLBACK_KEYSET = torch._C.DispatchKeySet(
    torch._C.DispatchKey.CompositeImplicitAutograd
)


@libentry()
@triton.jit
def logsumexp_kernel_multirow(
    output_ptr,
    input_ptr,
    M,
    N: tl.constexpr,
    TILE_M: tl.constexpr,
):
    """Reduce the innermost dim N for many rows per program.

    N is a constexpr so ``tl.arange(0, N)`` spans exactly [0, N): the
    ``[TILE_M, N]`` tile is one stride-1 contiguous block -> block DMA on XPU
    (vs. the discrete access a runtime N produces). One program handles TILE_M
    rows, amortizing launch overhead for large M.
    """
    pid = ext.program_id(0)
    m_offsets = pid * TILE_M + tl.arange(0, TILE_M)
    n_offsets = tl.arange(0, N)
    m_mask = m_offsets < M
    offsets = m_offsets[:, None] * N + n_offsets[None, :]
    inp = tl.load(input_ptr + offsets, mask=m_mask[:, None], other=-float("inf")).to(
        tl.float32
    )
    m = tl.max(inp, axis=1)
    safe_m = tl.where(m == float("-inf"), 0.0, m)
    z = tl.sum(tl.exp(inp - safe_m[:, None]), axis=1)
    out = tl.where(m == float("-inf"), m, safe_m + tl.log(z))
    tl.store(output_ptr + m_offsets, out, mask=m_mask)


@libentry()
@triton.jit
def logsumexp_kernel_inner(
    output_ptr,
    input_ptr,
    M,
    N,
    TILE_N: tl.constexpr,
):
    """Per-row 1D-loop kernel, used as the fallback for large N (N > budget)."""
    pid_m = ext.program_id(0)
    m = tl.full([TILE_N], value=float("-inf"), dtype=tl.float32)
    z = tl.full([TILE_N], value=0.0, dtype=tl.float32)
    input_ptr += pid_m * N

    for start_n in range(0, N, TILE_N):
        n_offsets = start_n + tl.arange(0, TILE_N)
        mask = n_offsets < N
        inp = tl.load(input_ptr + n_offsets, mask=mask, other=-float("inf")).to(
            tl.float32
        )
        m_new = tl.maximum(m, a)
        z = z * tl.exp(m - m_new) + tl.exp(a - m_new)
        # Keep z at 0 only for lanes still all-(-inf). The previous guard
        # `m_new == -inf` also swallowed a NaN element (IEEE max ignores
        # NaNs, so maximum(m, NaN) == m), which must instead flow into z and
        # make the row NaN.
        z = tl.where((m == float("-inf")) & (a == float("-inf")), 0.0, z)
        m = m_new

    m_r = tl.max(m, axis=0)
    z_r = tl.sum(z * tl.exp(m - m_r), axis=0)
    # all-(-inf) tails must contribute z=0 to the combine (exp(-inf - -inf)
    # would be NaN), and all-(-inf) rows are resolved by the combine's -inf
    # guard.
    tl.store(mrow_ptr + pid, m_r)
    tl.store(zrow_ptr + pid, tl.where(m_r == float("-inf"), 0.0, z_r))


def _reduce_inner_1d(inp, rows, N, out):
    """Exact per-row fallback for the small-N path.

    The 2D multirow tile's axis-1 reduction miscompiles on this XPU backend
    for widths N % 8 != 0 (deterministic wrong results, e.g. N in
    {5,6,7,9,10,11,12,13,14,15,17,...}, 2D axis-1 max AND sum affected, and
    neither column masking nor a plain fp32 reduce helps). Every tested
    multiple of 8 is exact, but for the other widths we route through the
    online per-row kernel (exact for all N) plus the tiny combine kernel.
    """
    mrow = torch.empty((rows,), dtype=torch.float32, device=inp.device)
    zrow = torch.empty_like(mrow)
    _reduce_tail_partials(mrow, zrow, inp, rows, N, N)
    logsumexp_kernel_combine[(rows, 1, 1)](
        out,
        mrow,
        zrow,
        mrow,
        zrow,
        rows,
        C_FULL=1,
        HAS_TAIL=0,
        TILE_C=1,
        num_warps=4,
        buffer_size_limit=2048,
    )


def _reduce_inner_small(inp, rows, N, out):
    """Single-tile multirow kernel for N <= _MULTIROW_MAX_N."""
    if N % 8:
        _reduce_inner_1d(inp, rows, N, out)
        return
    if N <= 64:
        TILE_M = 16
    elif N <= 256:
        TILE_M = 64
    elif N <= 1024:
        TILE_M = 32
    else:
        TILE_M = 8
    need_mask = 1 if rows % TILE_M else 0
    grid = (triton.cdiv(rows, TILE_M), 1, 1)
    logsumexp_kernel_multirow[grid](
        out,
        inp,
        rows,
        N=N,
        TILE_M=TILE_M,
        NEED_MASK=need_mask,
        num_warps=4,
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


def _reduce_inner(inp, rows, N):
    """logsumexp over the innermost dim N of a contiguous [rows, N] tensor."""
    out = torch.empty((rows,), dtype=inp.dtype, device=inp.device)
    if N <= _MULTIROW_BUDGET:
        TILE_M = max(1, _MULTIROW_BUDGET // N)
        grid = (triton.cdiv(rows, TILE_M), 1, 1)
        logsumexp_kernel_multirow[grid](
            out,
            inp,
            rows,
            N=N,
            TILE_M=TILE_M,
            isCloseVectorization=True,
            buffer_size_limit=2048,
        )
    else:
        # Chunk-split path: single data read, single exp per element. Full
        # 4096-chunks go through the tile kernel; any tail (N % 4096 != 0) is
        # reduced by the multirow kernel over a tail-slice view (masked-tail
        # reductions miscompute on this backend).
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
            # With C_full == 1 and a tail, `reshape` is a same-shape no-op
            # that returns a strided view (row stride N, not BN); the partial
            # kernel indexes row-major, so materialize an exact contiguous
            # copy when the flattened view is not actually contiguous.
            if not flat.is_contiguous():
                flat = flat.contiguous()
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
            inp,
            rows,
            N,
            TILE_N=TILE_N,
            isCloseVectorization=True,
            buffer_size_limit=2048,
        )
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
    # mis-computes (2D axis=0 reduce is rejected). N==1 is a trivial identity that
    # the native kernel does faster than a gems copy.
    if K > 1 or N == 1:
        return _native_logsumexp(inp, [dim], keepdim)

    # K == 1: innermost-dim reduction -> fast contiguous Triton multirow tile.
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
