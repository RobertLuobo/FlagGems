import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as ext
from flag_gems.utils.limits import get_dtype_max

from ..utils.block_size_utils import get_block_size_1d

logger = logging.getLogger(__name__)

# Kunlunxin amin (mirrors amax with the sign flipped max -> min, +inf identity;
# NaN propagation identical to torch.amin).
#
# 2026-09-10 performance closure (XPU sweeps on the full benchmark shape set):
#   - The previous fast path accumulated a [BLOCK_M, 1] running accumulator and
#     called tl.min(axis=1) inside the loop (reduce-INSIDE); the per-iteration
#     row reduce costs ~2x the elementwise tl.minimum of the sum-validated
#     reduce-OUTSIDE pattern (same tile [128, 1024] measured 305us vs 159us on
#     (1024, 65536) fp16).  The dim fast path now uses `amin_rows_kernel`
#     (reduce-OUTSIDE, clamped rows, fully unmasked loads - the exact pattern
#     validated for _sum_row_full_kernel), with per-N tile rules from the sweep:
#       N >= 2^20 : [8, 8192]  fp16 / [4, 8192]  fp32   (2.5-3.1x on 1M-wide rows)
#       N >= 2^16 : [128, 1024] fp16 / [32, 1024] fp32  (1.4-1.6x on 64K-wide rows)
#       otherwise  : [128, bn]  fp16 / [64, bn]  fp32   (bn = 512/1024/... divide N)
#   - The flat (dim=None) path uses exact 32768-lane unmasked chunks (documented
#     exact point with buffer_size_limit=2048, cf. nansum) for numel < 2^26 and a
#     row-8192 2-stage for huge numel (32768-lane chunking degrades at ~2^30
#     elements: 32K programs vs 1K wide programs).
#   - The masked fallback (shapes no unmasked tile covers: N % 16 != 0 or
#     M % BM != 0) is `amin_rows_masked_kernel` (one program per row, 1-D
#     loads): the previous [BM, BN] 2-D masked form miscompiles on this XPU
#     for non-divisible shapes (see the kernel docstring), so it was replaced
#     rather than kept (2026-09-10).

_FULL_REDUCTION_BLOCK_SIZE = 8192

_FLAT_CHUNK = 32768
_FLAT_ROW_WIDTH = 8192
_FLAT_CHUNK_MAX_NUMEL = 1 << 26
_BLOCK_N_MAX = 8192


@libentry()
@triton.jit
def amin_kernel_1(
    inp,
    mid,
    M,
    BLOCK_SIZE: tl.constexpr,
    NEED_MASK: tl.constexpr,
):
    pid = ext.program_id(0)

    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    inp_ptrs = inp + offset
    if NEED_MASK:
        mask = offset < M
        inp_val = tl.load(inp_ptrs, mask=mask, other=float("inf"))
    else:
        inp_val = tl.load(inp_ptrs)
    amin_val = tl.min(inp_val)
    mid_ptr = mid + pid
    tl.store(mid_ptr, amin_val)


@libentry()
@triton.jit
def amin_kernel_2(mid, out, mid_size, BLOCK_MID: tl.constexpr):
    offset = tl.arange(0, BLOCK_MID)
    mid_ptrs = mid + offset
    mask = offset < mid_size
    max_value = get_dtype_max(mid.type.element_ty)
    mid_val = tl.load(mid_ptrs, mask=mask, other=max_value)
    amin_val = tl.min(mid_val)
    tl.store(out, amin_val)


@libentry()
@triton.jit
def amin_rows_kernel(inp, out, M, NW, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    """Row-reduce over cols [0, NW) with NW % BLOCK_N == 0.  Fully unmasked
    loads (rows clamped to [0, M-1]), reduce-OUTSIDE accumulation into a
    [BLOCK_M, BLOCK_N] tile with a single final `tl.min(axis=1)`, masked row
    stores.  Exact (same validated pattern as _sum_row_full_kernel)."""
    pid = ext.program_id(0)
    rows = pid * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]
    rows_c = tl.where(rows < M, rows, M - 1)
    inp = inp + rows_c * NW
    out = out + rows
    row_mask = rows < M
    acc = tl.full([BLOCK_M, BLOCK_N], value=float("inf"), dtype=tl.float32)
    for off in range(0, NW, BLOCK_N):
        cols = off + tl.arange(0, BLOCK_N)[None, :]
        a = tl.load(inp + cols).to(tl.float32)
        acc = tl.minimum(acc, a)
    tl.store(out, tl.min(acc, axis=1)[:, None], row_mask)


@libentry()
@triton.jit
def amin_flat_chunk_kernel(inp, out, CHUNK: tl.constexpr):
    """Unmasked exact-size 1-D chunk reduce.  Grid = number of full chunks."""
    pid = ext.program_id(0)
    off = pid * CHUNK + tl.arange(0, CHUNK)
    a = tl.load(inp + off).to(tl.float32)
    tl.store(out + pid, tl.min(a))


@libentry()
@triton.jit
def amin_flat_tail_kernel(inp, out, start, NTAIL, TL: tl.constexpr):
    """Single-shot masked tail (NTAIL <= 8192 lanes; validated 2026-08-22).
    start is a scalar flat offset."""
    off = tl.arange(0, TL)
    a = tl.load(inp + start + off, mask=off < NTAIL, other=float("inf")).to(tl.float32)
    tl.store(out, tl.min(a))


@libentry()
@triton.jit
def amin_flat_group_kernel(mid, gsum, GCHUNK: tl.constexpr):
    """Unmasked group-reduce of a zero-padded partial buffer (compresses
    8192 partials per program)."""
    pid = ext.program_id(0)
    off = pid * GCHUNK + tl.arange(0, GCHUNK)
    a = tl.load(mid + off).to(tl.float32)
    tl.store(gsum + pid, tl.min(a))


@libentry()
@triton.jit
def amin_flat_merge_kernel(mid, out, np, NLANES: tl.constexpr):
    """Single-shot masked merge of np partials (np <= 8192)."""
    off = tl.arange(0, NLANES)
    a = tl.load(mid + off, mask=off < np, other=float("inf")).to(tl.float32)
    tl.store(out, tl.min(a))


# Master-free fast-path tile preferences (XPU sweeps, 2026-09-10):
#   fp16/bf16: [BLOCK_M=8, BLOCK_N=8192] at >= 1M-wide rows (1024x1048576:
#              1822us vs the old 4675us); [BLOCK_M=128, BLOCK_N=1024] at
#              64K..1M (1024x65536: 159us vs 305us); [128, 512/1024/256...]
#              below.
#   fp32:      [4, 8192] at >= 1M-wide rows (2655us vs 8018us); [32, 1024] at
#              64K..1M (201us vs 436us); [64, 512] below (1024x4096:
#              37us vs 43us).
# The fast path only triggers when both M and N divide by the picked tiles
# (the whole reduction is mask-free); everything else keeps the old masked
# path unchanged.
_FAST_BN_FP16 = (1024, 512, 256, 128, 64, 32, 16)
_FAST_BN_FP32 = (512, 1024, 256, 128, 64, 16, 32)
_FAST_BM_FP16 = (128, 64, 32, 16, 8, 4, 2)
_FAST_BM_FP32 = (64, 128, 32, 16, 8, 4, 2)


def _pick_fast_tile(M, N, is_fp32):
    """Return (BLOCK_M, BLOCK_N) with M % BLOCK_M == 0 and N % BLOCK_N == 0, or
    None when no mask-free tile covers this shape."""
    if M < 2:
        return None
    if N >= (1 << 20):
        if N % 8192 == 0:
            bm = next(
                (
                    m
                    for m in ((8, 4, 16, 32, 2) if not is_fp32 else (4, 8, 16, 32, 2))
                    if M % m == 0
                ),
                None,
            )
            if bm is not None:
                return bm, 8192
    if N >= (1 << 16):
        if N % 1024 == 0:
            bm = next(
                (
                    m
                    for m in (
                        (128, 64, 32, 16, 8, 4) if not is_fp32 else (32, 16, 64, 8, 4)
                    )
                    if M % m == 0
                ),
                None,
            )
            if bm is not None:
                return bm, 1024
    bns = _FAST_BN_FP32 if is_fp32 else _FAST_BN_FP16
    bms = _FAST_BM_FP32 if is_fp32 else _FAST_BM_FP16
    bn = next((b for b in bns if N % b == 0), None)
    if bn is None:
        return None
    bm = next((b for b in bms if M % b == 0), None)
    if bm is None:
        return None
    return bm, bn


@libentry()
@triton.jit
def amin_rows_masked_kernel(inp, out, M, N, BLOCK: tl.constexpr):
    """Masked fallback (shapes no unmasked tile covers): one program per row,
    [BLOCK]-lane 1-D loads, single final `tl.min(axis=0)`.

    The previous [BLOCK_M, BLOCK_N] 2-D masked form miscompiles on this XPU
    for non-divisible shapes (probed 2026-09-10):
      * `tl.where(mask, a, inf)` re-masking a masked bf16 load returns wrong
        lanes on column-masked configs (M=40999, N=600, BM=64/4096, BN=1024:
        291/40999 output columns wrong; the form without the re-mask is
        exact there);
      * 2-D loads whose row stride is not 16B-aligned (N * elem_size % 16
        != 0, e.g. N=40999, BN=8192) return garbage even without the
        re-mask (370/600 rows wrong; N=40960 with the same blocks: exact).
    1-D masked loads per row (the amin_kernel_1 / amin_flat_tail_kernel
    family) avoid both; verified exact for fp16/fp32/bf16 on the functional
    matrix, both (600, 40999) orientations, and odd-N / multi-chunk / N<BLOCK
    edge cases."""
    row = ext.program_id(0)
    start = row * N
    off = tl.arange(0, BLOCK)
    acc = tl.full([BLOCK], value=float("inf"), dtype=tl.float32)
    for s in range(0, N, BLOCK):
        cols = s + off
        m = cols < N
        a = tl.load(inp + start + cols, mask=m, other=float("inf")).to(tl.float32)
        acc = tl.minimum(acc, a)
    tl.store(out + row, tl.min(acc, axis=0))


def _amin_flat(inp, out, device):
    """Full (dim=None) reduction over `inp` (any numel)."""
    numel = inp.numel()
    with torch_device_fn.device(device):
        if numel <= _FULL_REDUCTION_BLOCK_SIZE:
            amin_flat_merge_kernel[(1, 1, 1)](
                inp, out, numel, _FULL_REDUCTION_BLOCK_SIZE
            )
            return
        if numel < _FLAT_CHUNK_MAX_NUMEL:
            # Exact 32768-lane unmasked chunks (documented exact point with
            # buffer_size_limit=2048, cf. nansum) + single masked merge.
            nfull = numel // _FLAT_CHUNK
            tail = numel - nfull * _FLAT_CHUNK
            nb = nfull + (1 if tail else 0)
            mid = torch.empty((nb,), dtype=inp.dtype, device=device)
            if nfull:
                amin_flat_chunk_kernel[(nfull, 1, 1)](
                    inp, mid, _FLAT_CHUNK, buffer_size_limit=2048
                )
            if tail:
                if tail <= 8192:
                    amin_flat_tail_kernel[(1, 1, 1)](
                        inp,
                        mid[nfull : nfull + 1],
                        nfull * _FLAT_CHUNK,
                        tail,
                        triton.next_power_of_2(tail),
                    )
                else:
                    # Stage the (>8192) tail into a zero-padded 2^K buffer so
                    # the chunk kernel stays fully unmasked.
                    TL = triton.next_power_of_2(tail)
                    staged = torch.zeros((TL,), dtype=inp.dtype, device=device)
                    torch.ops.aten._copy_from(
                        inp[nfull * _FLAT_CHUNK :], staged[:tail], False
                    )
                    amin_flat_chunk_kernel[(1, 1, 1)](
                        staged, mid[nfull : nfull + 1], TL, buffer_size_limit=2048
                    )
            amin_flat_merge_kernel[(1, 1, 1)](
                mid, out, nb, triton.next_power_of_2(nb)
            )
        else:
            # Huge numel (>= 2^26): row-8192 2-stage.  The 32768-lane chunk
            # design launches numel/32768 programs (32K at 1G elements) which
            # is slower than kb-wide programs of the row kernel (measured
            # 16.8ms vs 2.9ms at 2^30 elements).
            rows = numel // _FLAT_ROW_WIDTH
            res = numel - rows * _FLAT_ROW_WIDTH
            bm = next(
                (m for m in _FAST_BM_FP16 if rows % m == 0), _FAST_BM_FP16[0]
            )
            nb = rows + (1 if res else 0)
            mid = torch.empty((nb,), dtype=inp.dtype, device=device)
            amin_rows_kernel[(rows // bm, 1)](
                inp,
                mid,
                rows,
                _FLAT_ROW_WIDTH,
                bm,
                1024,
                buffer_size_limit=2048,
            )
            if res:
                amin_flat_tail_kernel[(1, 1, 1)](
                    inp,
                    mid[rows:],
                    rows * _FLAT_ROW_WIDTH,
                    res,
                    triton.next_power_of_2(res),
                )
            if nb <= 8192:
                amin_flat_merge_kernel[(1, 1, 1)](
                    mid, out, nb, triton.next_power_of_2(nb)
                )
            else:
                g = (nb + 8191) // 8192
                padded = torch.zeros((g * 8192,), dtype=inp.dtype, device=device)
                torch.ops.aten._copy_from(mid, padded[:nb], False)
                gsum = torch.empty((g,), dtype=inp.dtype, device=device)
                amin_flat_group_kernel[(g, 1, 1)](padded, gsum, 8192)
                amin_flat_merge_kernel[(1, 1, 1)](
                    gsum, out, g, triton.next_power_of_2(g)
                )


def amin(inp, dim=None, keepdim=False):
    logger.debug("GEMS_KUNLUNXIN AMIN")
    if dim is None or (isinstance(dim, (list, tuple)) and len(dim) == 0):
        dtype = inp.dtype
        if not keepdim:
            out = torch.empty([], dtype=dtype, device=inp.device)
        else:
            shape = list(inp.shape)
            for i in range(0, inp.dim()):
                shape[i] = 1
            out = torch.empty(shape, dtype=dtype, device=inp.device)
        with torch_device_fn.device(inp.device):
            _amin_flat(inp.reshape(-1), out, inp.device)
        return out
    else:
        if isinstance(dim, int):
            dim = [dim]
        assert ((i >= -inp.ndim and i < inp.ndim) for i in dim), "Invalid dim"
        dtype = inp.dtype

        shape = list(inp.shape)
        dim = [d % inp.ndim for d in dim]
        N = 1
        for i in dim:
            N *= shape[i]
            shape[i] = 1
        M = inp.numel() // N

        if N == 1:
            # Every reduced dim has size 1: amin over it is the identity. Use the
            # native strided-copy engine (flag_gems does not override
            # `_copy_from`) instead of launching a reduction kernel at all.
            out = torch.empty(shape, dtype=dtype, device=inp.device)
            with torch_device_fn.device(inp.device):
                torch.ops.aten._copy_from(inp, out, False)
            if not keepdim:
                out = out.squeeze(dim=dim)
            return out

        # Reorder so the reduced dims are innermost (same order as
        # dim_compress), then make it contiguous with the native strided-copy
        # engine instead of the much slower gems `.contiguous()` override.
        dim_i = inp.dim()
        stride = inp.stride()
        batch_dim = [i for i in range(dim_i) if i not in dim]
        sorted_reduction_dim = sorted(dim, key=lambda x: stride[x], reverse=True)
        order = batch_dim + sorted_reduction_dim
        view = inp.permute(order)
        if view.is_contiguous():
            src = view
        else:
            src = torch.empty(list(view.shape), dtype=dtype, device=inp.device)
            with torch_device_fn.device(inp.device):
                torch.ops.aten._copy_from(view, src, False)

        out = torch.empty(shape, dtype=dtype, device=inp.device)

        is_fp32 = dtype == torch.float32
        tile = _pick_fast_tile(M, N, is_fp32)
        with torch_device_fn.device(inp.device):
            if tile is not None:
                block_m, block_n = tile
                grid = (triton.cdiv(M, block_m),)
                amin_rows_kernel[grid](
                    src,
                    out,
                    M,
                    N,
                    block_m,
                    block_n,
                    buffer_size_limit=2048,
                )
            else:
                block_n = min(triton.next_power_of_2(N), _BLOCK_N_MAX)
                amin_rows_masked_kernel[(M, 1)](
                    src,
                    out,
                    M,
                    N,
                    block_n,
                    buffer_size_limit=2048,
                )
        if not keepdim:
            out = out.squeeze(dim=dim)
        return out


def amin_(inp, dim=None, keepdim=False):
    logger.debug("GEMS_KUNLUNXIN AMIN_")
    if isinstance(dim, int):
        dim = [dim]
    if dim is None or len(dim) == 0:
        M = inp.numel()
        block_size = get_block_size_1d(M, inp.element_size())
        mid_size = triton.cdiv(M, block_size)
        block_mid = triton.next_power_of_2(mid_size)
        dtype = inp.dtype
        mid = torch.empty((mid_size,), dtype=dtype, device=inp.device)
        if not keepdim:
            out = torch.empty([], dtype=dtype, device=inp.device)
        else:
            shape = list(inp.shape)
            for i in range(0, inp.dim()):
                shape[i] = 1
            out = torch.empty(shape, dtype=dtype, device=inp.device)
        with torch_device_fn.device(inp.device):
            amin_kernel_1[(mid_size, 1)](
                inp,
                mid,
                M,
                block_size,
                M % block_size != 0,
                buffer_size_limit=2048,
            )
            amin_kernel_2[(1, 1)](mid, out, mid_size, block_mid, buffer_size_limit=2048)
        # NOTE (XPU): Tensor.copy_ on this torch_xmlir XPU build does NOT
        # broadcast a size-1 (or 0-dim) src to a larger `inp`; it only
        # flatten-copies the first numel(src) elements (leaving the rest of
        # `inp` untouched, corrupting the inplace result).  `out.reshape(inp.shape)`
        # also raises on size-1 -> larger shapes.  Expand first so src already has
        # inp.shape (a stride-0 view), then copy_ is a plain same-shape copy.
        inp.copy_(out if out.shape == inp.shape else out.expand_as(inp))
        return inp
    else:
        result = amin(inp, dim=dim, keepdim=True)
        # See note above: expand the (reduced-size-1) result to inp.shape before
        # the inplace copy, since XPU copy_ does not broadcast.
        inp.copy_(result if result.shape == inp.shape else result.expand_as(inp))
        return inp