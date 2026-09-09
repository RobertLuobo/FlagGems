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

# from flag_gems import runtime
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as ext

from ..utils.block_size_utils import get_block_size_1d

logger = logging.getLogger(__name__)


@triton.jit
def reduce_mul(a, b):
    return a * b


@libentry()
@triton.jit
def prod_kernel_mid(
    inp,
    mid,
    M,
    BLOCK_SIZE: tl.constexpr,
):
    pid = ext.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    inp_ptrs = inp + offset
    mask = offset < M
    inp_val = tl.load(inp_ptrs, mask=mask, other=1.0).to(tl.float32)
    mid_value = tl.reduce(inp_val, axis=0, combine_fn=reduce_mul)
    mid_ptr = mid + pid
    tl.store(mid_ptr, mid_value.to(inp_val.dtype))


@libentry()
@triton.jit
def prod_kernel_result(mid, out, mid_size, BLOCK_MID: tl.constexpr):
    offset = tl.arange(0, BLOCK_MID)
    mid_ptrs = mid + offset
    mask = offset < mid_size
    mid_val = tl.load(mid_ptrs, mask=mask, other=1.0).to(tl.float32)
    prod_val = tl.reduce(mid_val, axis=0, combine_fn=reduce_mul)
    tl.store(out, prod_val)


def prod(inp, *, dtype=None):
    logger.debug("GEMS_KUNLUNXIN PROD")
    if dtype is None:
        dtype = inp.dtype

    M = inp.numel()
    # block_size = triton.next_power_of_2(math.ceil(math.sqrt(M)))
    block_size = get_block_size_1d(M, inp.element_size())
    mid_size = triton.cdiv(M, block_size)
    block_mid = triton.next_power_of_2(mid_size)

    mid = torch.empty((mid_size,), dtype=dtype, device=inp.device)
    out = torch.empty([], dtype=dtype, device=inp.device)

    with torch_device_fn.device(inp.device):
        prod_kernel_mid[(mid_size, 1, 1)](
            inp, mid, M, block_size, buffer_size_limit=2048
        )
        if mid_size == 1:
            return mid.reshape([])
        prod_kernel_result[(1, 1, 1)](
            mid, out, mid_size, block_mid, buffer_size_limit=2048
        )
    return out


def heur_m_block_size(args):
    # Bound the accumulator tile [BLOCK_M, BLOCK_N] to a fixed element budget so
    # BLOCK_M can never explode with M. The old `next_pow2(cdiv(M, 12))` built
    # multi-hundred-MB tiles with only ~8 programs when M was large and N small
    # (e.g. (1024,1024,1024) reduce dim=1 -> BLOCK_M=131072, a 512MB tile ->
    # 747ms catastrophe). A ~64K-element tile keeps per-program work high (few,
    # large contiguous DMA passes) while capping SRAM, and caps BLOCK_M at
    # next_pow2(M) so tiny-M rows are not over-allocated.
    import builtins

    M, N = args["M"], args["N"]
    block_n = builtins.min(triton.next_power_of_2(N), 8192)
    block_m = builtins.max(1, 65536 // block_n)
    return builtins.min(block_m, triton.next_power_of_2(M))


def heur_n_block_size(args):
    import builtins

    return builtins.min(triton.next_power_of_2(args["N"]), 8192)


@libentry()
@triton.heuristics(
    values={
        "BLOCK_M": heur_m_block_size,
        "BLOCK_N": heur_n_block_size,
    },
)
@triton.jit
def prod_kernel(
    inp,
    out,
    M,
    N,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    # Map program id to its rows and pre-offset the base pointer so the inner
    # `inp + cols` access is proven contiguous by OffsetAnalysis (block DMA).
    # Computing `m_offset[:, None] * N + n_offset` inline (old impl) blocks the
    # analysis -> discrete scalar gather.
    pid = ext.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]
    inp = inp + pid * N
    out = out + pid
    row_mask = pid < M

    acc = tl.full((BLOCK_M, BLOCK_N), value=1.0, dtype=tl.float32)
    for off in range(0, N, BLOCK_N):
        cols = off + tl.arange(0, BLOCK_N)[None, :]
        col_mask = cols < N
        inp_vals = tl.load(inp + cols, mask=row_mask & col_mask, other=1.0).to(
            tl.float32
        )
        acc *= inp_vals
    result = tl.reduce(acc, axis=1, combine_fn=reduce_mul)[:, None]
    tl.store(out, result, row_mask)


@libentry()
@triton.jit
def prod_kernel_kn(
    out,
    inp,
    M,
    N,
    K,
    TILE_N: tl.constexpr,
    TILE_K: tl.constexpr,
):
    # Middle-dim reduce (K>1) WITHOUT dim_compress transpose. View inp as
    # [M, N, K] (contiguous). A [TILE_K, TILE_N] tile puts K on axis0 (rows) and
    # the reduce target N on axis1 (cols) so we can use XPU's ONLY supported
    # `tl.reduce(axis=1)` -- axis=0 2D reduce is a dead end on this toolchain
    # (compile fail or numerically wrong). dim_compress would permute N to the
    # innermost then `.contiguous()` on a non-contiguous permuted tensor, which
    # under use_gems dispatches to the generic strided copy_ (~300x slower =
    # the "transpose wall"); this KN layout avoids that copy entirely.
    pid_m = ext.program_id(0)
    pid_k = ext.program_id(1)
    k_off = pid_k * TILE_K + tl.arange(0, TILE_K)[:, None]
    acc = tl.full([TILE_K, TILE_N], value=1.0, dtype=tl.float32)
    for start_n in range(0, N, TILE_N):
        n_off = start_n + tl.arange(0, TILE_N)[None, :]
        off = pid_m * N * K + n_off * K + k_off
        mask = (k_off < K) & (n_off < N)
        v = tl.load(inp + off, mask=mask, other=1.0).to(tl.float32)
        acc *= v
    result = tl.reduce(acc, axis=1, combine_fn=reduce_mul, keep_dims=True)
    out_off = pid_m * K + k_off
    tl.store(out + out_off, result, mask=k_off < K)


def prod_dim(inp, dim=None, keepdim=False, *, dtype=None):
    logger.debug("GEMS_KUNLUNXIN PROD_DIM")
    import builtins

    assert dim >= -inp.ndim and dim < inp.ndim, "Invalid dim"
    shape = list(inp.shape)
    dim = dim % inp.ndim

    N = shape[dim]
    M = 1
    for s in shape[:dim]:
        M *= s
    K = 1
    for s in shape[dim + 1 :]:
        K *= s

    out_shape = shape.copy()
    out_shape[dim] = 1
    if dtype is None:
        dtype = inp.dtype

    inp = inp.contiguous()
    out = torch.empty(out_shape, dtype=dtype, device=inp.device)
    if M == 0 or K == 0:
        if not keepdim:
            out = torch.squeeze(out, d)
        return out
    if N == 0:
        out.fill_(1)
        if not keepdim:
            out = torch.squeeze(out, d)
        return out
    if N == 1:
        # reduce over a size-1 dim is the identity
        with torch_device_fn.device(inp.device):
            torch.ops.aten._copy_from(inp, out, False)
        if not keepdim:
            out = torch.squeeze(out, d)
        return out

    # Move the reduced dim innermost (same order as dim_compress) and make it
    # contiguous with the native strided-copy engine instead of gems'
    # `.contiguous()` (which is ~1000x slower for big transposes).
    order = [i for i in range(inp.dim()) if i != d] + [d]
    view = inp.permute(order)
    if view.is_contiguous():
        src = view
    else:
        src = torch.empty(list(view.shape), dtype=inp.dtype, device=inp.device)
        with torch_device_fn.device(inp.device):
            torch.ops.aten._copy_from(view, src, False)

    rows = M * K
    out_flat = out.reshape(rows)
    src = src.reshape(rows, N) if src.dim() > 2 else src
    if (not inp.dtype.is_floating_point) or (rows * N >= 2**31):
        # ints (int64 exact accumulator) or >2^31-element reductions (the
        # split-K kernels use i32 tile indices): keep the generic row kernel
        is_fp32 = dtype == torch.float32
        acc32 = dtype.is_floating_point
        tree16 = False
        with torch_device_fn.device(inp.device):
            tile = _pick_fast_tile(rows, N, is_fp32)
            if tile is not None and (not tree16 or N % _TREE16_BLOCK == 0):
                bm, bn = tile
                if tree16:
                    bn = _TREE16_BLOCK  # fp16 native trees: BN <= 512
                prod_row2d[(max(rows // bm, 1), 1)](
                    src,
                    out_flat,
                    rows,
                    N,
                    bm,
                    bn,
                    False,
                    acc32,
                    tree16,
                    buffer_size_limit=2048,
                )
            else:
                bn = min(triton.next_power_of_2(N), _REDUCE_BLOCK)
                if tree16:
                    bn = min(bn, _TREE16_BLOCK)
                bm = triton.next_power_of_2(min(triton.cdiv(rows, 12), 65536 // bn))
                grid = (triton.cdiv(rows, bm),)
                prod_row2d[grid](
                    src,
                    out_flat,
                    rows,
                    N,
                    bm,
                    bn,
                    True,
                    acc32,
                    tree16,
                    buffer_size_limit=2048,
                )
        if not keepdim:
            out = torch.squeeze(out, d)
        return out

    # ---- float path: split-K with a mask-free two-level row tiling ---------
    # The previous single-launch row kernel either loops the whole N
    # sequentially in one program (rows == 1: ~0.5 s for a 2^28 reduction) or
    # is launch-bound when the row count has only small power-of-two divisors.
    # Instead: (1) split the reduced dim into 8192-wide (or pow2-tail) column
    # chunks, one 2D-grid launch per distinct chunk width, writing [rows, C]
    # fp32 partials; (2) reduce the C partial columns per row with the
    # existing prod_row2d. Rows are split into full BLOCK_M blocks plus
    # tr//bmt tail blocks of height bmt (bmt = largest power-of-two divisor of
    # the tail; bmt | tr so the tail blocks tile the tr rows exactly and every
    # load/store is in-bounds and compiles mask-free (the masked memory path
    # costs ~2x on this backend). The tail grid MUST use tr//bmt (not a single
    # block), otherwise the last tr-bmt rows are never written.
    if rows == 1:
        # whole-tensor product: identical to the flat path (M == K == 1)
        with torch_device_fn.device(inp.device):
            _prod_flat(src, out_flat.reshape(()), inp.device)
        if not keepdim:
            out = torch.squeeze(out, d)
        return out

    BM = 64
    BN = 512
    CHUNK = _REDUCE_BLOCK
    main = N // CHUNK
    r = N % CHUNK
    tail = _pow2_decomp(r) if r else []
    C = main + len(tail)
    nb = rows // BM
    tr = rows - nb * BM
    bmt = _bm_tail(tr) if tr else 1
    with torch_device_fn.device(inp.device):
        if C == 1:
            w = CHUNK if main else tail[0]
            if nb:
                prod_dim_single[(1, nb)](
                    src,
                    out_flat,
                    N,
                    w,
                    BM,
                    min(BN, w),
                    num_warps=8,
                    buffer_size_limit=2048,
                )
            if tr:
                prod_dim_single[(1, tr // bmt)](
                    src[nb * BM :],
                    out_flat[nb * BM :],
                    N,
                    w,
                    bmt,
                    min(BN, w),
                    num_warps=8,
                    buffer_size_limit=2048,
                )
        else:
            part = torch.empty((rows, C), dtype=torch.float32, device=inp.device)
            if main:
                if nb:
                    prod_dim_chunk[(main, nb)](
                        src,
                        part,
                        N,
                        0,
                        0,
                        C,
                        CHUNK,
                        BM,
                        BN,
                        num_warps=8,
                        buffer_size_limit=2048,
                    )
                if tr:
                    prod_dim_chunk[(main, tr // bmt)](
                        src[nb * BM :],
                        part[nb * BM :],
                        N,
                        0,
                        0,
                        C,
                        CHUNK,
                        bmt,
                        BN,
                        num_warps=8,
                        buffer_size_limit=2048,
                    )
            bpos = main * CHUNK
            for i, w in enumerate(tail):
                if nb:
                    prod_dim_chunk[(1, nb)](
                        src,
                        part,
                        N,
                        bpos,
                        main + i,
                        C,
                        w,
                        BM,
                        min(BN, w),
                        num_warps=8,
                        buffer_size_limit=2048,
                    )
                if tr:
                    prod_dim_chunk[(1, tr // bmt)](
                        src[nb * BM :],
                        part[nb * BM :],
                        N,
                        bpos,
                        main + i,
                        C,
                        w,
                        bmt,
                        min(BN, w),
                        num_warps=8,
                        buffer_size_limit=2048,
                    )
                bpos += w
            # stage 2: per-row product of the C partial columns; pad C to a
            # power of two with 1.0s (product identity) so the reduce is exact.
            CP = 1 << (C - 1).bit_length()
            if CP != C:
                part2 = torch.ones((rows, CP), dtype=torch.float32, device=inp.device)
                torch.ops.aten._copy_from(part, part2[:, :C], False)
                part = part2
            prod_row2d[((rows + 63) // 64,)](
                part,
                out_flat,
                rows,
                CP,
                64,
                min(1024, CP),
                rows % 64 != 0,
                True,
                False,
                num_warps=8,
                buffer_size_limit=2048,
            )

    if not keepdim:
        out = torch.squeeze(out, dim)

    return out
