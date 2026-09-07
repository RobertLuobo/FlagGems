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

# ---- design notes (Kunlunxin XPU, 2026-08-21) -------------------------------
# prod = multiply-reduction.
# * Flat (dim=None): rows-of-8192 main pass with the 2-D row kernel (few wide
#   programs, mask-free), pow2-decomposed unmasked tail chunks (XPU masked-tail
#   loads are unreliable -> never mask), staged fp32/idtree. 8192-lane reduce is
#   the reliable width on this backend.
# * dim path: reduced axes are moved innermost via `permute(order)`; the
#   physical reorder uses the native strided-copy engine
#   (`torch.ops.aten._copy_from`, flag_gems never overrides it) instead of the
#   slow gems `.contiguous()`; then a [BLOCK_M, BLOCK_N] 2D row kernel reduces
#   the innermost N (reduce-INSIDE accumulate in fp32).
# * TREE16 (2026-08-21, XPU 4): fp16 was ~5.5x slower per element than fp32
#   because of the per-element fp16->fp32 conversion before the tree
#   (measured: flat 2^30 fp16 14.2ms vs fp32 3.9ms). Keeping the in-tile
#   reduce tree in fp16 (native width) and converting only the per-chunk
#   [BM,1] partial to fp32 recovers the speed (2^30 fp16: 13.7ms -> 2.5ms,
#   ~800 GB/s effective) with no meaningful precision loss (fp16 input
#   quantization dominates; verified equal to the fp32-tree on fp16 data).
#   fp16 native trees are limited to BLOCK_N <= 512 (>=1024 lanes -> uni_sram
#   OOM on this backend); bf16 keeps the fp32 tree (bf16 native trees
#   miscompile in TritonXPUDtypeConvert -> OutOfResources at any width).
# * N == 1 (all reduced dims of size 1): identity -> native `_copy_from`.
# * N == 0: product over an empty dim = 1 -> fill.
# * ints: accumulate in the input width (wrap-around like torch.prod) since fp32
#   accumulation saturates to inf for large int products.

_REDUCE_BLOCK = 8192  # reliable single-load tl.reduce width
_TREE16_BLOCK = 512  # max fp16-native reduce width (larger -> uni_sram OOM)
_FAST_BN_FP16 = (1024, 256, 512, 128, 64, 32, 16)
_FAST_BN_FP32 = (512, 256, 1024, 128, 64, 32, 16)
_FAST_BM_FP16 = (128, 64, 32, 256, 16, 8, 4, 2, 1)
_FAST_BM_FP32 = (64, 128, 32, 256, 16, 8, 4, 2, 1)


@triton.jit
def reduce_mul(a, b):
    return a * b


def _pick_fast_tile(M, N, is_fp32):
    """Return (BLOCK_M, BLOCK_N) with M % BLOCK_M == 0 and N % BLOCK_N == 0, so
    the whole reduction runs mask-free, or None."""
    bns = _FAST_BN_FP32 if is_fp32 else _FAST_BN_FP16
    bms = _FAST_BM_FP32 if is_fp32 else _FAST_BM_FP16
    bn = next((b for b in bns if N % b == 0), None)
    if bn is None:
        return None
    bm = next((m for m in bms if M % m == 0), None)
    if bm is None:
        return None
    return bm, bn


def _tree16(dt):
    # fp16 native-width reduce tree; bf16/fp32 keep the fp32 tree
    return dt == torch.float16


def _work_dtype(dt):
    # floats: fp32 partials; ints: int64 (torch.prod promotes int inputs to
    # int64 and returns the exact int64 product, not the wrapped input width)
    return torch.float32 if dt.is_floating_point else torch.int64


@libentry()
@triton.jit
def prod_row2d(
    inp,
    out,
    M,
    N,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    NEED_MASK: tl.constexpr,
    ACC32: tl.constexpr,
    TREE16: tl.constexpr = False,
):
    # Map the program id to its rows and pre-offset the base pointer so the
    # inner `inp + cols` access is proven contiguous by OffsetAnalysis.
    pid = ext.program_id(0)
    rows = pid * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]
    inp = inp + rows * N
    out = out + rows
    row_mask = rows < M

    # reduce-INSIDE accumulation: each [BLOCK_M, BLOCK_N] block is reduced over
    # N (axis=1) first, then multiplied into a [BLOCK_M, 1] accumulator. This
    # is the only form that is numerically reliable on this XPU (the
    # reduce-OUTSIDE variant with a persistent tile miscompiles for bf16 with
    # masked tails). NEED_MASK=False compiles to a fully mask-free kernel.
    # TREE16 keeps the in-tile tree in fp16 (only [BM,1] partials converted to
    # fp32); the fp16->fp32 per-element convert is the fp16 slowdown on XPU.
    if ACC32:
        acc = tl.full([BLOCK_M, 1], value=1.0, dtype=tl.float32)
        for off in range(0, N, BLOCK_N):
            cols = off + tl.arange(0, BLOCK_N)[None, :]
            if TREE16:
                # fp16 native tile tree (BLOCK_N <= 512); partial converted
                if NEED_MASK:
                    mask = row_mask and (cols < N)
                    a = tl.load(inp + cols, mask, other=1.0)
                else:
                    a = tl.load(inp + cols)
                blk = tl.reduce(a, axis=1, combine_fn=reduce_mul)[:, None]
                acc = acc * blk.to(tl.float32)
            else:
                if NEED_MASK:
                    mask = row_mask and (cols < N)
                    a = tl.load(inp + cols, mask, other=1.0).to(tl.float32)
                else:
                    a = tl.load(inp + cols).to(tl.float32)
                blk = tl.reduce(a, axis=1, combine_fn=reduce_mul)[:, None]
                acc = acc * blk
    else:
        # int path: torch.prod promotes any int input to int64 and computes
        # the exact product in int64 (wrapping only in int64); the per-input
        # width must NOT wrap (e.g. int8 rows of -648 wrap to 120 in int8
        # but torch returns -648 in int64).
        acc = tl.full([BLOCK_M, 1], value=1, dtype=tl.int64)
        for off in range(0, N, BLOCK_N):
            cols = off + tl.arange(0, BLOCK_N)[None, :]
            if NEED_MASK:
                mask = row_mask and (cols < N)
                a = tl.load(inp + cols, mask, other=1).to(tl.int64)
            else:
                a = tl.load(inp + cols).to(tl.int64)
            blk = tl.reduce(a, axis=1, combine_fn=reduce_mul)[:, None]
            acc = acc * blk
    if NEED_MASK:
        tl.store(out, acc, row_mask)
    else:
        tl.store(out, acc)


@libentry()
@triton.jit
def prod_mid_block(inp, mid, BLOCK: tl.constexpr, ACC32: tl.constexpr):
    # one PID per BLOCK (<= 8192) contiguous unmasked chunk -> partial
    pid = ext.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    if ACC32:
        v = tl.load(inp + offs).to(tl.float32)
    else:
        v = tl.load(inp + offs).to(tl.int64)
    p = tl.reduce(v, axis=0, combine_fn=reduce_mul)
    tl.store(mid + pid, p)


@libentry()
@triton.jit
def prod_tailk(inp, out, START, WIDTH: tl.constexpr, ACC32: tl.constexpr):
    # exact pow2 slice [START, START+WIDTH) -> fully in-bounds, no mask
    offs = START + tl.arange(0, WIDTH)
    if ACC32:
        v = tl.load(inp + offs).to(tl.float32)
    else:
        v = tl.load(inp + offs).to(tl.int64)
    p = tl.reduce(v, axis=0, combine_fn=reduce_mul)
    tl.store(out, p)


@libentry()
@triton.jit
def prod_final(inp, out, WIDTH: tl.constexpr, ACC32: tl.constexpr):
    offs = tl.arange(0, WIDTH)
    if ACC32:
        v = tl.load(inp + offs).to(tl.float32)
    else:
        v = tl.load(inp + offs).to(tl.int64)
    p = tl.reduce(v, axis=0, combine_fn=reduce_mul)
    tl.store(out, p)


def _bm_tail(tr):
    # largest power of two dividing tr (> 0): a tail block of this height is
    # fully in-bounds, so the tail launch stays mask-free.
    b = 1
    while tr % (b * 2) == 0:
        b *= 2
    return b


@libentry()
@triton.jit
def prod_dim_chunk(
    inp,
    part,
    N,
    B0,
    C0,
    C: tl.constexpr,
    CHUNK: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    # program (c, mb): partial[rows, C0 + c] = prod(inp[rows, B0 + c*CHUNK : B0 + (c+1)*CHUNK]).
    # inp is [R, N] row-major; the host guarantees every row block is fully
    # in-bounds (two-level row split), so the kernel is 100% mask-free.
    c = ext.program_id(0)
    mb = ext.program_id(1)
    # keep the tile index in i32: i64 tensor index arithmetic OOMs the
    # uni_sram budget on this backend (rows*N <= 2^31 is enforced by the host)
    rows = (mb * BLOCK_M + tl.arange(0, BLOCK_M)).to(tl.int32)[:, None]
    inp = inp + rows * N + B0 + c * CHUNK
    acc = tl.full([BLOCK_M, 1], value=1.0, dtype=tl.float32)
    for off in range(0, CHUNK, BLOCK_N):
        cols = off + tl.arange(0, BLOCK_N)[None, :]
        a = tl.load(inp + cols).to(tl.float32)
        blk = tl.reduce(a, axis=1, combine_fn=reduce_mul)[:, None]
        acc = acc * blk
    tl.store(part + rows * C + C0 + c, acc)


@libentry()
@triton.jit
def prod_dim_single(
    inp,
    out,
    N,
    CHUNK: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    # single-chunk (C == 1) variant: the whole reduction fits one chunk, so
    # store straight into `out` (implicit f32 -> out dtype cast).
    mb = ext.program_id(1)
    rows = (mb * BLOCK_M + tl.arange(0, BLOCK_M)).to(tl.int32)[:, None]
    inp = inp + rows * N
    acc = tl.full([BLOCK_M, 1], value=1.0, dtype=tl.float32)
    for off in range(0, CHUNK, BLOCK_N):
        cols = off + tl.arange(0, BLOCK_N)[None, :]
        a = tl.load(inp + cols).to(tl.float32)
        blk = tl.reduce(a, axis=1, combine_fn=reduce_mul)[:, None]
        acc = acc * blk
    tl.store(out + rows, acc)


def _pow2_decomp(r):
    parts = []
    while r:
        p = 1 << (r.bit_length() - 1)
        parts.append(p)
        r -= p
    return parts


def _reduce_partials(data, n, out, device, acc32):
    """Staged product of `n` fp32/int partials -> scalar `out`, mask-free
    (tails pow2-decomposed; final pad of 1.0s)."""
    with torch_device_fn.device(device):
        while n > _REDUCE_BLOCK:
            k = n // _REDUCE_BLOCK
            r = n - k * _REDUCE_BLOCK
            chunks = _pow2_decomp(r)
            sz = k + len(chunks)
            midn = torch.empty((sz,), dtype=data.dtype, device=device)
            if k:
                prod_mid_block[(k, 1)](
                    data, midn, _REDUCE_BLOCK, acc32, buffer_size_limit=2048
                )
            pos = k * _REDUCE_BLOCK
            for i, w in enumerate(chunks):
                prod_tailk[(1, 1)](
                    data,
                    midn[k + i : k + i + 1],
                    pos,
                    w,
                    acc32,
                    buffer_size_limit=2048,
                )
                pos += w
            data = midn
            n = sz
        width = triton.next_power_of_2(n)
        if width == n:
            prod_final[(1, 1)](data, out, width, acc32, buffer_size_limit=2048)
        else:
            pad = torch.full((width,), 1, dtype=data.dtype, device=device)
            if n:
                torch.ops.aten._copy_from(data, pad[:n], False)
            prod_final[(1, 1)](pad, out, width, acc32, buffer_size_limit=2048)


def _prod_flat(inp, out, device):
    numel = inp.numel()
    block = _REDUCE_BLOCK
    rows = numel // block
    res = numel - rows * block
    is_fp32 = inp.dtype == torch.float32
    acc32 = inp.dtype.is_floating_point
    tree16 = _tree16(inp.dtype)
    wdt = _work_dtype(inp.dtype)
    with torch_device_fn.device(device):
        if rows:
            tile = _pick_fast_tile(rows, block, is_fp32)
            bm = tile[0] if tile else 2
            if tree16:
                bn = _TREE16_BLOCK  # fp16 native trees: BN <= 512
            else:
                bn = 1024 if not is_fp32 else 512
            chunks = _pow2_decomp(res) if res else []
            mid = torch.empty((rows + len(chunks),), dtype=wdt, device=device)
            prod_row2d[(max(rows // bm, 1), 1)](
                inp,
                mid,
                rows,
                block,
                bm,
                bn,
                False,
                acc32,
                tree16,
                buffer_size_limit=2048,
            )
            pos = rows * block
            for i, w in enumerate(chunks):
                prod_tailk[(1, 1)](
                    inp,
                    mid[rows + i : rows + i + 1],
                    pos,
                    w,
                    acc32,
                    buffer_size_limit=2048,
                )
                pos += w
        else:
            chunks = _pow2_decomp(res)
            mid = torch.empty((len(chunks),), dtype=wdt, device=device)
            pos = 0
            for i, w in enumerate(chunks):
                prod_tailk[(1, 1)](
                    inp, mid[i : i + 1], pos, w, acc32, buffer_size_limit=2048
                )
                pos += w
        _reduce_partials(mid, mid.numel(), out, device, acc32)


def prod(inp, *, dtype=None):
    logger.debug("GEMS_KUNLUNXIN PROD")
    if dtype is None:
        # torch.prod promotes integer inputs to int64 (exact product)
        dtype = torch.int64 if not inp.dtype.is_floating_point else inp.dtype
    numel = inp.numel()
    out = torch.empty([], dtype=dtype, device=inp.device)
    if numel == 0:
        out.fill_(1)
        return out
    if numel == 1:
        with torch_device_fn.device(inp.device):
            torch.ops.aten._copy_from(inp.reshape([]), out, False)
        return out
    with torch_device_fn.device(inp.device):
        _prod_flat(inp, out, inp.device)
    return out


def prod_dim(inp, dim=None, keepdim=False, *, dtype=None):
    logger.debug("GEMS_KUNLUNXIN PROD_DIM")
    assert dim >= -inp.ndim and dim < inp.ndim, "Invalid dim"
    if dtype is None:
        dtype = torch.int64 if not inp.dtype.is_floating_point else inp.dtype
    shape = list(inp.shape)
    d = dim % inp.ndim
    N = shape[d]
    M = 1
    for s in shape[:d]:
        M *= s
    K = 1
    for s in shape[d + 1 :]:
        K *= s

    out_shape = shape.copy()
    out_shape[d] = 1
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
    # existing prod_row2d. Rows are split into full BLOCK_M blocks plus one
    # tail block whose height is the largest power-of-two divisor of the tail,
    # so every load/store is in-bounds and compiles mask-free (the masked
    # memory path costs ~2x on this backend).
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
                prod_dim_single[(1, 1)](
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
                    prod_dim_chunk[(main, 1)](
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
                    prod_dim_chunk[(1, 1)](
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
        out = torch.squeeze(out, d)
    return out
