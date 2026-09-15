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
import math
from collections import namedtuple

import torch
import triton
import triton.language as tl

# from flag_gems import runtime
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as ext
from flag_gems.utils.limits import get_dtype_max

logger = logging.getLogger(__name__)

# NOTE (kunlunxin/XPU): performance recipe (2026-08-17) follows the
# amax/amin 2026-08-16 closure for the value-only paths and the argmin
# 2026-08-11 "packed index" closure for the min.dim (values, indices) path:
#   * masked loading uses `other=+inf` everywhere (NOT get_dtype_max), so
#     all-+inf blocks stay +inf (tests/test_min.py::test_min_all_inf);
#   * NEED_MASK constexpr fast 2D row-reduction: mask-free tiles when M and N
#     both divide the picked [BLOCK_M, BLOCK_N] (skips the XPU masked-memory
#     slow path);
#   * the reduced dim is brought innermost with the native strided copy
#     (`torch.ops.aten._copy_from`; flag_gems does not override _copy_from)
#     instead of the slow gems `.contiguous()` override;
#   * flat (dim=None) reduction = rows-of-8192 with the 2D tile kernel plus a
#     staged 8192-wide mid reduce -- the HEAD staged `min_kernel_1` path
#     (tl.reduce + combine_fn) stalls in the XPU compiler on the very first
#     launch (probe: >10min on a 2-element tensor; matches the 2026-08-08 /
#     2026-08-10 BLOCKED records), so the flat path is replaced entirely;
#   * N == 1 identity: `_copy_from` for values + a zeros index tensor.
# min.dim needs values AND indices. A per-element index-carrying reduce
# (`tl.min(... return_indices=True)` or a packed (value, index) int64 word)
# is ~15-20x slower than a value-only reduce on this XPU, so like argmin we
# split it into three passes.  Two fast-path variants share the same
# pass-2/pass-3 structure and differ only in the pass-1 tiling / part_val
# layout; A/B picks the variant per shape (see _NEW_MIN_VR):
#   1a. min_chunk_val_kernel (wide views, VR >= 65536): the (M2, N) matrix is
#       a contiguous (M2*nc, CHUNK) reshape (CHUNK | N), so one amin-style
#       reduce-OUTSIDE value-only row min over that view yields the per-chunk
#       minima into 1-D part_val[r*nc + ic] (single tl.min per program
#       instead of one tl.min per chunk, ~1.9x faster on (1024, 2^20));
#   1b. min_split_kernel (narrow views, VERBATIM HEAD): one [BM, CHUNK] tile
#       per chunk, column-major part_val[row + ic * M] -- faster than the
#       merge-view form on (1024, 4096)/(4096, 4096)/(1024, 256);
#   2. min_chunk_kernel / min_chunk_kernel_col: per-row min over the nc chunk
#      minima (leftmost chunk wins ties on this XPU backend, matching torch
#      "first minimal").
#   3. min_scan_kernel / min_scan_kernel_col: re-read only the winning chunk,
#      pack (order-preserving fp32 bit-map << 30) | column into an int64 and
#      take the plain int64 min so the first minimal lane wins; NaN and +inf
#      map above every finite value, exactly the XPU device-native fmin
#      family semantics (amax/amin 2026-08-16 evidence; device native min
#      ignores NaN, an all-NaN row yields NaN/value and index 0).

_FULL_REDUCTION_BLOCK_SIZE = 8192
_FAST_MIN_N = 64
_FAST_BN_FP16 = (1024, 256, 512, 128, 64)
_FAST_BN_FP32 = (512, 256, 1024, 128, 64)
_FAST_BM_FP16 = (128, 64, 32, 256, 512, 16, 8, 4, 2)
_FAST_BM_FP32 = (64, 128, 32, 256, 512, 16, 8, 4, 2)
# chunk-major view (pass 1) tile preference: a view row is one CHUNK-wide
# block, so the load loop has CHUNK / BLOCK_N trips; 512 is the widest tile
# that still pipelines (>= 2 trips) for the common CHUNK = 1024, 1024 only
# appears as the single-trip last resort for very narrow views.
_VIEW_BN_FP16 = (512, 256, 128, 64, 1024)
_VIEW_BN_FP32 = (512, 256, 128, 64, 1024)
_VIEW_BM_FP16 = (128, 64, 32, 256, 512, 16, 8, 4, 2)
_VIEW_BM_FP32 = (64, 128, 32, 256, 512, 16, 8, 4, 2)
# XPU compile guard: tiny [BLOCK_M, BLOCK_N] tiles (e.g. [64, 64]) fail the
# TritonXPU `uni_sram` pass ("PassManager::run failed"). Only tiles with at
# least this many lanes are routed to the mask-free row kernels; smaller
# shapes fall back to the legacy masked kernel.
_MIN_FAST_TILE_LANES = 8192
# chunk-major view (min_chunk_val_kernel) pays off only when the view is wide
# (>= 2^16 view rows, i.e. M2*nc >= 65536): A/B on one card shows the
# merge-view form 1.9x faster than the HEAD split form at (1024, 2^20) but
# 1.4-2.1x slower at (1024, 4096)/(4096, 4096)/(1024, 256), so narrower
# views keep the HEAD min_split_kernel form verbatim.
_NEW_MIN_VR = 65536


def _is_fast_dtype(dtype):
    return dtype in (torch.float16, torch.float32, torch.bfloat16)


def _pick_chunk_tile(VR, CHUNK, is_fp32):
    """Return (BLOCK_M, BLOCK_N) for pass 1 over the chunk-major view
    (VR, CHUNK) with VR % BLOCK_M == 0, CHUNK % BLOCK_N == 0,
    BLOCK_M * BLOCK_N >= _MIN_FAST_TILE_LANES, and CHUNK / BLOCK_N >= 2 loop
    trips (single-trip kernels do not pipeline on this XPU, so they are only
    a last resort), or None when no such mask-free tile covers this shape
    (small shapes keep the legacy masked kernel)."""
    bns = _VIEW_BN_FP32 if is_fp32 else _VIEW_BN_FP16
    bms = _VIEW_BM_FP32 if is_fp32 else _VIEW_BM_FP16
    for min_trips in (2, 1):
        for bn in bns:
            if CHUNK % bn != 0 or CHUNK // bn < min_trips:
                continue
            for bm in bms:
                if VR % bm == 0 and bm * bn >= _MIN_FAST_TILE_LANES:
                    return bm, bn
    return None


def _pick_fast_tile(M, N, is_fp32):
    """Return (BLOCK_M, BLOCK_N) with M % BLOCK_M == 0, N % BLOCK_N == 0 and
    BLOCK_M * BLOCK_N >= _MIN_FAST_TILE_LANES, or None when no such mask-free
    tile covers this shape (small shapes keep the legacy masked kernel)."""
    if N < _FAST_MIN_N:
        return None
    bns = _FAST_BN_FP32 if is_fp32 else _FAST_BN_FP16
    bms = _FAST_BM_FP32 if is_fp32 else _FAST_BM_FP16
    for bn in bns:
        if N % bn != 0:
            continue
        for bm in bms:
            if M % bm != 0:
                continue
            if bm * bn >= _MIN_FAST_TILE_LANES:
                return bm, bn


@libentry()
@triton.jit
def min_split_kernel(
    inp,
    part_val,
    M,
    N,
    BLOCK_M: tl.constexpr,
    BLOCK_CHUNK: tl.constexpr,
    NEED_MASK: tl.constexpr,
):
    # Pass 1 (HEAD form, narrow chunk-major views; value-only f32 min,
    # NaN/inf never win: same fmin family semantics as the XPU device-native
    # min; see module note).  One [BLOCK_M, BLOCK_CHUNK] tile per chunk, the
    # chunk minima are stored column-major at part_val[row + ic * M].
    pid = ext.program_id(0)
    rows = pid * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]
    inp = inp + rows.to(tl.int64) * N
    row_mask = rows < M
    ic = 0
    for off in range(0, N, BLOCK_CHUNK):
        cols = off + tl.arange(0, BLOCK_CHUNK)[None, :]
        if NEED_MASK:
            mask = row_mask and cols < N
            a = tl.load(inp + cols, mask=mask, other=float("inf")).to(tl.float32)
        else:
            a = tl.load(inp + cols).to(tl.float32)
        blk = tl.min(a, axis=1)[:, None]
        if NEED_MASK:
            tl.store(part_val + rows.to(tl.int64) + ic * M, blk, mask=row_mask)
        else:
            tl.store(part_val + rows.to(tl.int64) + ic * M, blk)
        ic += 1


@libentry()
@triton.jit
def min_chunk_kernel_col(
    part_val,
    best_c,
    M,
    NC,
    BLOCK_M: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    # Pass 2 (HEAD form): per row, argmin over the NC chunk minima (leftmost
    # chunk wins ties on this XPU backend, matching the torch "first minimal"
    # rule; NC is small, so return_indices is cheap).  part_val is
    # column-major [chunk + M * row]; the fast path only runs when NC <= 1024
    # -- an NC-walking loop with return_indices inside hits the XPU `tt.addptr`
    # encoding miscompilation (one-shot tile only).
    pid = ext.program_id(0)
    rows = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    row_mask = rows < M
    c_off = tl.arange(0, BLOCK_C)
    cmask = c_off < NC
    vals = tl.load(
        part_val + rows[:, None] + c_off[None, :].to(tl.int64) * M,
        mask=row_mask[:, None] and cmask[None, :],
        other=float("inf"),
    )
    _, bc = tl.min(vals, axis=1, return_indices=True)
    tl.store(best_c + rows.to(tl.int64), bc, mask=row_mask)


@libentry()
@triton.jit
def min_scan_kernel_col(
    inp,
    part_val,
    best_c,
    out_val,
    out_idx,
    M,
    N,
    BLOCK_M: tl.constexpr,
    BLOCK_CHUNK: tl.constexpr,
):
    # Pass 3 (HEAD form, column-major part_val): re-read only the winning
    # chunk per row and take the earliest lane equal to the chunk min.  The
    # 1/NC data slice is packed into (ordered fp32 value << 30) | column
    # words and reduced with a plain int64 min: the first minimal lane wins
    # (torch tie rule), -0.0 sorts below +0.0, and NaN/+inf bits sort above
    # every finite value so they never win (XPU fmin / device-native
    # semantics; an all-NaN row falls back to index 0 and the NaN value).
    pid = ext.program_id(0)
    rows = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    row_mask = rows < M
    c = tl.load(best_c + rows)  # [BM] int32
    base = rows.to(tl.int64) * N + c.to(tl.int64) * BLOCK_CHUNK
    cols = tl.arange(0, BLOCK_CHUNK)
    col_ok = cols[None, :] < (N - c * BLOCK_CHUNK)[:, None]
    a = tl.load(
        inp + base[:, None] + cols[None, :],
        mask=row_mask[:, None] and col_ok,
        other=float("inf"),
    ).to(tl.float32)
    u = a.to(tl.int32, bitcast=True)
    neg = u < 0
    ordered = tl.where(neg, ~u, u ^ -2147483648)
    # NOTE: the value/index word must use a 30-bit column shift (like the
    # argmin scan kernel). A full 64-bit `<< 32` is miscompiled by the XPU
    # backend (probed: the int64 min then picks wrong lanes); << 30 with
    # BLOCK_CHUNK <= 8192 < 2^30 is correct on every probe.
    pack = ((ordered.to(tl.int64) & 0xFFFFFFFF) << 30) | cols.to(tl.int64)
    blk = tl.min(pack, axis=1)
    pos = (blk & 0x3FFFFFFF) + c.to(tl.int64) * BLOCK_CHUNK
    # value of the winning lane: the chunk min computed by pass 1 (equal to
    # the winning lane's value, incl. the XPU fmin NaN semantics).
    m = tl.load(part_val + rows.to(tl.int64) + c.to(tl.int64) * M, mask=row_mask)
    tl.store(out_val + rows.to(tl.int64), m, mask=row_mask)
    tl.store(out_idx + rows.to(tl.int64), pos, mask=row_mask)


@libentry()
@triton.jit
def min_kernel_1(
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
    min_val = tl.min(inp_val)
    mid_ptr = mid + pid
    tl.store(mid_ptr, min_val)


def heur_m_block_size(args):
    return triton.next_power_of_2(triton.cdiv(args["M"], 12))  # cluster_num


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
def min_kernel(
    inp,
    out_value,
    out_index,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    # Legacy masked (values, indices) kernel for the non-fast paths (small N,
    # int dtypes). Preserved byte-for-byte from HEAD.
    # set offset
    pid_m = ext.program_id(0)
    pid_k = ext.program_id(1)
    m_offset = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)

    dtype = inp.type.element_ty
    # you just cannot create a function that return a tl.dtype in triton lang
    acc_type = tl.float32 if dtype is tl.bfloat16 else dtype
    max_value = get_dtype_max(dtype)
    min_values = tl.full([BLOCK_M], dtype=acc_type, value=max_value)
    argmin_values = tl.full([BLOCK_M], dtype=tl.int64, value=0)
    for start_n in range(0, N, BLOCK_N):
        n_offset = start_n + tl.arange(0, BLOCK_N)
        offset = m_offset[:, None] * N * K + n_offset[None, :] * K + pid_k
        mask = m_offset[:, None] < M and n_offset[None, :] < N
        inp_ptrs = inp + offset
        inp_vals = tl.load(inp_ptrs, mask=mask, other=max_value)
        local_min, local_argmin = tl.min(inp_vals, 1, return_indices=True)
        update = local_min < min_values
        min_values = tl.where(update, local_min, min_values)
        argmin_values = tl.where(update, start_n + local_argmin, argmin_values)

    offset_index = m_offset * K + pid_k
    out_value_ptrs = out_value + offset_index
    out_index_ptrs = out_index + offset_index
    mask1 = m_offset < M
    tl.store(out_value_ptrs, min_values, mask=mask1)
    tl.store(out_index_ptrs, argmin_values, mask=mask1)


@libentry()
@triton.jit
def min_kernel_2d(
    inp,
    out,
    M,
    N,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    NEED_MASK: tl.constexpr,
):
    # Value-only 2D row reduction (flat rows-of-8192 / plain dim reduction).
    pid = ext.program_id(0)
    rows = pid * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]
    inp = inp + rows * N
    out = out + rows
    row_mask = rows < M

    # Keep only a [BLOCK_M, 1] running accumulator and reduce each [BLOCK_M,
    # BLOCK_N] block along N *inside* the loop (reduce-INSIDE; the only form
    # that is numerically correct on this XPU -- see amax_kernel_2d notes).
    # NEED_MASK=False compiles to a mask-free kernel (M and N both divide by
    # the block sizes), avoiding the XPU masked-memory slow path entirely.
    acc = tl.full([BLOCK_M, 1], value=float("inf"), dtype=tl.float32)
    for off in range(0, N, BLOCK_N):
        cols = off + tl.arange(0, BLOCK_N)[None, :]
        if NEED_MASK:
            col_mask = cols < N
            mask = row_mask and col_mask
            a = tl.load(inp + cols, mask, other=float("inf")).to(tl.float32)
            a = tl.where(mask, a, float("inf"))
            blk = tl.min(a, axis=1)[:, None]
        else:
            a = tl.load(inp + cols).to(tl.float32)
            blk = tl.min(a, axis=1)[:, None]
        acc = tl.minimum(acc, blk)
    if NEED_MASK:
        tl.store(out, acc, row_mask)
    else:
        tl.store(out, acc)


def _pad_value(dtype):
    """Identity value for the min reduction: +inf for floats, dtype max otherwise."""
    if dtype.is_floating_point:
        return float("inf")
    return torch.iinfo(dtype).max


def _pad_buffer(src, pad_to, device):
    """Return a buffer of `pad_to` elements: [0, n) = src, [n, pad_to) filled
    with the reduction identity (+inf for floats, dtype-max for ints)."""
    n = src.numel()
    buf = torch.full((pad_to,), _pad_value(src.dtype), dtype=src.dtype, device=device)
    if n:
        torch.ops.aten._copy_from(src, buf[:n], False)
    return buf


def _reduce_to_scalar(src, out):
    """Repeatedly reduce `src` with 8192-wide blocks until a scalar remains.
    Every masked lane is eliminated by padding each stage to a multiple of
    8192 with the reduction identity (XPU masked tails read OOB -- backend
    limitation; see the module note)."""
    block = _FULL_REDUCTION_BLOCK_SIZE
    n = src.numel()
    data = src
    while n > block:
        n_pad = triton.cdiv(n, block) * block
        if n_pad > n:
            data = _pad_buffer(data, n_pad, data.device)
        mid_size = triton.cdiv(n, block)
        mid = torch.empty((mid_size,), dtype=data.dtype, device=data.device)
        min_kernel_1[(mid_size, 1)](
            data,
            mid,
            n,
            block,
            False,
            buffer_size_limit=2048,
        )
        data = mid
        n = mid_size
    if n < block:
        data = _pad_buffer(data, block, data.device)
    min_kernel_1[(1, 1)](
        data,
        out,
        n,
        block,
        False,
        buffer_size_limit=2048,
    )


def _min_flat(inp, out, device):
    """Full (dim=None) reduction over `inp` (any numel). Mask-free 2-D row
    tiles over rows-of-8192 plus a staged scalar reduce; every tail is reads
    are padded so no masked/OOB load can happen on the XPU backend."""
    block = _FULL_REDUCTION_BLOCK_SIZE
    numel = inp.numel()
    pad_val = _pad_value(inp.dtype)
    if numel <= block:
        buf = torch.full((block,), pad_val, dtype=inp.dtype, device=device)
        torch.ops.aten._copy_from(inp, buf[:numel], False)
        min_kernel_1[(1, 1)](
            buf,
            out,
            numel,
            block,
            False,
            buffer_size_limit=2048,
        )
        return
    rows = numel // block
    res = numel - rows * block
    is_fp32 = inp.dtype == torch.float32
    bn = 1024 if not is_fp32 else 256
    if block % bn != 0:
        bn = next((b for b in _FAST_BN_FP32 if block % b == 0), block)
    bm = 128 if not is_fp32 else 64
    if rows % bm != 0:
        bm = next(
            (
                m
                for m in _FAST_BM_FP32
                if rows % m == 0 and m * bn >= _MIN_FAST_TILE_LANES
            ),
            128,
        )
    # exact (rows // bm) * bm rows run mask-free on the input directly; the
    # leftover rows are copied into a padded [bm, block] buffer first so the
    # same mask-free kernel covers them.
    rows_exact = (rows // bm) * bm
    mid = torch.empty((rows + (1 if res else 0),), dtype=inp.dtype, device=device)
    with torch_device_fn.device(device):
        if rows_exact:
            min_kernel_2d[(rows_exact // bm, 1)](
                inp,
                mid,
                rows_exact,
                block,
                bm,
                bn,
                False,
                buffer_size_limit=2048,
            )
        if rows > rows_exact:
            # The leftover rows are copied into a fully-allocated padded
            # [bm, block] buffer and reduced with a fully mask-free launch
            # (M = bm covers the whole padded buffer; extra rows are the
            # reduction identity and cannot win). The bm row-minima go to a
            # scratch buffer first so the real mid output is never written
            # out of bounds. (XPU masked tails read OOB and can also return
            # wrong values -- backend limitation, see the module note.)
            tail_rows = rows - rows_exact
            tail_buf = torch.full(
                (bm * block,), pad_val, dtype=inp.dtype, device=device
            )
            torch.ops.aten._copy_from(
                inp[rows_exact * block : rows * block],
                tail_buf[: tail_rows * block],
                False,
            )
            tail_mid = torch.empty((bm,), dtype=inp.dtype, device=device)
            min_kernel_2d[(1, 1)](
                tail_buf,
                tail_mid,
                bm,
                block,
                bm,
                bn,
                False,
                buffer_size_limit=2048,
            )
            torch.ops.aten._copy_from(
                tail_mid[:tail_rows], mid[rows_exact : rows_exact + tail_rows], False
            )
        if res:
            res_buf = torch.full((block,), pad_val, dtype=inp.dtype, device=device)
            torch.ops.aten._copy_from(inp[rows * block :], res_buf[:res], False)
            min_kernel_1[(1, 1)](
                res_buf,
                mid[rows:],
                res,
                block,
                False,
                buffer_size_limit=2048,
            )
        _reduce_to_scalar(mid, out)


def min(inp):
    logger.debug("GEMS_KUNLUNXIN MIN")
    dtype = inp.dtype
    out = torch.empty([], dtype=dtype, device=inp.device)
    with torch_device_fn.device(inp.device):
        _min_flat(inp, out, inp.device)
    return out


@libentry()
@triton.jit
def min_chunk_val_kernel(
    inp,
    part_val,
    M,
    W,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    # Pass 1 (value-only f32 min, NaN/inf never win: same fmin family
    # semantics as the XPU device-native min; see module note). The (M2, N)
    # input is a contiguous (M2*nc, CHUNK) view; the loop tiles the CHUNK
    # columns with the amin_rows reduce-OUTSIDE accumulation and a single
    # tl.min per program (vs one tl.min per chunk before), and every chunk
    # minimum is stored as part_val[view_row].
    pid = ext.program_id(0)
    rows = pid * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]
    rows_c = tl.where(rows < M, rows, M - 1)
    inp = inp + rows_c.to(tl.int64) * W
    out = part_val + rows.to(tl.int64)
    row_mask = rows < M
    acc = tl.full([BLOCK_M, BLOCK_N], value=float("inf"), dtype=tl.float32)
    for off in range(0, W, BLOCK_N):
        cols = off + tl.arange(0, BLOCK_N)[None, :]
        a = tl.load(inp + cols).to(tl.float32)
        acc = tl.minimum(acc, a)
    tl.store(out, tl.min(acc, axis=1)[:, None], row_mask)


@libentry()
@triton.jit
def min_chunk_kernel(
    part_val,
    best_c,
    M,
    NC,
    BLOCK_M: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    # Pass 2: per row, argmin over the NC chunk minima (leftmost chunk wins
    # ties on this XPU backend, matching the torch "first minimal" rule;
    # return_indices is cheap here because pass 2 only touches the nc
    # per-row minima). part_val is chunk-major 1-D [row * NC + chunk]; a
    # single [BLOCK_M, BLOCK_C] tile covers the whole NC (the fast path only
    # runs when NC <= 1024 -- an NC-walking loop with return_indices inside
    # hits the XPU `tt.addptr` encoding miscompilation).
    pid = ext.program_id(0)
    rows = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    row_mask = rows < M
    c_off = tl.arange(0, BLOCK_C)
    cmask = c_off < NC
    vals = tl.load(
        part_val + rows[:, None].to(tl.int64) * NC + c_off[None, :],
        mask=row_mask[:, None] and cmask[None, :],
        other=float("inf"),
    )
    _, bc = tl.min(vals, axis=1, return_indices=True)
    tl.store(best_c + rows.to(tl.int64), bc, mask=row_mask)


@libentry()
@triton.jit
def min_scan_kernel(
    inp,
    part_val,
    best_c,
    out_val,
    out_idx,
    M,
    N,
    BLOCK_M: tl.constexpr,
    BLOCK_CHUNK: tl.constexpr,
):
    # Pass 3: re-read only the winning chunk per row and take the earliest
    # lane equal to the chunk min. The 1/NC data slice is packed into
    # (ordered fp32 value << 32) | column words and reduced with a plain
    # int64 min: the first minimal lane wins (torch tie rule), -0.0 sorts
    # below +0.0, and NaN/+inf bits sort above every finite value so they
    # never win (XPU fmin / device-native semantics; an all-NaN row falls
    # back to index 0 and the NaN value).
    pid = ext.program_id(0)
    rows = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    row_mask = rows < M
    c = tl.load(best_c + rows)  # [BM] int32
    base = rows.to(tl.int64) * N + c.to(tl.int64) * BLOCK_CHUNK
    cols = tl.arange(0, BLOCK_CHUNK)
    col_ok = cols[None, :] < (N - c * BLOCK_CHUNK)[:, None]
    a = tl.load(
        inp + base[:, None] + cols[None, :],
        mask=row_mask[:, None] and col_ok,
        other=float("inf"),
    ).to(tl.float32)
    u = a.to(tl.int32, bitcast=True)
    neg = u < 0
    ordered = tl.where(neg, ~u, u ^ -2147483648)
    # NOTE: the value/index word must use a 30-bit column shift (like the
    # argmin scan kernel). A full 64-bit `<< 32` is miscompiled by the XPU
    # backend (probed: the int64 min then picks wrong lanes); << 30 with
    # BLOCK_CHUNK <= 8192 < 2^30 is correct on every probe.
    pack = ((ordered.to(tl.int64) & 0xFFFFFFFF) << 30) | cols.to(tl.int64)
    blk = tl.min(pack, axis=1)
    pos = (blk & 0x3FFFFFFF) + c.to(tl.int64) * BLOCK_CHUNK
    # value of the winning lane: the chunk min computed by pass 1 (equal to
    # the winning lane's value, incl. the XPU fmin NaN semantics).
    m = tl.load(
        part_val + rows.to(tl.int64) * (N // BLOCK_CHUNK) + c.to(tl.int64),
        mask=row_mask,
    )
    tl.store(out_val + rows.to(tl.int64), m, mask=row_mask)
    tl.store(out_idx + rows.to(tl.int64), pos, mask=row_mask)


def min_dim(inp, dim=None, keepdim=False):
    logger.debug("GEMS_KUNLUNXIN MIN_DIM")
    assert dim >= -inp.ndim and dim < inp.ndim, "Invalid dim"
    shape = inp.shape
    dim = dim % inp.ndim
    N = shape[dim]
    M = math.prod(shape[:dim])
    K = inp.numel() // M // N

    shape_list = list(shape)
    shape_list[dim] = 1
    out_value = torch.empty(shape_list, dtype=inp.dtype, device=inp.device)
    out_index = torch.empty(shape_list, dtype=torch.int64, device=inp.device)

    if N == 1:
        # min along a size-1 dim is the identity (value = input, index = 0) --
        # the native strided copy engine instead of launching a kernel.
        with torch_device_fn.device(inp.device):
            torch.ops.aten._copy_from(inp, out_value, False)
        out_index.zero_()
        if not keepdim:
            out_value = torch.squeeze(out_value, dim)
            out_index = torch.squeeze(out_index, dim)
        Min_out = namedtuple("min", ["values", "indices"])
        return Min_out(values=out_value, indices=out_index)

    # ---- fast chunked three-pass path (floats only, N >= _FAST_MIN_N) ----
    if N >= _FAST_MIN_N and _is_fast_dtype(inp.dtype):
        # chunk width (pass 3 scans the winning chunk in one [BM, CHUNK] tile,
        # so CHUNK must be a power of two; the largest such divisor of N is
        # used so nc stays small).
        CHUNK = (
            1024
            if N % 1024 == 0
            else (512 if N % 512 == 0 else (256 if N % 256 == 0 else (128 if N % 128 == 0 else 64 if N % 64 == 0 else 0)))
        )
        M2 = M * K
        is_fp32 = inp.dtype == torch.float32
        # (M2, N) is a contiguous (M2 * (N // CHUNK), CHUNK) reshape, so pass 1
        # can run as an amin-style reduce-OUTSIDE value-only row min over that
        # chunk-major view (row-major part_val[r*nc + ic]).  This merge-view
        # form is only faster than the HEAD split reduce for wide views
        # (VR >= _NEW_MIN_VR; e.g. (1024, 65536) and (1024, 2^20)); for
        # narrower views the HEAD min_split_kernel form wins the A/B, so they
        # keep it verbatim.  The fast path also needs enough rows (pass 2/3
        # grids) and a single-tile pass 2 (NC <= 1024); flat 1-D shapes
        # (M2 == 1) and >1024-chunk rows keep the legacy path (identical
        # behavior to before).
        if CHUNK:
            nc = N // CHUNK
            VR = M2 * nc
            if M2 >= 8 and nc <= 1024 and VR >= _NEW_MIN_VR:
                tile = _pick_chunk_tile(VR, CHUNK, is_fp32)
                if tile is not None:
                    block_m, block_n = tile
                    # Bring the reduced dim innermost (same order as
                    # dim_compress) and materialize with the native strided
                    # copy (not gems contiguous).
                    perm = [d for d in range(inp.dim()) if d != dim] + [dim]
                    view = inp.permute(perm)
                    if view.is_contiguous():
                        src = view
                    else:
                        src = torch.empty(
                            list(view.shape), dtype=inp.dtype, device=inp.device
                        )
                        with torch_device_fn.device(inp.device):
                            torch.ops.aten._copy_from(view, src, False)
                    part_val = torch.empty(VR, dtype=torch.float32, device=inp.device)
                    best_c = torch.empty((M2,), dtype=torch.int32, device=inp.device)
                    out_flat = out_value.reshape(-1)
                    out_idx_flat = out_index.reshape(-1)
                    bm2 = (
                        32
                        if M2 % 32 == 0
                        else (16 if M2 % 16 == 0 else (8 if M2 % 8 == 0 else (4 if M2 % 4 == 0 else 2 if M2 % 2 == 0 else 1)))
                    )
                    bm3 = (
                        8
                        if M2 % 8 == 0
                        else (4 if M2 % 4 == 0 else (2 if M2 % 2 == 0 else 1))
                    )
                    with torch_device_fn.device(inp.device):
                        min_chunk_val_kernel[(VR // block_m,)](
                            src,
                            part_val,
                            VR,
                            CHUNK,
                            block_m,
                            block_n,
                            buffer_size_limit=2048,
                        )
                        min_chunk_kernel[(M2 // bm2,)](
                            part_val,
                            best_c,
                            M2,
                            nc,
                            bm2,
                            triton.next_power_of_2(nc),
                            buffer_size_limit=2048,
                        )
                        min_scan_kernel[(M2 // bm3,)](
                            src,
                            part_val,
                            best_c,
                            out_flat,
                            out_idx_flat,
                            M2,
                            N,
                            bm3,
                            CHUNK,
                            buffer_size_limit=2048,
                        )
                    if not keepdim:
                        out_value = torch.squeeze(out_value, dim)
                        out_index = torch.squeeze(out_index, dim)
                    Min_out = namedtuple("min", ["values", "indices"])
                    return Min_out(values=out_value, indices=out_index)

        # HEAD split path (narrow chunk-major views, verbatim control flow):
        # one [BLOCK_M, BLOCK_CHUNK] tile per chunk, column-major part_val.
        tile = _pick_fast_tile(M2, N, is_fp32)
        if tile is not None:
            block_m, block_n = tile
            grid_m = M2 // block_m
            need_mask = False
            # Bring the reduced dim innermost (same order as dim_compress) and
            # materialize with the native strided copy (not gems contiguous).
            perm = [d for d in range(inp.dim()) if d != dim] + [dim]
            view = inp.permute(perm)
            if view.is_contiguous():
                src = view
            else:
                src = torch.empty(list(view.shape), dtype=inp.dtype, device=inp.device)
                with torch_device_fn.device(inp.device):
                    torch.ops.aten._copy_from(view, src, False)
            nc = triton.cdiv(N, block_n)
            part_val = torch.empty((M2, nc), dtype=torch.float32, device=inp.device)
            best_c = torch.empty((M2,), dtype=torch.int32, device=inp.device)
            out_flat = out_value.reshape(-1)
            out_idx_flat = out_index.reshape(-1)
            with torch_device_fn.device(inp.device):
                min_split_kernel[(grid_m,)](
                    src,
                    part_val,
                    M2,
                    N,
                    block_m,
                    block_n,
                    need_mask,
                    buffer_size_limit=2048,
                )
                min_chunk_kernel_col[(grid_m,)](
                    part_val,
                    best_c,
                    M2,
                    nc,
                    block_m,
                    triton.next_power_of_2(nc),
                    buffer_size_limit=2048,
                )
                min_scan_kernel_col[(grid_m,)](
                    src,
                    part_val,
                    best_c,
                    out_flat,
                    out_idx_flat,
                    M2,
                    N,
                    block_m,
                    block_n,
                    buffer_size_limit=2048,
                )
            if not keepdim:
                out_value = torch.squeeze(out_value, dim)
                out_index = torch.squeeze(out_index, dim)
            Min_out = namedtuple("min", ["values", "indices"])
            return Min_out(values=out_value, indices=out_index)

    # ---- legacy path (unchanged HEAD behavior, incl. int dtypes) ----------
    inp = inp.contiguous()

    grid = lambda meta: (
        triton.cdiv(M, meta["BLOCK_M"]),
        K,
    )
    # NOTE (kunlunxin/XPU): the `tl.min(..., return_indices=True)` argmin
    # combine makes the `TritonXPUCoreTiling` pass emit incompatible `tt.reduce`
    # slice orders (order=[0,1] value vs order=[1,0] index) for most 2D (K==1)
    # tiles, which fails compilation ("out of resource: uni_sram /
    # PassManager::run failed"). Closing core-tiling side-steps the buggy
    # layout for the argmin reduce and compiles every shape/dtype.
    isCloseCoreTiling = True
    with torch_device_fn.device(inp.device):
        min_kernel[grid](
            inp, out_value, out_index, M, N, K, isCloseCoreTiling=isCloseCoreTiling
        )
    if not keepdim:
        out_value = torch.squeeze(out_value, dim)
        out_index = torch.squeeze(out_index, dim)
    Min_out = namedtuple("min", ["values", "indices"])
    out = Min_out(values=out_value, indices=out_index)
    return out
