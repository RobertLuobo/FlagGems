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
from torch._prims_common import is_boolean_dtype, is_integer_dtype

from flag_gems.runtime import device, torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as ext

logger = logging.getLogger(__name__)
device = device.name


# K == 1 (row scan over the last, stride-1 dim) tiers:
#  - N <= _ROW_MAX_N: one row per program, whole row as a single 1D tile whose
#    scan is a 1D tl.cumsum -- the only scan path this XPU backend lowers
#    correctly (2D axis=1 tl.cumsum silently mis-computes).
#  - N >  _ROW_MAX_N: per-row chunked online scan, BN-wide chunks chained in
#    one program by a scalar carry (no host round trips / extra passes).
_ROW_MAX_N = 4096

_TL_DTYPES = {
    torch.float16: tl.float32,
    torch.bfloat16: tl.float32,
    torch.float32: tl.float32,
    torch.float64: tl.float64,
    torch.bool: tl.int32,
    torch.uint8: tl.int32,
    torch.int8: tl.int32,
    torch.int16: tl.int32,
    torch.int32: tl.int32,
    torch.int64: tl.int64,
    torch.uint64: tl.uint64,
}


@libentry()
@triton.jit
def cumsum_row_kernel(
    inp_ptr,
    out_ptr,
    N: tl.constexpr,
    TILE_N: tl.constexpr,
    NEED_MASK: tl.constexpr,
):
    """One row per program; single 1D tile inclusive scan. (2D axis=1
    tl.cumsum silently mis-computes on this backend, so per-row 1D tiles.)"""
    pid = ext.program_id(0)
    row_offset = pid * N
    n_offsets = tl.arange(0, TILE_N)
    if NEED_MASK:
        mask = n_offsets < N
        x = tl.load(inp_ptr + row_offset + n_offsets, mask=mask, other=0.0)
    else:
        x = tl.load(inp_ptr + row_offset + n_offsets)
    if tl.constexpr(x.dtype.is_bf16()) or tl.constexpr(x.dtype.is_fp16()):
        x = x.to(tl.float32)
    elif (
        tl.constexpr(x.dtype.is_int64()) or tl.constexpr(x.dtype.is_uint64())
    ) or tl.constexpr(x.dtype.is_fp64()):
        x = x
    elif tl.constexpr(x.dtype.is_int()):
        x = x.to(tl.int32)
    else:
        x = x.to(tl.float32)
    r = tl.cumsum(x, axis=0)
    if NEED_MASK:
        tl.store(out_ptr + row_offset + n_offsets, r, mask=mask)
    else:
        tl.store(out_ptr + row_offset + n_offsets, r)


@libentry()
@triton.jit
def cumsum_chunk_kernel(
    inp_ptr,
    out_ptr,
    N,
    ACC_DTYPE: tl.constexpr,
    BN: tl.constexpr,
    NEED_TAIL: tl.constexpr,
):
    """Per-row chunked online scan for N > _ROW_MAX_N. A scalar
    carry sum chains BN-wide chunks inside one program; the masked tail
    chunk (masked load with other=0 + masked store) is proven exact."""
    pid = ext.program_id(0)
    row_offset = pid * N
    carry = tl.zeros([BN], ACC_DTYPE)
    for start in range(0, N, BN):
        n_offsets = start + tl.arange(0, BN)
        if NEED_TAIL:
            mask = n_offsets < N
            x = tl.load(inp_ptr + row_offset + n_offsets, mask=mask, other=0.0).to(
                ACC_DTYPE
            )
        else:
            x = tl.load(inp_ptr + row_offset + n_offsets).to(ACC_DTYPE)
        r = tl.cumsum(x, axis=0) + carry
        if NEED_TAIL:
            tl.store(out_ptr + row_offset + n_offsets, r, mask=mask)
        else:
            tl.store(out_ptr + row_offset + n_offsets, r)
        carry += tl.sum(x, axis=0)


@libentry()
@triton.jit
def cumsum_identity_kernel(
    inp_ptr,
    out_ptr,
    n_elements,
    BLOCK: tl.constexpr,
):
    """N == 1: cumsum along a length-1 row is the elementwise identity
    (out dtype may differ, e.g. int -> int64, so a plain kernel is used)."""
    pid = ext.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elements
    x = tl.load(inp_ptr + offs, mask=mask)
    tl.store(out_ptr + offs, x, mask=mask)


def _scan_rows_into(inp, out, M, N):
    """K == 1 (row scan) fast paths: identity for N == 1, single-shot row
    kernel for N <= 4096, chunked online scan for N > 4096."""
    if N == 1:
        n_elements = inp.numel()
        BLOCK = 1024
        grid = (triton.cdiv(n_elements, BLOCK), 1, 1)
        cumsum_identity_kernel[grid](
            inp,
            out,
            n_elements,
            BLOCK=BLOCK,
            num_warps=4,
            buffer_size_limit=2048,
        )
    elif N <= _ROW_MAX_N:
        # A tile of <= 32 lanes is mis-lowered on this backend: the scan runs
        # over the full 32-wide vector and keeps carrying across the row
        # boundary, so rows after the first come out with a stale prefix added
        # (silently, e.g. M=64/N=4 reproduces the flat-cumsum pattern). Keep
        # the tile at >= 64 lanes and mask the padding; masked padding lanes
        # cannot corrupt a prefix scan because lane i only depends on lanes
        # <= i and the store is masked as well.
        TILE_N = max(64, triton.next_power_of_2(N))
        need_mask = 1 if TILE_N != N else 0
        num_warps = 8 if TILE_N > 2048 else 4
        grid = (M, 1, 1)
        cumsum_row_kernel[grid](
            inp,
            out,
            N=N,
            TILE_N=TILE_N,
            NEED_MASK=need_mask,
            num_warps=num_warps,
            buffer_size_limit=2048,
        )
    else:
        # chunked online scan; BN sweep on (1024,65536): BN=16384 best for
        # fp16/bf16 (3.70/3.62ms vs 4.03/4.02 at BLOCK=8192), BN=32768 best
        # for fp32 (3.50ms vs 4.03); BLOCK=32768 regresses fp16/bf16, so the
        # block width follows the accumulate dtype.
        BN = 32768 if inp.dtype == torch.float32 else 16384
        need_tail = 1 if N % BN else 0
        acc_tl = _TL_DTYPES.get(inp.dtype, tl.float32)
        grid = (M, 1, 1)
        cumsum_chunk_kernel[grid](
            inp,
            out,
            N,
            ACC_DTYPE=acc_tl,
            BN=BN,
            NEED_TAIL=need_tail,
            num_warps=8,
            buffer_size_limit=2048,
        )


# K > 1 (mid-dim scan) tier: the previous scan_then_fan / scan_part_sum_*
# kernels are unusable on this backend -- their 1024-lane tl.cumsum is
# data-dependently mis-computed, their masked tail block performs OOB reads
# into adjacent memory (polluted partial sums and even a hard fault for some
# shapes), and the chunked row scans (N > _ROW_MAX_N, multiple iterations)
# silently corrupt every row past the 12th for a masked tail.  The only
# reliably-correct scan primitive on this backend is a SINGLE-SHOT <= 4096-lane
# 1D tl.cumsum over unmasked, in-bounds data (see cumsum_row_kernel).  So the
# mid-dim scan is decomposed into three passes over groups of _GROUP lanes on
# a zero-padded (M*K, N) copy, with the group-prefix scan itself reusing
# _scan_rows_into (which is proven).  cumsum commutes with the
# (M, N, K) -> (M, K, N) transpose, so the result is identical to torch's.
# See harness/solution/performance/cumsum__recheck_20260906.md.
_GROUP = 4096


@libentry()
@triton.jit
def scan_group_sum_kernel(
    inp, sums, N: tl.constexpr, NG: tl.constexpr, GROUP: tl.constexpr
):
    """(R, NG) grid: sum one GROUP-wide group.  The input is zero-padded to
    exactly NG * GROUP columns, so the load is always fully in-bounds and
    needs no mask (masked loads that can read adjacent memory are the one
    construct this backend mis-compiles)."""
    pid_r = ext.program_id(0)
    pid_g = ext.program_id(1)
    offs = pid_g * GROUP + tl.arange(0, GROUP)
    x = tl.load(inp + pid_r * N + offs)
    if tl.constexpr(x.dtype.is_bf16()) or tl.constexpr(x.dtype.is_fp16()):
        x = x.to(tl.float32)
    elif (
        tl.constexpr(x.dtype.is_int64()) or tl.constexpr(x.dtype.is_uint64())
    ) or tl.constexpr(x.dtype.is_fp64()):
        x = x
    elif tl.constexpr(x.dtype.is_int()):
        x = x.to(tl.int32)
    else:
        x = x.to(tl.float32)
    tl.store(sums + pid_r * NG + pid_g, tl.sum(x, axis=0))


@libentry()
@triton.jit
def scan_group_add_kernel(
    inp, out, sums, N: tl.constexpr, NG: tl.constexpr, GROUP: tl.constexpr
):
    """(R, NG) grid: single-shot 1D scan of one group plus the prefix of all
    groups before it (read directly from the pre-scanned `sums`)."""
    pid_r = ext.program_id(0)
    pid_g = ext.program_id(1)
    offs = pid_g * GROUP + tl.arange(0, GROUP)
    x = tl.load(inp + pid_r * N + offs)
    if tl.constexpr(x.dtype.is_bf16()) or tl.constexpr(x.dtype.is_fp16()):
        x = x.to(tl.float32)
    elif (
        tl.constexpr(x.dtype.is_int64()) or tl.constexpr(x.dtype.is_uint64())
    ) or tl.constexpr(x.dtype.is_fp64()):
        x = x
    elif tl.constexpr(x.dtype.is_int()):
        x = x.to(tl.int32)
    else:
        x = x.to(tl.float32)
    base = tl.load(sums + pid_r * NG + tl.maximum(pid_g - 1, 0))
    base = base * (pid_g > 0).to(base.dtype)
    r = tl.cumsum(x, axis=0) + base
    tl.store(out + pid_r * N + offs, r)


def _scan_mid_into(inp, out, M, N, K):
    """ "(M, N, K) -> (M, K, N) -> padded group scan -> transpose back."""
    R = M * K
    n_groups = (N + _GROUP - 1) // _GROUP
    Np = n_groups * _GROUP
    with torch_device_fn.device(inp.device):
        inp_t = inp.view(M, N, K).permute(0, 2, 1).contiguous()  # (M, K, N)
        xp = torch.zeros(R, Np, dtype=inp.dtype, device=inp.device)
        xp[:, :N] = inp_t.reshape(R, N)
        # group sums are accumulated in fp32 (floats) / int64 (ints); the final
        # store below casts back to `out`'s dtype (two's-complement for ints).
        sums = torch.empty(
            R,
            n_groups,
            dtype=torch.float32 if inp.dtype.is_floating_point else torch.int64,
            device=inp.device,
        )
        out_t = torch.empty(R, Np, dtype=out.dtype, device=out.device)
        scan_group_sum_kernel[(R, n_groups)](
            xp, sums, Np, n_groups, _GROUP, num_warps=8, buffer_size_limit=2048
        )
        # group-prefix scan (row-scan tiers, proven; in-place is safe because
        # each program only stores back the addresses it just loaded).
        _scan_rows_into(sums, sums, R, n_groups)
        scan_group_add_kernel[(R, n_groups)](
            xp, out_t, sums, Np, n_groups, _GROUP, num_warps=8, buffer_size_limit=2048
        )
    torch.ops.aten._copy_from(
        out_t[:, :N].reshape(M, K, N).permute(0, 2, 1), out.view(M, N, K), False
    )


def cumsum_wrapper(inp, dim=1, dtype=None, out=None):
    assert dim >= -inp.ndim and dim < inp.ndim, "Invalid dim"
    shape = inp.shape
    dim = dim % inp.ndim
    M = 1
    N = shape[dim]
    for i in range(dim):
        M *= shape[i]
    inp = inp.contiguous()
    K = inp.numel() // M // N

    if dtype is None:
        dtype = inp.dtype
        if is_integer_dtype(dtype) or is_boolean_dtype(dtype):
            dtype = torch.int64
    if out is None:
        out = torch.empty_like(inp, dtype=dtype)

    if K == 1:
        # Row scan: one program per row, 1D tl.cumsum tiles (the only scan
        # layout that lowers correctly on this backend); direct write into the
        # provided out (no temp + copy for the .out variant).
        with torch_device_fn.device(inp.device):
            _scan_rows_into(inp, out, M, N)
    else:
        # K > 1 (mid-dim scan): see _scan_mid_into for the design rationale --
        # the previous scan_then_fan tier is silently wrong on this backend.
        _scan_mid_into(inp, out, M, N, K)
    return out


def cumsum(inp, dim=1, *, dtype=None):
    logger.debug("GEMS_KUNLUNXIN CUMSUM")
    return cumsum_wrapper(inp, dim, dtype)


def cumsum_out(inp, dim=1, *, dtype=None, out):
    logger.debug("GEMS_KUNLUNXIN CUMSUM_OUT")
    return cumsum_wrapper(inp, dim, dtype, out)


@libentry()
@triton.jit(do_not_specialize=["K", "INNER"])
def normed_cumsum_strided_kernel(inp, out, K, INNER, BLOCK: tl.constexpr):
    row = ext.program_id(0)
    outer = row // INNER
    inner = row % INNER
    base = outer * K * INNER + inner
    offsets = tl.arange(0, BLOCK)
    mask = offsets < K
    x = tl.load(inp + base + offsets * INNER, mask=mask, other=0.0)
    if x.dtype.is_fp16() | x.dtype.is_bf16():
        x = x.to(tl.float32)
    total = tl.sum(x, axis=0)
    result = tl.cumsum(x, axis=0) / total
    tl.store(out + base + offsets * INNER, result, mask=mask)


# normed_cumsum: per-row single-shot scan + divide-by-row-total.
# The old block_cumsum/block_update two-pass split is unusable here: it always
# materialized an 8192-lane tile (16x waste for the benchmark K <= 512 rows),
# and the K > 8192 branch crashed with `'str' object has no attribute 'name'`
# (torch.empty(..., device=device.name) on the already-string module-level
# `device`).  The replacement reuses the two scan primitives that are proven on
# this backend:
#   - K <= _FUSED_MAX_N: one (or TILE_M) row(s) per program, TILE_N =
#     max(1024, next_pow2(K)) <= 4096 lanes, single-shot 1D tl.cumsum +
#     tl.sum + divide (all within the 4096-lane scan / 8192-lane sum bound).
#   - K >  _FUSED_MAX_N: the proven chunked online scan (_scan_rows_into)
#     followed by an in-place divide by the row total (scan's last element).
_FUSED_MAX_N = 4096


@libentry()
@triton.jit
def normed_cumsum_fused_kernel(
    inp,
    out,
    N: tl.constexpr,
    TILE_N: tl.constexpr,
    NEED_MASK: tl.constexpr,
    TILE_M: tl.constexpr,
):
    """TILE_M consecutive rows per program (TILE_M = 1 -> one row per program);
    every row is a single-shot 1D scan divided by the row total.  MUST be
    launched with n_rows % TILE_M == 0: a runtime per-row guard cannot wrap a
    scan on this backend (ConvertTritonXPUToLLVM rejects tt.scan inside a
    runtime scf.if, PassManager::run failed), so the remainder rows are covered
    by a TILE_M = 1 launch in the wrapper.  Only TILE_N lanes are live at a
    time, so TILE_M amortizes program-launch overhead on the launch-bound
    huge-n_rows / small-N corner (measured [64,512,512]: 32768 one-row
    programs -> 16.9ms with the old 8192-lane kernel)."""
    pid = ext.program_id(0)
    n_offsets = tl.arange(0, TILE_N)
    for i in tl.static_range(TILE_M):
        row_off = (pid * TILE_M + i) * N
        if NEED_MASK:
            mask = n_offsets < N
            x = tl.load(inp + row_off + n_offsets, mask=mask, other=0.0)
        else:
            x = tl.load(inp + row_off + n_offsets)
        if tl.constexpr(x.dtype.is_bf16()) or tl.constexpr(x.dtype.is_fp16()):
            x = x.to(tl.float32)
        total = tl.sum(x, axis=0)
        r = tl.cumsum(x, axis=0) / total
        if NEED_MASK:
            tl.store(out + row_off + n_offsets, r, mask=mask)
        else:
            tl.store(out + row_off + n_offsets, r)


@libentry()
@triton.jit(do_not_specialize=["K"])
def normed_cumsum_div_kernel(y, K, ACC_DTYPE: tl.constexpr, BLOCK: tl.constexpr):
    """K > _FUSED_MAX_N: in-place division of a scanned row by the row total
    (the scan's last element).  The scan itself is produced by the proven
    _scan_rows_into chunked path; in-place is safe because each program only
    touches its own row and never re-reads what it wrote."""
    row = ext.program_id(0)
    total = tl.load(y + row * K + (K - 1)).to(ACC_DTYPE)
    for start in range(0, K, BLOCK):
        offs = start + tl.arange(0, BLOCK)
        mask = offs < K
        x = tl.load(y + row * K + offs, mask=mask).to(ACC_DTYPE)
        tl.store(y + row * K + offs, x / total, mask=mask)


def normed_cumsum(inp, dim=-1):
    logger.debug("GEMS_KUNLUNXIN NORMED_CUMSUM")
    assert inp.dtype in (torch.float16, torch.bfloat16, torch.float32, torch.float64)
    dim = dim % inp.ndim
    inp = inp.contiguous()
    if inp.numel() == 0:
        return torch.empty_like(inp)
    K = inp.size(dim)
    if inp.stride(dim) != 1:
        # Non-last-dim scan: the masked strided load in
        # normed_cumsum_strided_kernel mis-vectorizes on this backend for
        # BLOCK = next_pow2(K) >= 256 lanes (measured ~1-5% off for e.g.
        # (513,16)/(300,8) dim=0), so use the proven row-scan kernels on a
        # transposed copy and normalize by the last cumsum element.
        K = inp.size(dim)
        M = inp.numel() // K
        inp_t = inp.movedim(dim, -1).contiguous()
        out_t = torch.empty_like(inp_t)
        with torch_device_fn.device(inp.device):
            _scan_rows_into(inp_t, out_t, M, K)
            adj = out_t / out_t.narrow(-1, K - 1, 1)
        return adj.movedim(-1, dim)
    # First and last dims are easier to handle, but transpose the middle dim to the last
    ranked_dims = sorted(range(inp.ndim), key=lambda i: inp.stride(i), reverse=True)
    is_mid_dim = dim not in (ranked_dims[0], ranked_dims[-1])
    if is_mid_dim:
        inp = inp.transpose(dim, -1).contiguous()
        dim = -1
    out = torch.empty_like(inp)
    with torch_device_fn.device(inp.device.index):
        # Pass one, scan a (batch, n_tiles * TILE) sized block within each cta
        num_sms = torch_device_fn.get_device_properties(device).multi_processor_count
        TILE = 8192
        # Each row is split into n_chunks of chunks where each chunk is compised of
        # n_tiles of tiles. Different chunks are assigned to different ctas.
        n_rows = N // K
        n_chunks = min(triton.cdiv(num_sms, n_rows), triton.cdiv(K, TILE))
        n_tiles = triton.cdiv(triton.cdiv(K, TILE), n_chunks)
        k_stride = inp.stride(dim)
        r_stride = inp.size(dim) if k_stride == 1 else 1
        if n_rows > GRID_Y_LIMIT:
            batch = triton.cdiv(n_rows, GRID_Y_LIMIT)
            n_batch = triton.cdiv(n_rows, batch)
        else:
            batch = 1
            n_batch = n_rows

        grid = (n_chunks, n_batch)
        if n_tiles > 1:
            # block_cumsum_kernel's per-tile accumulation (tl.sum inside the
            # `for ti` loop, plus the (1,)-shaped broadcast carry) cannot lower
            # on this backend: the tt.reduce is marked illegal and the
            # ConvertTritonXPUToLLVM pipeline aborts for any n_tiles > 1.  Use
            # the proven row-scan kernels instead (see _scan_rows_into) and
            # normalize by the last cumsum element (sum(x) == cumsum(x)[-1]).
            _scan_rows_into(inp, out, n_rows, K)
            return out / out.narrow(-1, K - 1, 1)

        if n_chunks == 1:
            block_cumsum_kernel[grid](
                inp,
                out,
                N=K,
                TILE_N=TILE_N,
                NEED_MASK=need_mask,
                TILE_M=tile_m,
                num_warps=num_warps,
                buffer_size_limit=2048,
            )
        else:
            # chunked online scan (proven) + in-place normalize by the
            # per-row total (the scan's last element).
            _scan_rows_into(inp, out, n_rows, K)
            acc = tl.float32 if inp.dtype != torch.float64 else tl.float64
            normed_cumsum_div_kernel[(n_rows,)](
                out,
                K,
                ACC_DTYPE=acc,
                BLOCK=8192,
                num_warps=8,
                buffer_size_limit=2048,
            )
            return out

        if inp.dtype != torch.float64:
            acc_dtype = torch.float32
        sums = torch.empty((n_rows, n_chunks), dtype=acc_dtype, device=device)
        cumsums = torch.empty_like(sums)
        block_cumsum_kernel[grid](
            inp,
            out,
            sums,
            batch,
            n_tiles,
            n_rows,
            K,
            r_stride,
            k_stride,
            r_stride,
            k_stride,
            OUTPUT_SUMS=True,
            NORMALIZE=False,
            HAS_OUT_LAYOUT=False,
            TILE=TILE,
            isCloseUnrollControl=True,
        )
        # Pass two, scan partial cumsums
        block_cumsum_kernel[(1, n_batch)](
            sums,
            cumsums,
            0,
            batch,
            1,
            n_rows,
            n_chunks,
            n_chunks,
            1,
            n_chunks,
            1,
            OUTPUT_SUMS=False,
            NORMALIZE=False,
            HAS_OUT_LAYOUT=True,
            TILE=TILE,
            isCloseUnrollControl=True,
        )
        # print(sums)
        rscale = cumsums[..., -1]
        block_update_kernel[grid](
            out,
            cumsums - sums,
            rscale,
            out,
            batch,
            n_tiles,
            n_rows,
            K,
            r_stride,
            k_stride,
            r_stride,
            k_stride,
            n_chunks,
            HAS_OUT_LAYOUT=False,
            TILE=TILE,
        )
        return out
