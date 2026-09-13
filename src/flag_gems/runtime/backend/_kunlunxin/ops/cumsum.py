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
#  - N in _FLAT_TM: flat multi-row sweep, TM consecutive rows packed into one
#    BN = TM*N lane 1D tile, scanned once and corrected per row (see
#    cumsum_flat_kernel).  This is the only tier that fixes the small-N
#    load-granularity bound: the one-row kernel loads only N * (dtype-size)
#    bytes per program (512B for N=128 / fp16 @ ~4GB/s) while the flat sweep
#    loads BN >= 1024 bytes per program (> 20GB/s), measured 6.2x on
#    (10000, 256) (see harness/solution/cumsum_/README.md).
#  - N <= _ROW_MAX_N: one row per program, whole row as a single 1D tile whose
#    scan is a 1D tl.cumsum -- the only scan path this XPU backend lowers
#    correctly (2D axis=1 tl.cumsum silently mis-computes).
#  - N >  _ROW_MAX_N: per-row chunked online scan, BN-wide chunks chained in
#    one program by a scalar carry (no host round trips / extra passes).
_ROW_MAX_N = 4096

# TM (rows per program) chosen per N so the flat sweep keeps BN = TM*N <= 4096,
# the widest single-shot scan measured exact on this backend.  N must be a
# power of two (the band correction relies on exact N-aligned boundaries and
# the lane count must be a power of two); other N fall through to the
# one-row / chunked tiers.
_FLAT_TM = {2: 8, 4: 8, 8: 8, 16: 8, 32: 8, 64: 8, 128: 8, 256: 4, 512: 2}

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
def cumsum_flat_kernel(
    inp_ptr,
    out_ptr,
    N: tl.constexpr,
    BN: tl.constexpr,
    TM: tl.constexpr,
):
    """TM consecutive rows (N elements each) flattened into one BN = TM*N lane
    1D tile, scanned once, then corrected per row: the global scan at lane i
    of row j equals (sum of all rows before j, a per-row constant b_j) plus the
    in-row inclusive scan, so r -= b_j on the j-th band of N lanes recovers
    the exact per-row scan.

    Memory access is fully in-bounds and unmasked (program p owns the
    contiguous block [p*BN, (p+1)*BN) and only ever writes back the addresses
    it loaded, so in-place use is safe), which lets the single 1D tl.cumsum --
    the only scan path this backend lowers correctly -- run at BN >= 1024
    lanes.  Only pure-1D ops are used: any 2D scan / reshape / transpose
    mis-lowers or hits UNREACHABLE on this backend (see _scan_rows_into)."""
    pid = ext.program_id(0)
    base = pid * BN
    idx = tl.arange(0, BN)
    x = tl.load(inp_ptr + base + idx)
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
    for k in tl.static_range(1, TM):
        b = tl.sum(tl.where(idx < k * N, x, 0), axis=0)
        r = r - tl.where((idx >= k * N) & (idx < (k + 1) * N), b, 0)
    tl.store(out_ptr + base + idx, r)


@libentry()
@triton.jit
def cumsum_chunk_scan_kernel(
    inp_ptr,
    out_ptr,
    sums_ptr,
    N,
    C,
    S,
    ACC_DTYPE: tl.constexpr,
    BN: tl.constexpr,
):
    """Unmasked single-shot scan + sum of one full BN-wide chunk of one row
    (BN | N, so every lane is in-bounds by construction).

    Deliberately contains NO runtime loop around the tl.cumsum and NO masked
    load: the previous chunked online scan (carry chained across multiple
    16k-lane scans in one program) corrupted rows past the 12th, and masked
    4096-lane loads mis-compile on this backend (garbage from the OOB/padding
    lanes of the load leaks into the scan and its sum -- reproduced
    deterministically: (64, 16896)/(40, 25600)/(20, 70000) with a partial
    last chunk are wrong, while every exact BN multiple is exact).  Partial
    chunks are therefore scanned on a zero-padded buffer by
    cumsum_chunk_tail_kernel, and the row prefix from all chunks before c is
    applied afterwards by cumsum_chunk_prefix_kernel.

    S is the row stride of `sums` (== C when there is no tail chunk, C+1
    otherwise); the chunk index c below parses with C while the sum goes to
    row * S + c so full-chunk sums and the tail sum (written by the host)
    share one contiguous (M, S) matrix."""
    pid = ext.program_id(0)
    row = pid // C
    c = pid % C
    n_offsets = c * BN + tl.arange(0, BN)
    x = tl.load(inp_ptr + row * N + n_offsets).to(ACC_DTYPE)
    r = tl.cumsum(x, axis=0)
    tl.store(out_ptr + row * N + n_offsets, r)
    tl.store(sums_ptr + row * S + c, tl.sum(x, axis=0))


@libentry()
@triton.jit
def cumsum_chunk_tail_kernel(
    inp_ptr,
    out_ptr,
    ACC_DTYPE: tl.constexpr,
    TL: tl.constexpr,
):
    """Scan of one zero-padded tail: exactly TL lanes, all in-bounds (the
    buffer is TL wide, padding lanes hold 0), hence unmasked.  No reduction
    inside: a tl.sum at 512-2048 lanes (the tail widths) mis-compiles on this
    backend while the scan itself is exact, so the chunk sum is read by the
    host as the scanned tail's last valid element (r[T-1] == sum(all T))."""
    pid = ext.program_id(0)
    n_offsets = tl.arange(0, TL)
    x = tl.load(inp_ptr + pid * TL + n_offsets).to(ACC_DTYPE)
    r = tl.cumsum(x, axis=0)
    tl.store(out_ptr + pid * TL + n_offsets, r)


@libentry()
@triton.jit
def cumsum_chunk_prefix_kernel(
    out_ptr,
    sums_ptr,
    N,
    CF,
    S,
    BN: tl.constexpr,
):
    """Add the scanned prefix of all chunks before chunk c (a per-row
    constant) to every element of full chunk c.  Pure elementwise
    read-modify-write: no reduction, no scan, so no shared-memory/barrier
    interaction; unmasked (c < N // BN, all lanes in-bounds).  The grid is
    (M * CF,): CF full chunks per row, rows of `sums` are S wide (S == CF
    without a tail, CF + 1 with one)."""
    pid = ext.program_id(0)
    row = pid // CF
    c = pid % CF
    prefix = tl.load(sums_ptr + row * S + tl.maximum(c - 1, 0))
    # Integer-exact select (a `0.0` literal would promote int64 sums to fp64
    # and lose precision above 2^52); values in `out`/`sums` are int64 for
    # int input, so no truncation happens on integer paths -- except the
    # in-place int16/int32 case, where the wrap is bit-identical to torch's
    # own in-place cumsum.  x is widened to prefix's dtype before the add:
    # a mixed i16+i64 vector add miscompiles at 4096 lanes on this backend
    # (LLVM "size mismatch when packing elements for LLVM struct").
    prefix = tl.where(c > 0, prefix, 0)
    n_offsets = c * BN + tl.arange(0, BN)
    x = tl.load(out_ptr + row * N + n_offsets)
    x = x.to(prefix.dtype)
    tl.store(out_ptr + row * N + n_offsets, x + prefix)


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
    """K == 1 (row scan) fast paths: identity for N == 1, flat multi-row sweep
    for power-of-two N <= 512 (see cumsum_flat_kernel), single-shot row kernel
    for N <= 4096, chunked online scan for N > 4096."""
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
    elif N in _FLAT_TM and _FLAT_TM[N] * N >= 64 and M >= _FLAT_TM[N]:
        tm = _FLAT_TM[N]
        n_full = (M // tm) * tm
        BN = tm * N
        grid = (n_full // tm, 1, 1)
        cumsum_flat_kernel[grid](
            inp,
            out,
            N=N,
            BN=BN,
            TM=tm,
            num_warps=8,
            buffer_size_limit=2048,
        )
        if n_full < M:
            # Remainder rows (M % tm): the flat sweep needs full TM-row
            # groups, so the tail is covered by the one-row tier below.  The
            # kernels index the row-major buffer flat (row * N + offset), so
            # the slice is taken on the 1D view -- dim 0 of an ndim > 2 input
            # has size shape[0], not M.
            inp = inp.reshape(-1).narrow(0, n_full * N, (M - n_full) * N)
            out = out.reshape(-1).narrow(0, n_full * N, (M - n_full) * N)
            M = M - n_full
            _scan_rows_into(inp, out, M, N)
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
        # N > _ROW_MAX_N: split into full BN=4096 chunks (exact multiples --
        # every load/store is unmasked and in-bounds, the only scan shape
        # proven exact at 4096 lanes on this backend) plus, if N is not a
        # multiple of 4096, a partial tail scanned on a zero-padded buffer
        # (so no kernel ever issues a masked 4096-lane load, which miscompiles
        # and leaks padding garbage into the scan).  Passes: per-(row, chunk)
        # scan + chunk sum; zero-padded full-width row scan of the chunk
        # sums (see the S-padding comment below); pure elementwise prefix add.
        #
        # Callers may pass a 1-D flattened buffer (M == 1: the dim-0 scan of a
        # 1-D tensor, e.g. the sparse paths of nonzero/nonzero_numpy; or the
        # 1-D narrows of the _FLAT_TM recursion).  The tail copy sites below
        # index the second axis (`inp[:, ...]`), which raises IndexError on a
        # 1-D input, so normalize to the row-major (M, N) view first (numel ==
        # M*N by construction; a no-op view for 2-D callers).
        inp = inp.reshape(M, N)
        out = out.reshape(M, N)
        BN = 4096
        C_full = N // BN
        T = N - C_full * BN
        C = C_full + (1 if T else 0)
        acc_tl = _TL_DTYPES.get(inp.dtype, tl.float32)
        if inp.dtype == torch.float64:
            sums_dtype = torch.float64
        elif inp.dtype.is_floating_point:
            sums_dtype = torch.float32
        elif inp.dtype in (torch.int64, torch.uint64):
            sums_dtype = inp.dtype
        else:
            sums_dtype = torch.int64
        # Chunk sums live in a zero-padded (M, S) matrix (S >= C).  The (M, C)
        # chunk-sum row scan must NOT use the masked 64+ lane row kernel on an
        # un-padded (M, C) matrix: with TILE_C > C its masked load reads past
        # the row (and past the tensor for the last row), and this backend's
        # masked loads leak OOB garbage into the scan and its stores --
        # (3, 16384) / (4, 16384) with C = 4 reproduced a nondeterministic
        # error growing by the chunk-sum magnitude on every invocation.
        # Padding to a full S columns of zeros and scanning unmasked at
        # exactly S lanes makes every load/store in-bounds (the proven-exact
        # <= 4096-lane full-width window); for C > 4096 the sums scan recurses
        # through _scan_rows_into, whose own chunked level pads again.
        if C > 4096:
            S = C
        else:
            S = max(64, triton.next_power_of_2(C))
        with torch_device_fn.device(inp.device):
            sums = torch.zeros(M, S, dtype=sums_dtype, device=inp.device)
        # 1D grids everywhere: on this backend a 3-tuple grid of the same
        # extent compiles cumsum_chunk_prefix_kernel into a different (broken)
        # binary that computes the row as (iter/BN)/grid, and both forms share
        # one in-process cache key, so the first launch shape wins for the
        # whole process (reproduced: (3, 16384) 3d-first err=222.38 vs 1d
        # err=0.0005).  All kernels here use program_id(0) only.
        grid = (M * C_full,)
        cumsum_chunk_scan_kernel[grid](
            inp,
            out,
            sums,
            N,
            C_full,
            S,
            ACC_DTYPE=acc_tl,
            BN=BN,
            num_warps=8,
            buffer_size_limit=2048,
        )
        if T:
            # Partial last chunk: scan on a zero-padded (M, TL) buffer so the
            # tail's load/scan/store are all full-width; the (padded) scan is
            # exact and its sum is the chunk sum.  Then add the prefix in
            # torch and slice back (the slice-copy only takes the valid T
            # columns).
            TL = max(64, triton.next_power_of_2(T))
            # The scan is performed IN PLACE on tail_buf, so the buffer must be
            # at least as wide as the accumulated values: for bool/int8/uint8
            # inputs the scan of a > 127/255-element tail reaches values past
            # the i8 range and an inp.dtype buffer truncates them (e.g. a
            # 904-element bool tail scans to 485..487, stored back as
            # 229..231).  int32 is the ACC width for these dtypes and holds
            # any T <= 4096; the copy below converts, and the final add/assign
            # widens to `out`.  Float inputs keep inp.dtype (existing
            # fp16/bf16 tail rounding semantics).
            tail_dtype = (
                torch.int32
                if inp.dtype in (torch.bool, torch.int8, torch.uint8)
                else inp.dtype
            )
            tail_buf = torch.empty(M, TL, dtype=tail_dtype, device=inp.device)
            tail_buf.fill_(0)
            tail_buf[:, :T] = inp[:, C_full * BN :]
            cumsum_chunk_tail_kernel[(M, 1, 1)](
                tail_buf,
                tail_buf,
                ACC_DTYPE=acc_tl,
                TL=TL,
                num_warps=8,
                buffer_size_limit=2048,
            )
            # Chunk sum = the scanned tail's last valid element (no reduction).
            sums[:, C - 1] = tail_buf[:, T - 1]
        # Full-width (M, S) row scan of the chunk sums, in-place and
        # unmasked (S lanes are all in-bounds by construction; the padding
        # columns are zero, so the first C lanes are the exact scan).
        if C > 4096:
            _scan_rows_into(sums, sums, M, C)
        else:
            cumsum_row_kernel[(M,)](
                sums,
                sums,
                N=S,
                TILE_N=S,
                NEED_MASK=0,
                num_warps=8 if S > 2048 else 4,
                buffer_size_limit=2048,
            )
        # Add the exclusive prefix to every full chunk.
        if C_full > 1:
            cumsum_chunk_prefix_kernel[grid](
                out,
                sums,
                N,
                C_full,
                S,
                BN=BN,
                num_warps=8,
                buffer_size_limit=2048,
            )
        if T:
            # Tail prefix = scanned sums at column C_full - 1 (exclusive of
            # the tail chunk's own sum); torch does the masked add + slice.
            pref_tail = sums[:, C_full - 1]
            out[:, C_full * BN :] = tail_buf[:, :T] + pref_tail[:, None]


# K > 1 (mid-dim scan) tier: the previous scan_then_fan / scan_part_sum_*
# kernels are unusable on this backend -- their 1024-lane tl.cumsum is
# data-dependently mis-computed and their masked tail block performs OOB reads
# into adjacent memory (polluted partial sums and even a hard fault for some
# shapes).  The only reliably-correct scan primitive on this backend is a
# SINGLE-SHOT <= 4096-lane 1D tl.cumsum over unmasked, in-bounds data (see
# cumsum_row_kernel); the N > _ROW_MAX_N row path is chunked at 4096 lanes with
# partial tails zero-padded (see cumsum_chunk_scan_kernel).  So the mid-dim
# scan is decomposed into three passes over groups of _GROUP lanes on a
# zero-padded (M*K, N) copy, with the group-prefix scan itself reusing
# _scan_rows_into (which is proven).  cumsum commutes with the
# (M, N, K) -> (M, K, N) transpose, so the result is identical to torch's.
# See harness/solution/performance/cumsum__recheck_20260906.md.
_GROUP = 4096


@libentry()
@triton.jit
def scan_group_sum_kernel(inp, sums, N: tl.constexpr, NG: tl.constexpr, GROUP: tl.constexpr):
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
def scan_group_add_kernel(inp, out, sums, N: tl.constexpr, NG: tl.constexpr, GROUP: tl.constexpr):
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
    """"(M, N, K) -> (M, K, N) -> padded group scan -> transpose back."""
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
    # The scan dim now has stride 1 (a contiguous tensor's only stride-1 dim
    # is the last; the mid-dim case above returned through the transposed
    # path), so the input is an (M, K) row-major view with M = numel // K.
    # The proven row-scan tiers (_scan_rows_into: flat multi-row sweep for
    # power-of-two K <= 512, one-row tiles for K <= 4096, chunked online scan
    # for K > 4096) are faster than the per-row fused kernel on this backend
    # (the flat sweep runs one scan per TM rows instead of one per row, and
    # the one-row tier's masked load at TILE_N = next_pow2(K) is the same
    # cost as the fused kernel's), so the scan and the normalize-by-total are
    # split; sum(x) == cumsum(x)[-1] makes the divide a single narrow.
    M = inp.numel() // K
    out = torch.empty_like(inp)
    with torch_device_fn.device(inp.device):
        _scan_rows_into(inp, out, M, K)
        return out / out.narrow(-1, K - 1, 1)
