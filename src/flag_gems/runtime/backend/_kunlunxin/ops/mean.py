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

import builtins
import logging

import torch
import triton
import triton.language as tl

# from flag_gems import runtime
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import dim_compress, libentry
from flag_gems.utils import triton_lang_extension as ext

from ..utils.block_size_utils import get_block_size_1d

logger = logging.getLogger(__name__)


@libentry()
@triton.jit
def mean_scalar_kernel(inp, out, M, BLOCK_SIZE: tl.constexpr):
    """Scalar mean over all M elements.
    On XPU (USE_XHPC): intercepted by baidu::xpu::api::mean binding.
    Triton fallback (single CTA): sequential accumulation for correctness.
    Params for binding:
      kernelParams[0] = inp, kernelParams[1] = out
      kernelConsts[2] = M,   kernelConsts[3] = BLOCK_SIZE
    """
    acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for off in range(0, M, BLOCK_SIZE):
        offset = off + tl.arange(0, BLOCK_SIZE)
        mask = offset < M
        v = tl.load(inp + offset, mask=mask, other=0.0).to(tl.float32)
        acc += v
    result = tl.sum(acc) / M
    tl.store(out, result)


def mean(inp, *, dtype=None):
    logger.debug("GEMS_KUNLUNXIN MEAN")
    M = inp.numel()
    if dtype is None:
        dtype = inp.dtype
    if M == 0:
        # torch.mean of an empty tensor is 0/0 = NaN; the flat kernel below
        # cannot handle a zero trip count.
        return torch.full([], float("nan"), dtype=dtype, device=inp.device)
    BLOCK_SIZE = get_block_size_1d(M, inp.element_size())
    out = torch.empty([], dtype=dtype, device=inp.device)

    with torch_device_fn.device(inp.device):
        mean_scalar_kernel[(1, 1, 1)](inp, out, M, BLOCK_SIZE, buffer_size_limit=2048)
    return out

_TILE_BUDGET = 32768
_N_WIDE = 8192

# Mid-dim (K>1) reduction tunables. The reduction cuts the strided middle
# axis of [M, N, K], so three strategies are used (2-D tiles and in-loop
# tl.sum / tl.reshape are rejected or miscompiled by this backend):
#  - small K (<= _MID_CHUNK_K_MAX) with large N (> _MID_CHUNK_N_MIN): the
#    two-kernel chunked path (JCHUNK-wide strided gathers are dense when
#    consecutive j are only a few elements apart).
#  - float32 with 32 < K <= 1024: the wide path (mean_dim_mid_wide_*), one
#    program per (m, chunk) reducing a [T, K_PAD] tile with one fp32
#    ``dot(ones[1, T], B)`` (the mma pipeline loads the B-tile through SRAM,
#    ~970 GB/s vs ~31 GB/s for the old CHUNK-unrolled 64-lane adds).  Only
#    *affine* (completely unmasked) T x K_PAD loads keep that rate - any
#    mask/min in the load makes the backend emit narrow loads AND the loaded
#    tile is then miscompiled into garbage (dot kernels: even an all-true
#    mask miscompiles).  The OOB risk (k-pad lanes of the very last element
#    when K_PAD > K) is avoided by leaving row N-1 to a small clamped tail
#    kernel.  fp16/bf16 wide loads are ~17 GB/s (no better than the online
#    kernel), so those keep the online path.
#  - otherwise (or non-float32): the serial online kernel below with 128-wide
#    block-DMA loads (in-loop lane widths above 128 overflow uni_sram at
#    TritonXPUUnrollControl on this backend; fp16/bf16 accumulate in fp32
#    in-kernel via cdtype).
_MID_ONLINE_TILE_K = 128
_MID_CHUNK_JCHUNK = 4096
_MID_CHUNK_K_MAX = 32
_MID_CHUNK_N_MIN = 512
# Wide path tunables.  The partial sums are computed with a single
# [1, T] @ [T, K_PAD] fp32 tl.dot per program: the mma pipeline loads the
# B-tile through SRAM (measured ~970 GB/s at [2048, 64] / [256, 1024] vs
# ~31 GB/s for the old CHUNK-unrolled 64-lane adds, whose load width was
# the dominant bandwidth factor on this backend).  The B-tile is capped at
# _MID_WIDE_TILE_BYTES (2^20): [512, 1024] = 2MB fails triton-legalize
# (uni_sram out of resources) while [2048, 64] = 512KB and [256, 1024] =
# 1MB compile, so the SRAM budget is ~1MB per B-tile.
_MID_WIDE_TILE_BYTES = 1 << 20
_MID_WIDE_BN_MAX = 2048
# The clamped 1-D tail kernel unrolls its rows; the unroll count (CHUNK) is
# the stack limiter on this backend: CHUNK <= 64 compiles for every K_PAD
# (64 x 512 = 32K lanes OK, 32 x 1024 = 32K lanes OK) while CHUNK = 128
# fails even at 128 x 64 = 8K lanes ("Failed to tune buffer size"), i.e. the
# envelope is NOT lane-based.  K_PAD x CHUNK stays within the lane guard too
# (for K_PAD = 1024 that caps CHUNK at 32).  The affine tail group size is
# chosen so the remaining rows (the clamped tail) are at most
# _MID_WIDE_TAIL_REM.
_MID_WIDE_TILE = 32768
_MID_WIDE_TAIL_REM = 64
_MID_WIDE_COMB_MAX_UNROLL = 64


def _block_n(N):
    if N > _N_WIDE:
        return builtins.min(triton.next_power_of_2(N), 2048)  # wide for large N
    return builtins.min(triton.next_power_of_2(N), 512)  # tall-friendly otherwise


def heur_n_block_size(args):
    return _block_n(args["N"])


def heur_m_block_size(args):
    block_n = _block_n(args["N"])
    block_m = triton.next_power_of_2(triton.cdiv(args["M"], 12))  # cluster_num
    return builtins.min(block_m, builtins.max(_TILE_BUDGET // block_n, 1))


@libentry()
# @triton.autotune(
#     configs=runtime.get_tuned_config("mean"),
#     key=["M", "N"],
# )
@triton.heuristics(
    values={
        "BLOCK_M": heur_m_block_size,
        "BLOCK_N": heur_n_block_size,
    },
)
@triton.jit
def mean_dim_kernel(X, Mean, M, N, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    """2-D reduction: reduce N-dim for each of M rows.
    On XPU (USE_XHPC): intercepted by baidu::xpu::api::mean_dim binding.
    Params for binding:
      kernelParams[0] = X,    kernelParams[1] = Mean
      kernelParams[2] = M,    kernelParams[3] = N  (runtime scalars)
      kernelConsts[4] = BLOCK_M (constexpr), kernelConsts[5] = BLOCK_N (constexpr)
    """
    # Map the program id to the row of X it should compute.
    pid = ext.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]
    X = X + pid * N
    Mean = Mean + pid
    row_mask = pid < M

    _mean = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
    for off in range(0, N, BLOCK_N):
        cols = off + tl.arange(0, BLOCK_N)[None, :]
        col_mask = cols < N
        mask = row_mask and col_mask

        a = tl.load(X + cols, mask, other=0.0).to(tl.float32)
        _mean += a
    mean = tl.sum(_mean, axis=1)[:, None] / N
    tl.store(Mean, mean, row_mask)


@libentry()
@triton.jit
def mean_dim_mid_kernel(X, Out, M, N, K, BLOCK_K: tl.constexpr):
    """Mid-dim (K>1) row-reduce WITHOUT the dim_compress transpose copy.

    X is the original [M, N, K] (M = outer product, N = reduction length,
    K = inner product) contiguous layout; each program handles one m-row and
    one BLOCK_K-slice of K. Per XPU constraints (HARNESS_SUMMARY 2.5/3.6):
    fully UNMASKED loads with the K index clamped in-bounds (no OOB read, no
    garbage), fp32 accumulation (cdtype: fp16/bf16 inputs accumulate in
    fp32), NO tl.sum / where-in-reduce inside the loop (pure elementwise
    add), the clamped tail lanes are zeroed by an arithmetic multiply (not
    tl.where inside a reduce), and stores are masked.
    BLOCK_K must be <= 128 (larger in-loop lane widths overflow uni_sram
    at TritonXPUUnrollControl) or the next power of two of a small K.
    """
    if tl.constexpr(X.dtype.element_ty == tl.float16) or tl.constexpr(
        X.dtype.element_ty == tl.bfloat16
    ):
        cdtype = tl.float32
    else:
        cdtype = X.dtype.element_ty

    pid_m = ext.program_id(0)
    pid_k = ext.program_id(1)

    k_off = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    k_clamped = tl.minimum(k_off, K - 1)
    k_mask = k_off < K

    p = X + pid_m * N * K + k_clamped
    acc = tl.zeros([BLOCK_K], dtype=cdtype)
    for _ in range(0, N):
        v = tl.load(p).to(cdtype)
        acc += v
        p += K
    mean = (acc / N) * k_mask.to(cdtype)
    tl.store(Out + pid_m * K + k_off, mean, mask=k_mask)


@libentry()
@triton.jit
def mean_dim_mid_partial_kernel(X, Sum, M, N, K, stride, JCHUNK: tl.constexpr):
    """Per-chunk partial sums of the middle dim of a [M, N, K] input.

    One program per (chunk, m, k) gathers JCHUNK lanes of one (m, k) column
    (element (m, j0+l, k)) and stores the chunk sum into Sum[c * stride + mk],
    stride = M * K. Used when K is small (consecutive j lanes are a few
    elements apart, so the gather is dense). The strided load must be
    UNMASKED (masked strided 1-D gathers return garbage on this backend):
    out-of-range lanes are clamped to N-1 (reading a duplicated in-bounds
    element) and then zeroed in registers with ``tl.where(j < N, ...)``. The
    only tl.sum is at top level.
    """
    pid = ext.program_id(0)
    chunk = pid // (M * K)
    mk = pid % (M * K)
    m = mk // K
    k = mk % K
    j = chunk * JCHUNK + tl.arange(0, JCHUNK)
    jc = tl.minimum(j, N - 1)
    a = tl.load(X + m * N * K + jc * K + k).to(tl.float32)
    a = tl.where(j < N, a, 0.0)
    s = tl.sum(a, axis=0)
    tl.store(Sum + chunk * stride + mk, s)


@libentry()
@triton.jit
def mean_dim_mid_combine_kernel(Sum, Out, nchunks, stride, N, TILE_C: tl.constexpr):
    """Combine the nchunks per-chunk partial sums of one (m, k): out = s / N.

    Strided loads use the same clamp + register-where pattern as the partial
    kernel; invalid slots read the in-bounds element nchunks-1 and are
    replaced with 0.0 in registers.
    """
    pid = ext.program_id(0)  # m * K + k
    c_offsets = tl.arange(0, TILE_C)
    c_mask = c_offsets < nchunks
    c = tl.minimum(c_offsets, nchunks - 1)
    sc = tl.load(Sum + c * stride + pid)
    sc = tl.where(c_mask, sc, 0.0)
    s = tl.sum(sc, axis=0) / N
    tl.store(Out + pid, s)


@libentry()
@triton.jit
def mean_dim_mid_wide_partial_kernel(
    X, Sum, M, N, K, nchunks, n_groups, n_base, slot_base, K_PAD: tl.constexpr,
    T: tl.constexpr,
):
    """Affine per-chunk partial sums of the middle dim, via one mma dot.

    One program per (m, g) loads the T x K_PAD tile of rows
    [n_base + g*T, n_base + (g+1)*T) (completely unmasked affine 2-D load:
    any min/where/mask in the load address makes the backend emit narrow
    loads, and -- worse for the mma -- the loaded tile is then miscompiled
    into garbage, so the launcher guarantees every covered row's K_PAD-wide
    read stays in-bounds) and reduces it with one fp32
    ``dot(ones[1, T], B)`` (the mma pipeline loads B via SRAM, ~970 GB/s).
    The [1, K_PAD] result is stored at slot ``slot_base + g``.

    K_PAD may exceed K: lanes [K, K_PAD) of a row read the FOLLOWING row's
    elements (in-bounds because the launcher leaves the last row to the
    clamped tail kernel when K_PAD > K).  Those lanes' sums are garbage but
    are discarded by the combine kernel's ``offs_k < K`` store mask; lanes
    [0, K) are exact.
    """
    pid = ext.program_id(0)
    m = pid // n_groups
    g = pid % n_groups
    offs_n = n_base + g * T + tl.arange(0, T)
    offs_k = tl.arange(0, K_PAD)
    base = m * (N * K) + offs_n[:, None] * K + offs_k[None, :]
    b = tl.load(X + base).to(tl.float32)
    a = tl.full((1, T), 1.0, dtype=tl.float32)
    acc = tl.dot(a, b, allow_tf32=False)
    s = tl.reshape(acc, (K_PAD,))
    tl.store(Sum + (m * nchunks + slot_base + g) * K_PAD + offs_k, s)


@libentry()
@triton.jit
def mean_dim_mid_wide_single_kernel(X, Out, M, N, K, K_PAD: tl.constexpr, T: tl.constexpr):
    """Single-chunk wide path: N == T covers the whole reduction in one
    (m, g)-program, so the mma-dot partial is already the final sum and is
    stored directly to Out (scaled by 1/N) -- no partial buffer, no combine
    launches (each combine launch costs ~0.4 ms on this backend at large M
    because its per-row narrow loads are latency-bound).  Used only when the
    K_PAD-wide row read of the last row stays in-bounds (K_PAD == K), so the
    store needs no k-mask.
    """
    m = ext.program_id(0)
    offs_n = tl.arange(0, T)
    offs_k = tl.arange(0, K_PAD)
    base = m * (N * K) + offs_n[:, None] * K + offs_k[None, :]
    b = tl.load(X + base).to(tl.float32)
    a = tl.full((1, T), 1.0, dtype=tl.float32)
    acc = tl.dot(a, b, allow_tf32=False)
    s = tl.reshape(acc, (K_PAD,))
    tl.store(Out + m * K + offs_k, s * (1.0 / N))


@libentry()
@triton.jit
def mean_dim_mid_wide_tail_kernel(
    X, Sum, M, N, K, nchunks, nA, K_PAD: tl.constexpr, CHUNK: tl.constexpr
):
    """Clamped rows [nA, nA + CHUNK) of the middle dim (the 1-D fallback).

    Handles the rows that cannot be dot-reduced with an unmasked affine load
    (the remainder of the affine tail groups, and always the last row when
    K_PAD > K, whose pad lanes would read OOB).  Same unrolled structure as
    the historical partial kernel, but the row and K-lane indices are
    clamped in-bounds (``n_c``, ``k_c``) so the loads stay unmasked (1-D);
    out-of-range iterations are zeroed in registers with ``tl.where`` (never
    inside the load address).  CHUNK is ``next_power_of_2(n - nA)`` capped so
    CHUNK * K_PAD stays in the stable envelope, and the tail always fills the
    LAST chunk slot (``nchunks - 1``) of each m-row.
    """
    m = ext.program_id(0)
    offs_k = tl.arange(0, K_PAD)
    k_c = tl.minimum(offs_k, K - 1)
    n0 = nA
    base = m * (N * K) + n0 * K
    s = tl.zeros([K_PAD], dtype=tl.float32)
    for n in tl.static_range(CHUNK):
        n_glob = n0 + n
        n_c = tl.minimum(n_glob, N - 1)
        a = tl.load(X + base + (n_c - n0) * K + k_c).to(tl.float32)
        s += tl.where(n_glob < N, a, 0.0)
    tl.store(Sum + (m * nchunks + (nchunks - 1)) * K_PAD + offs_k, s)


@libentry()
@triton.jit
def mean_dim_mid_wide_combine_groups_kernel(
    SumIn, SumOut, M, K, nchunks, nch_groups, K_PAD: tl.constexpr, UNROLL: tl.constexpr
):
    """Level-1 combine: group the nchunks chunk slots of one m-row into
    nch_groups = ceil(nchunks / UNROLL) group sums.

    One program per (m, g) unrolls UNROLL slots (UNROLL <= 32; larger static
    unrolls or dynamic c-loops fail TritonXPUUnrollControl or exceed
    uni_sram on this backend at some widths).  Slots >= nchunks are loaded
    from the clamped in-bounds slot ``nchunks - 1`` (same row region) and
    zeroed in registers with ``tl.where``, so the loads stay un-masked.
    """
    pid = ext.program_id(0)
    m = pid // nch_groups
    g = pid % nch_groups
    offs_k = tl.arange(0, K_PAD)
    base = m * (nchunks * K_PAD)
    s = tl.zeros([K_PAD], dtype=tl.float32)
    for i in tl.static_range(UNROLL):
        c = g * UNROLL + i
        c_cl = tl.minimum(c, nchunks - 1)
        v = tl.load(SumIn + base + c_cl * K_PAD + offs_k)
        s += tl.where(c < nchunks, v, 0.0)
    tl.store(SumOut + (m * nch_groups + g) * K_PAD + offs_k, s)


@libentry()
@triton.jit
def mean_dim_mid_wide_combine_final_kernel(
    SumIn, Out, M, N, K, nch_groups, K_PAD: tl.constexpr, UNROLL: tl.constexpr
):
    """Level-2 (final) combine: sum the nch_groups group slots of one m-row
    and scale: out = s / N.

    nch_groups (or nchunks, when it fits in one level) is always <= 32, so
    the slot loop is fully unrolled.  Lanes [K, K_PAD) of the slots are
    garbage (clamped reads) and are masked off by ``k_ok``.
    """
    m = ext.program_id(0)
    offs_k = tl.arange(0, K_PAD)
    k_ok = offs_k < K
    s = tl.zeros([K_PAD], dtype=tl.float32)
    base = m * (nch_groups * K_PAD)
    for g in tl.static_range(UNROLL):
        s += tl.load(SumIn + base + g * K_PAD + offs_k)
    tl.store(Out + m * K + offs_k, s * (1.0 / N), mask=k_ok)


def _mean_dim_mid_wide(x, M, N, K, dtype, out_shape):
    """Wide-path middle-dim mean: dot-based partial sums, then a tiny combine.

    f32 only (f16/bf16 wide loads are ~17 GB/s on this backend, no better
    than the 128-wide online kernel).  K_PAD = next_power_of_2(K) with
    64 <= K_PAD <= 1024 (K_PAD=32 and >= 2048 are miscompiled / hang this
    backend).  Rows are reduced in T-row groups by
    ``mean_dim_mid_wide_partial_kernel`` (T = _MID_WIDE_BN_MAX for the body,
    chosen so T * K_PAD * 4 <= _MID_WIDE_TILE_BYTES); the tail rows
    [nA, N) -- and, when K_PAD > K, always the last row whose pad lanes
    would read OOB -- go to the clamped 1-D tail kernel.  The combine never
    uses a dynamic c-loop (broken at 256/512-lane widths): slots are reduced
    in groups of <= 32 via a two-level unrolled scheme.
    """
    K_PAD = triton.next_power_of_2(K)
    BN = builtins.min(_MID_WIDE_BN_MAX, _MID_WIDE_TILE_BYTES // (K_PAD * 4))
    # Start of the tail chunk: every full chunk that cannot read past the end
    # of X.  When K_PAD > K a K_PAD-wide row load of the very last row would
    # be OOB, so the last row always goes to the tail kernel.
    nA = (N // BN) * BN if K_PAD == K else ((N - 1) // BN) * BN
    nchA = nA // BN
    n_tail = N - nA
    # Affine (dot) tail rows: everything except the rows handed to the
    # clamped 1-D tail.  When K_PAD > K the last row must stay clamped (its
    # pad lanes are OOB); otherwise the whole tail can be affine.
    n_a = n_tail if K_PAD == K else n_tail - 1
    # Largest power-of-two group size (T <= BN) whose clamped remainder fits
    # the 1-D tail envelope (n_rem <= _MID_WIDE_TAIL_REM and
    # n_rem * K_PAD <= _MID_WIDE_TILE).  T = 32 always fits (n_rem < 32 +
    # T/2), so the loop is guaranteed to terminate.
    T_TAIL = BN
    n_tail_groups = 0
    while T_TAIL >= 32:
        n_g = n_a // T_TAIL
        if n_g >= 1:
            n_rem = n_tail - n_g * T_TAIL
            if n_rem <= _MID_WIDE_TAIL_REM and n_rem * K_PAD <= _MID_WIDE_TILE:
                n_tail_groups = n_g
                break
        T_TAIL >>= 1
    n_rem = n_tail - n_tail_groups * T_TAIL
    nchunks = nchA + n_tail_groups + (1 if n_rem > 0 else 0)
    out = torch.empty(out_shape, dtype=dtype, device=x.device)
    if nchunks == 1 and nchA == 1:
        # One chunk covered entirely by the main kernel (N == BN, K_PAD == K):
        # the mma-dot partial is already the final sum, so store it directly
        # (the two combine launches cost ~0.4 ms on this backend at large M).
        with torch_device_fn.device(x.device):
            mean_dim_mid_wide_single_kernel[(M,)](
                x, out, M, N, K, K_PAD=K_PAD, T=BN
            )
        return out
    spart = torch.empty(M * nchunks * K_PAD, dtype=torch.float32, device=x.device)
    with torch_device_fn.device(x.device):
        if nchA > 0:
            mean_dim_mid_wide_partial_kernel[(M * nchA,)](
                x,
                spart,
                M,
                N,
                K,
                nchunks,
                nchA,
                0,
                0,
                K_PAD=K_PAD,
                T=BN,
            )
        if n_tail_groups > 0:
            mean_dim_mid_wide_partial_kernel[(M * n_tail_groups,)](
                x,
                spart,
                M,
                N,
                K,
                nchunks,
                n_tail_groups,
                nA,
                nchA,
                K_PAD=K_PAD,
                T=T_TAIL,
            )
        if n_rem > 0:
            mean_dim_mid_wide_tail_kernel[(M,)](
                x,
                spart,
                M,
                N,
                K,
                nchunks,
                nA + n_tail_groups * T_TAIL,
                K_PAD=K_PAD,
                CHUNK=triton.next_power_of_2(n_rem),
                buffer_size_limit=2048,
            )
        nch_groups = (nchunks + _MID_WIDE_COMB_MAX_UNROLL - 1) // _MID_WIDE_COMB_MAX_UNROLL
        if nch_groups > 1:
            sp2 = torch.empty(
                M * nch_groups * K_PAD, dtype=torch.float32, device=x.device
            )
            mean_dim_mid_wide_combine_groups_kernel[(M * nch_groups,)](
                spart,
                sp2,
                M,
                K,
                nchunks,
                nch_groups,
                K_PAD=K_PAD,
                UNROLL=_MID_WIDE_COMB_MAX_UNROLL,
                buffer_size_limit=2048,
            )
            mean_dim_mid_wide_combine_final_kernel[(M,)](
                sp2,
                out,
                M,
                N,
                K,
                nch_groups,
                K_PAD=K_PAD,
                UNROLL=nch_groups,
                buffer_size_limit=2048,
            )
        else:
            mean_dim_mid_wide_combine_final_kernel[(M,)](
                spart,
                out,
                M,
                N,
                K,
                nchunks,
                K_PAD=K_PAD,
                UNROLL=nchunks,
                buffer_size_limit=2048,
            )
    return out


def _mean_dim_mid_chunked(x, M, N, K, dtype, out_shape):
    """Chunk-split middle-dim mean: one pass over x, then a tiny combine."""
    nchunks = triton.cdiv(N, _MID_CHUNK_JCHUNK)
    TILE_C = max(1, triton.next_power_of_2(nchunks))
    stride = M * K
    spart = torch.empty((nchunks * M * K,), dtype=torch.float32, device=x.device)
    out = torch.empty(out_shape, dtype=dtype, device=x.device)
    with torch_device_fn.device(x.device):
        mean_dim_mid_partial_kernel[(nchunks * M * K,)](
            x, spart, M, N, K, stride, JCHUNK=_MID_CHUNK_JCHUNK, buffer_size_limit=2048
        )
        mean_dim_mid_combine_kernel[(M * K,)](
            spart, out, nchunks, stride, N, TILE_C=TILE_C, buffer_size_limit=2048
        )
    return out


def mean_dim(x, dim, keepdim=False, *, dtype=None):
    logger.debug("GEMS_KUNLUNXIN MEAN_DIM")

    if dtype is None:
        dtype = x.dtype
    if dim is None or dim == () or dim == []:
        out = mean(x, dtype=dtype)
        if keepdim:
            out = out.reshape([1] * x.ndim)
        return out

    shape = list(x.shape)
    dim = [d % x.ndim for d in dim]

    if len(dim) == 1:
        dim0 = dim[0]
        N = shape[dim0]
        M = 1
        for i in shape[:dim0]:
            M *= i
        K = (x.numel() // (M * N)) if (M * N) else 0
        if K > 1 and M > 0 and N > 0:
            x = x.contiguous()
            out_shape = shape[:dim0] + [1] + shape[dim0 + 1 :]
            if N == 1:
                # N=1: mean over a size-1 dim is the identity (same as the
                # historic N==1 fast path; no dim_compress here, so this is a
                # zero-copy view / dtype cast only).
                out = x.to(dtype=dtype).reshape(out_shape)
                if not keepdim:
                    out = out.squeeze(dim=dim0)
                return out
            if x.dtype == torch.float32 and K > 32 and K <= 1024:
                # f32: the wide path, one affine [1, T] @ [T, K_PAD] dot per
                # group (mma pipeline, ~970 GB/s at T=2048/K_PAD=64) + a
                # tiny combine.  Only the value-verified envelope is routed
                # here: K <= 32 (K_PAD=32 is silently miscompiled) and
                # K > 1024 (K_PAD>=2048 tail hang) stay on the other paths.
                out = _mean_dim_mid_wide(x, M, N, K, dtype, out_shape)
            elif K <= _MID_CHUNK_K_MAX and N > _MID_CHUNK_N_MIN:
                # Small K + large N: the JCHUNK-wide strided gather of the
                # chunked path is dense (consecutive j lanes are K elements
                # apart), so one pass reads x once (replaces the old
                # torch.bmm(ones, x) fast path with a pure-Triton reduction).
                out = _mean_dim_mid_chunked(x, M, N, K, dtype, out_shape)
            else:
                # Serial online kernel with 128-wide block-DMA loads; the
                # kernel accumulates in fp32 (cdtype) for every input dtype
                # (f16/bf16 wide loads are ~17 GB/s here, no better than
                # this path, so they stay online).
                out = torch.empty(out_shape, dtype=dtype, device=x.device)
                BLOCK_K = (
                    _MID_ONLINE_TILE_K if K >= _MID_ONLINE_TILE_K else triton.next_power_of_2(K)
                )
                grid = (M, triton.cdiv(K, BLOCK_K))
                with torch_device_fn.device(x.device):
                    mean_dim_mid_kernel[grid](
                        x, out, M, N, K, BLOCK_K=BLOCK_K, buffer_size_limit=2048
                    )
            if not keepdim:
                out = out.squeeze(dim=dim0)
            return out
    # ------------------------------------------------------------------------

    x = dim_compress(x, dim)
    N = 1
    for i in dim:
        N *= shape[i]
        shape[i] = 1
    M = x.numel() // N if N > 0 else 0

    # Reducing over an empty (size-0) dimension means 0/0 = NaN for every
    # output element, matching torch's reference behavior.
    if N == 0:
        out = torch.full(shape, float("nan"), dtype=dtype, device=x.device)
        if not keepdim:
            out = out.squeeze(dim)
        return out

    # No output rows at all: the result is empty, no computation needed.
    if M == 0:
        out = torch.empty(shape, dtype=dtype, device=x.device)
        if not keepdim:
            out = out.squeeze(dim)
        return out

    # Edge case: M=1 means all dims are reduced → global mean over N elements.
    # mean_dim XPU API does not support M=1.
    if M == 1:
        scalar_out = mean(x, dtype=dtype)  # 0-d tensor
        out = scalar_out.reshape(shape)
        if not keepdim:
            out = out.squeeze(dim)
        return out

    # Edge case: N=1 means reducing a trivial (size-1) dimension.
    # mean of 1 element = that element; just copy with dtype conversion.
    # mean_dim XPU API does not support N=1.
    if N == 1:
        out = x.to(dtype=dtype).reshape(shape)
        if not keepdim:
            out = out.squeeze(dim)
        return out

    out = torch.empty(shape, dtype=dtype, device=x.device)
    grid = lambda META: (triton.cdiv(M, META["BLOCK_M"]),)

    with torch_device_fn.device(x.device):
        mean_dim_kernel[grid](x, out, M, N, buffer_size_limit=2048)
    if not keepdim:
        out = out.squeeze(dim)
    return out
