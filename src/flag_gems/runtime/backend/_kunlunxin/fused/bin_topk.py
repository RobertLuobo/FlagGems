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

"""
Kunlunxin (XPU) specialized implementation of ``flag_gems.fused.DSA.bin_topk.bucket_sort_topk``.

Why a specialized file (XPU, measured 2026-09-04):
- The general implementation in ``flag_gems/fused/DSA/bin_topk.py`` cannot run
  on XPU at all: its Triton legacy kernel uses ``tl.histogram`` which is not
  lowered by the Triton-XPU backend (``ConvertTritonXPUToLLVM`` fails:
  "failed to legalize operation 'tt.reduce'"), so the kernel never compiles
  (``OutOfResources ... uni_sram``); the TLE kernel is disabled because the
  kunlunxin VendorDescriptor sets ``tle_enabled=False``.

XPU backend constraints discovered while writing this file (all verified
with minimal kernels):
- ``tl.histogram`` is unsupported; ``tl.sum`` / ``tl.max`` / ``tl.argmax`` /
  ``tl.cumsum`` only legalize at ONE level of loop nesting (a reduce inside
  two nested ``scf.for`` loops is "explicitly marked illegal").
- A scatter store whose address tensor is data-dependent and whose value
  tensor is an affine ``arange`` mis-pairs lanes (the value is stored from the
  wrong lane; cluster-layout mismatch). Stores work when the value and the
  address come from the same data-dependent chain (e.g. ``out[pos] = pos``)
  or when the address is affine.
- Masked stores whose mask is derived from value comparisons (``gt | eq``)
  misapply the mask; affine masks (``offs < n``) are fine. Consequently the
  fill kernel uses *dynamic loop bounds* (``range(0, c1)`` /
  ``range(0, min(c2, K - c1))``) with unconditional affine stores instead of
  value-derived store masks.
- The ``@triton.autotune``-style multi-config path and ``tl.static_range``
  unrolls hang / OOM the XPU compiler.

Algorithm (numeric-safe, no ``tl.histogram``, no data-dependent scatter):
per row: (1) binary search (32 host-driven iterations, one count kernel per
bit) for the K-th largest order key; (2) rank kernel: per-lane 1-based rank of
the strict-greater keys (+r) and of the equal keys (-r), stored via affine
addresses; (3) fill kernel: for each output slot p, ``out[p] = sum((rank ==
p+1) * lane)`` (affine store, pure ``tl.sum`` reductions). Rows longer than
the 8192-lane block are processed chunk-wise (top-K per 8192-chunk) and the
candidates are merged by re-applying the same pipeline recursively until the
candidate count fits one block (candidate values are gathered with a simple
indexed load; the final mapping from candidate positions back to global
indices is an affine-store gather).
"""

import sys

import torch
import triton
import triton.language as tl

BS = 8192   # count/rank tile = rank buffer width (sum/cumsum safe block size)
RANK_SUB = 512   # sub-tile width of the cumulative scan inside _bst_rank_kernel
RANK_NSUB = BS // RANK_SUB   # number of sub-tiles per row buffer (16)


@triton.jit
def _ord_i32(x):
    bits = x.to(tl.int32, bitcast=True)
    # UNSIGNED-ascending int32 key: u(x) < u(y) (unsigned)  <=>  x < y
    return tl.where(x >= 0, bits | (-0x80000000), ~bits)


@triton.jit
def _uge(a, b):
    return (a ^ (-0x80000000)) >= (b ^ (-0x80000000))


@triton.jit
def _ugt(a, b):
    return (a ^ (-0x80000000)) > (b ^ (-0x80000000))


@triton.jit
def _bst_count_kernel(inputs, starts, ends, cands, cnts, S: tl.constexpr, BS: tl.constexpr):
    b = tl.program_id(0)
    s_base = inputs + b * S
    start = tl.load(starts + b).to(tl.int32)
    end = tl.load(ends + b).to(tl.int32)
    n = end - start
    TS = tl.cdiv(n, BS)
    cand = tl.load(cands + b)
    cnt = 0
    for t in range(TS):
        offs = t * BS + tl.arange(0, BS)
        m = offs < n
        x = tl.load(s_base + start + offs, mask=m, other=0.0).to(tl.float32)
        u = _ord_i32(x)
        cnt += tl.sum((m & _uge(u, cand)).to(tl.int32), axis=0)
    tl.store(cnts + b, cnt)


@triton.jit
def _bst_rank_kernel(
    inputs, ranks, starts, ends, thrs,
    S: tl.constexpr, RB: tl.constexpr, SUB: tl.constexpr, NSUB: tl.constexpr,
):
    # per-lane signed rank: +r for u > thr (r = 1..c1), -r for u == thr (r = 1..c2), 0 otherwise
    # The cumulative scan is done in SUB-lane sub-tiles (SUB=512) with a running
    # global total: an 8192-lane single tl.cumsum silently mis-computes on the
    # XPU/flagtree backend (only ~512 lanes end up ranked; measured 2026-09-04 as
    # test_bucket_sort_topk_large_scale at ~0.25-0.26 intersection).
    b = tl.program_id(0)
    s_base = inputs + b * S
    rank_base = ranks + b * RB
    start = tl.load(starts + b).to(tl.int32)
    end = tl.load(ends + b).to(tl.int32)
    n = end - start
    TS = tl.cdiv(n, SUB)
    thr = tl.load(thrs + b)
    prev_gt = 0
    prev_eq = 0
    for t in range(TS):
        offs = t * SUB + tl.arange(0, SUB)
        m = offs < n
        x = tl.load(s_base + start + offs, mask=m, other=0.0).to(tl.float32)
        u = _ord_i32(x)
        gt = _ugt(u, thr) & m
        eq = (u == thr) & m
        cums_gt = tl.cumsum(gt.to(tl.int32), axis=0)
        cums_eq = tl.cumsum(eq.to(tl.int32), axis=0)
        # signed rank: +r (gt) / -r (eq) / 0 (neither), via pure arithmetic
        # (a nested tl.where as the stored value mis-pairs lanes on XPU)
        val = (cums_gt + prev_gt) * gt.to(tl.int32) - (cums_eq + prev_eq) * eq.to(tl.int32)
        tl.store(rank_base + offs, val, mask=m)
        prev_gt += tl.sum(gt.to(tl.int32), axis=0)
        prev_eq += tl.sum(eq.to(tl.int32), axis=0)


@triton.jit
def _bst_fill_kernel(
    ranks, out, scratch, n_arr, starts, gmap,
    K: tl.constexpr, BSF: tl.constexpr, RB: tl.constexpr, HAS_GMAP: tl.constexpr,
):
    b = tl.program_id(0)
    rank_base = ranks + b * RB
    out_base = out + b * K
    scr = scratch + b * (K + 4)
    n = tl.load(n_arr + b)
    st = tl.load(starts + b).to(tl.int32)
    offs = tl.arange(0, BSF)
    m = offs < n
    rk = tl.load(rank_base + offs, mask=m, other=0)
    if HAS_GMAP:
        lane = tl.load(gmap + st + offs, mask=m, other=-1)
    else:
        lane = offs + st
    c1 = tl.sum((rk > 0).to(tl.int32), axis=0)
    c2 = tl.sum((rk < 0).to(tl.int32), axis=0)
    # NOTE(2026-09-04, R2): the loops are bounded by the *runtime* counts
    # c1 / min(c2, K - c1) and the stores are unconditional. The previous
    # implementation kept the loop up to K and masked the stores with
    # value-derived predicates (mask=c1 > p / (q < c2) & (c1 + q < K)); on the
    # XPU/flagtree backend such data-dependent store masks are silently
    # dropped, so for rows shorter than K (c1 + c2 < K, e.g. the tail chunk of
    # a large-scale variable-length row) the leftover slots [c1+c2, K) were
    # written with v = sum((rk == p+1)*lane) = 0, i.e. a spurious duplicate of
    # position 0. Those 0-entries (value == x[0], often a large value) entered
    # the recursive candidate pool and evicted genuine top-K positions
    # (observed intersection 0.49-0.78 instead of 1.0; e.g. n = 51911,
    # K = 4096 -> 1337 spurious -> 2759/4096). c1 < K always holds (binary
    # search invariant count(u > thr) < K), so min(c2, K - c1) >= 0 and the
    # equal loop lowers to exactly the slots [c1, min(c1+c2, K)).
    for p in range(0, c1):
        v = tl.sum((rk == p + 1).to(tl.int32) * lane, axis=0)
        tl.store(scr + p, v)
        tl.store(out_base + p, v)
    for q in range(0, tl.minimum(c2, K - c1)):
        v = tl.sum((rk == -(q + 1)).to(tl.int32) * lane, axis=0)
        tl.store(scr + K + q, v)
        tl.store(out_base + c1 + q, v)


@triton.jit
def _bst_gather_vals_kernel(x, cidx, cval, K: tl.constexpr):
    # cidx: (NCH*K,) global idx (0..S-1, -1 = invalid); cval: (NCH*K,) f32
    j = tl.program_id(0)
    off = j * K + tl.arange(0, K)
    idx = tl.load(cidx + off)
    m = idx >= 0
    idx_c = tl.where(m, idx, 0)
    v = tl.load(x + idx_c).to(tl.float32)
    v = tl.where(m, v, float("-inf"))
    tl.store(cval + off, v)


@triton.jit
def _bst_map_idx_kernel(gmap, idx, out, N: tl.constexpr):
    # out[j] = gmap[idx[j]]  (idx = positions, -1 -> -1)
    j = tl.program_id(0)
    off = j * 1024 + tl.arange(0, 1024)
    m = off < N
    i = tl.load(idx + off, mask=m, other=0)
    im = m & (i >= 0)
    ic = tl.where(im, i, 0)
    v = tl.load(gmap + ic)
    tl.store(out + off, tl.where(im, v, -1), mask=m)


def _bst_select(x, starts, ends, n, k_eff, K, out, gmap=None):
    """3-kernel pipeline. x: (B, S). starts/ends/n/k_eff: (B,). out: (B, K)."""
    Bb = x.shape[0]
    thrs = torch.zeros(Bb, dtype=torch.int32, device=x.device)
    cnts = torch.empty(Bb, dtype=torch.int32, device=x.device)
    one = torch.tensor(1, dtype=torch.int32, device=x.device)
    for bit in range(31, -1, -1):
        cands = thrs | (one << bit)
        _bst_count_kernel[(Bb,)](x, starts, ends, cands, cnts, x.shape[1], BS, num_warps=4, num_stages=1)
        thrs = torch.where(cnts >= k_eff, cands, thrs)
    ranks = torch.zeros(Bb, BS, dtype=torch.int32, device=x.device)
    _bst_rank_kernel[(Bb,)](x, ranks, starts, ends, thrs, x.shape[1], BS, RANK_SUB, RANK_NSUB, num_warps=4, num_stages=1)
    scratch = torch.zeros(Bb, 2 * K + 8, dtype=torch.int32, device=x.device)
    if gmap is None:
        gm = torch.zeros(1, dtype=torch.int32, device=x.device)
        _bst_fill_kernel[(Bb,)](ranks, out, scratch, n, starts, gm, K, BS, BS, False,
                                num_warps=4, num_stages=1)
    else:
        _bst_fill_kernel[(Bb,)](ranks, out, scratch, n, starts, gmap, K, BS, BS, True,
                                num_warps=4, num_stages=1)


def _bst_rows(xv, st_val, en_val, n_val, K, out, gmap=None):
    """Recursive chunked select on a single row. xv: (1, S) values;
       st_val/en_val: (1,) int32 row [start, end); gmap: (S,) global index map
       or None; out: (1, K) global indices."""
    nn = int(n_val[0].item())
    if nn <= 0:
        return
    S = xv.shape[1]
    st0 = int(st_val[0].item())
    starts = torch.tensor([st0], dtype=torch.int32, device=xv.device)
    if nn <= BS:
        n1 = torch.tensor([nn], dtype=torch.int32, device=xv.device)
        k1 = torch.tensor([min(K, nn)], dtype=torch.int32, device=xv.device)
        _bst_select(xv, starts, en_val, n1, k1, K, out, gmap)
        return
    nch = (nn + BS - 1) // BS
    cidx = torch.full((nch, K), -1, dtype=torch.int32, device=xv.device)
    for j in range(nch):
        cst = st0 + j * BS
        cn = min(BS, nn - j * BS)
        if cn <= 0:
            continue
        st_c = torch.tensor([cst], dtype=torch.int32, device=xv.device)
        en_c = torch.tensor([cst + cn], dtype=torch.int32, device=xv.device)
        n_c = torch.tensor([cn], dtype=torch.int32, device=xv.device)
        k_c = torch.tensor([min(K, cn)], dtype=torch.int32, device=xv.device)
        # chunk candidates are POSITIONS in xv (no gmap); the composition happens
        # via _bst_map_idx_kernel below.
        _bst_select(xv, st_c, en_c, n_c, k_c, K, cidx[j:j + 1])
    cflat = cidx.reshape(-1)
    M = nch * K
    cvals = torch.full((M,), float("-inf"), device=xv.device)
    _bst_gather_vals_kernel[(nch,)](xv[0], cflat, cvals, K, num_warps=4, num_stages=1)
    if gmap is None:
        gm = cflat
    else:
        gm = torch.full((M,), -1, dtype=torch.int32, device=xv.device)
        _bst_map_idx_kernel[((M + 1023) // 1024,)](gmap, cflat, gm, M, num_warps=4, num_stages=1)
    z1 = torch.tensor([0], dtype=torch.int32, device=xv.device)
    n1 = torch.tensor([M], dtype=torch.int32, device=xv.device)
    _bst_rows(cvals.view(1, M), z1, n1, n1, K, out, gm)


def bucket_sort_topk_xpu(inputs, starts, ends, topk):
    x = inputs.float() if inputs.dtype != torch.float32 else inputs
    B, S = x.shape
    K = topk
    out = torch.full((B, K), -1, dtype=torch.int32, device=x.device)
    if B == 0 or S == 0:
        return out
    n = (ends - starts).to(torch.int32)
    with torch.no_grad():
        for b in range(B):
            _bst_rows(x[b:b + 1], starts[b:b + 1], ends[b:b + 1], n[b:b + 1], K, out[b:b + 1])
    return out


# ---------------------------------------------------------------------------
# wiring: patch the direct-import entrypoint (mhc_pre-style self-install)
# ---------------------------------------------------------------------------
def _install():
    """Replace ``flag_gems.fused.DSA.bin_topk.bucket_sort_topk`` with the XPU
    implementation. The DSA bin_topk family is called via direct module
    import (``from flag_gems.fused.DSA.bin_topk import bucket_sort_topk``) in
    tests/test_DSA/test_bin_topk.py, so the SpecOpRegistrar namespace swap
    cannot reach it; the attribute of the already-imported module (loaded
    during ``import flag_gems``) is patched here instead."""
    gmod = sys.modules.get("flag_gems.fused.DSA.bin_topk")
    if gmod is not None:
        gmod.bucket_sort_topk = bucket_sort_topk_xpu


_install()
