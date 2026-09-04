import logging
import math

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as tle

logger = logging.getLogger(__name__)

# Row pitch of the working buffer.  Vector stores on this backend always cover
# 64 contiguous elements and ignore their mask, so the row-swap kernel -- the
# only kernel that writes a single row -- needs a pitch of at least 64 to keep
# its stores inside the row they address.
_MIN_LDA = 64
# Upper bound on the flat tile of one program.  4096 lanes were validated; a
# 16384-lane tile of the same shape produced an illegal memory access.
_MAX_BLK = 4096


@libentry()
@triton.jit
def _det4_kernel(A, OUT, TOT: tl.constexpr):
    """Closed-form 4x4 determinant.

    The cofactor expansion is evaluated directly in registers: the matrix is
    fetched with 16 scalar loads (one per entry; TOT == 16) and the formula
    only does multiplies/subtracts.  There is no working-buffer store/load
    round trip, so this path is immune to the backend's unsafe in-kernel
    store->load reordering (the reason every other kernel is launched once per
    step).
    """
    b = tle.program_id(0).to(tl.int64)
    base = b * TOT
    a00 = tl.load(A + base + 0)
    a01 = tl.load(A + base + 1)
    a02 = tl.load(A + base + 2)
    a03 = tl.load(A + base + 3)
    a10 = tl.load(A + base + 4)
    a11 = tl.load(A + base + 5)
    a12 = tl.load(A + base + 6)
    a13 = tl.load(A + base + 7)
    a20 = tl.load(A + base + 8)
    a21 = tl.load(A + base + 9)
    a22 = tl.load(A + base + 10)
    a23 = tl.load(A + base + 11)
    a30 = tl.load(A + base + 12)
    a31 = tl.load(A + base + 13)
    a32 = tl.load(A + base + 14)
    a33 = tl.load(A + base + 15)
    det = (
        a00
        * (
            a11 * (a22 * a33 - a23 * a32)
            - a12 * (a21 * a33 - a23 * a31)
            + a13 * (a21 * a32 - a22 * a31)
        )
        - a01
        * (
            a10 * (a22 * a33 - a23 * a32)
            - a12 * (a20 * a33 - a23 * a30)
            + a13 * (a20 * a32 - a22 * a30)
        )
        + a02
        * (
            a10 * (a21 * a33 - a23 * a31)
            - a11 * (a20 * a33 - a23 * a30)
            + a13 * (a20 * a31 - a21 * a30)
        )
        - a03
        * (
            a10 * (a21 * a32 - a22 * a31)
            - a11 * (a20 * a32 - a22 * a30)
            + a12 * (a20 * a31 - a21 * a30)
        )
    )
    tl.store(OUT + b, det)


@triton.jit
def _reduce_mul(a, b):
    return a * b


def _plan(n):
    rows = triton.next_power_of_2(n)
    # For n in {16, 32} the n x n matrix is itself a power-of-two tile
    # (triton arange size) and a multiple of the 64-element store granule, so
    # the padded rows*64 layout is unnecessary: the matrix is used in place
    # (no pack kernel) and every step touches n^2 lanes instead of n*64.
    if n in (16, 32):
        lda = n
    elif n == 8:
        # Measured optimum for n=8: the 256-lane (ROWS=8, LDA=32) tile is ~2.5%
        # faster than the 512-lane (LDA=64) one at batch 1024 (sweep: 600.7us
        # vs 615.7us) while staying well above the 64-lane store granule.
        lda = 32
    else:
        lda = max(_MIN_LDA, rows)
    tot = rows * lda
    blk = min(_MAX_BLK, tot)
    return rows, lda, tot, blk, tot // blk


@libentry()
@triton.jit
def _det_pack_kernel(
    SRC, DST, N, LDA: tl.constexpr, BLK: tl.constexpr, TOT: tl.constexpr
):
    """Scatter a contiguous (batch, N, N) buffer into (batch, ROWS, LDA).

    Padding lanes are zeroed rather than masked away: ``other=`` silently
    pollutes live lanes here, and a masked store is not honoured at all.
    """
    b = tle.program_id(0).to(tl.int64)
    blk = tle.program_id(1).to(tl.int64)
    e = blk * BLK + tl.arange(0, BLK)
    row = e // LDA
    col = e % LDA
    live = (row < N) & (col < N)
    idx = tl.where(live, row * N + col, 0)
    val = tl.load(SRC + b * N * N + idx)
    tl.store(DST + b * TOT + e, tl.where(live, val, 0.0))


@libentry()
@triton.jit
def _det_step0_kernel(SRC, W, DG, N, LDA: tl.constexpr, TOT: tl.constexpr):
    """Elimination step K=0 with the pack fused in.

    Only valid when W is a separate (padded) buffer: every read comes from SRC
    (contiguous N*N) and W is written only, so this one launch replaces the
    pack kernel plus step 0.  The sum-based ``akk``/``apk`` extraction is kept
    (a scalar load of the runtime ``prow`` address races against the stores of
    the previous launch on this backend).
    """
    b = tle.program_id(0).to(tl.int64)
    base = b * TOT
    sbase = b * N * N
    e = tl.arange(0, TOT)
    row = e // LDA
    col = e % LDA
    live = (row < N) & (col < N)
    idx = tl.where(live, row * N + col, 0)
    w = tl.load(SRC + sbase + idx)
    cand = tl.where((col == 0) & (row < N), tl.abs(w), -1.0)
    best = tl.max(cand, axis=0)
    prow = tl.min(tl.where(cand == best, row, TOT), axis=0)
    akk = tl.sum(tl.where((row == 0) & (col == 0), w, 0.0), axis=0)
    apk = tl.sum(tl.where((row == prow) & (col == 0), w, 0.0), axis=0)
    cidx = tl.where(col < N, col, 0)
    ridx = tl.where(row < N, row, 0)
    row_k = tl.load(SRC + sbase + cidx)
    row_p = tl.load(SRC + sbase + prow * N + cidx)
    col_k = tl.load(SRC + sbase + ridx * N)
    swapped = tl.where(row == 0, row_p, tl.where(row == prow, row_k, w))
    lcol = tl.where(row == 0, apk, tl.where(row == prow, akk, col_k))
    safe = tl.where(apk == 0.0, 1.0, apk)
    mult = tl.where((row > 0) & (row < N), lcol / safe, 0.0)
    urow = tl.where(col > 0, row_p, 0.0)
    tl.store(W + base + e, swapped - mult * urow)
    tl.store(DG + b * LDA, tl.where(prow != 0, -apk, apk))


@libentry()
@triton.jit
def _det_step_kernel(W, DG, N, K, LDA: tl.constexpr, TOT: tl.constexpr):
    """One complete elimination step for a matrix that fits in a single program.

    Pivot search, row swap and the trailing rank-1 update are fused.  Because
    exactly one program owns the whole matrix there is no cross-program race,
    and because ``K`` comes from the host there is no runtime loop around the
    global store/load round trip (an in-kernel loop over K silently corrupted
    ~4% of the matrices from 48 matrices upward even with debug barriers).

    Every non-scalar value has the identical shape [TOT]: TritonXPU refuses to
    lower a kernel that mixes a [TM] vector with a [TM, LDA] tile, and refuses
    a 2-D reduction inside a runtime loop.
    """
    b = tle.program_id(0).to(tl.int64)
    base = b * TOT
    e = tl.arange(0, TOT)
    row = e // LDA
    col = e % LDA
    w = tl.load(W + base + e)
    cand = tl.where((col == K) & (row >= K) & (row < N), tl.abs(w), -1.0)
    best = tl.max(cand, axis=0)
    prow = tl.min(tl.where(cand == best, row, TOT), axis=0)
    akk = tl.sum(tl.where((row == K) & (col == K), w, 0.0), axis=0)
    apk = tl.sum(tl.where((row == prow) & (col == K), w, 0.0), axis=0)
    row_k = tl.load(W + base + K * LDA + col)
    row_p = tl.load(W + base + prow * LDA + col)
    col_k = tl.load(W + base + row * LDA + K)
    swapped = tl.where(row == K, row_p, tl.where(row == prow, row_k, w))
    lcol = tl.where(row == K, apk, tl.where(row == prow, akk, col_k))
    safe = tl.where(apk == 0.0, 1.0, apk)
    mult = tl.where(row > K, lcol / safe, 0.0)
    urow = tl.where(col > K, row_p, 0.0)
    tl.store(W + base + e, swapped - mult * urow)
    tl.store(DG + b * LDA + K, tl.where(prow != K, -apk, apk))


@libentry()
@triton.jit
def _det_pivot_swap_kernel(W, DG, N, K, LDA: tl.constexpr, ROWS: tl.constexpr):
    """Pivot search plus physical row swap, one matrix per program.

    ``tl.argmax`` is unreliable on this backend, so the pivot row is the
    smallest row attaining a plain 1-D ``tl.max`` (LAPACK first-strict-maximum
    order).  The signed pivot goes straight into DG[K] so the determinant is a
    plain product over K and no separate parity buffer is needed.
    """
    b = tle.program_id(0).to(tl.int64)
    base = b * ROWS * LDA
    rows = tl.arange(0, ROWS)
    cols = tl.arange(0, LDA)
    live = rows < N
    vals = tl.load(W + base + tl.where(live, rows, 0) * LDA + K)
    cand = tl.where(live & (rows >= K), tl.abs(vals), -1.0)
    best = tl.max(cand, axis=0)
    prow = tl.min(tl.where(cand == best, rows, ROWS), axis=0)
    prow = tl.where(prow >= N, K, prow)
    row_k = tl.load(W + base + K * LDA + cols)
    row_p = tl.load(W + base + prow * LDA + cols)
    tl.store(W + base + K * LDA + cols, row_p)
    tl.store(W + base + prow * LDA + cols, row_k)
    pivot = tl.sum(tl.where(cols == K, row_p, 0.0), axis=0)
    tl.store(DG + b * LDA + K, tl.where(prow != K, -pivot, pivot))


@libentry()
@triton.jit
def _det_update_kernel(
    W, N, K, LDA: tl.constexpr, BLK: tl.constexpr, TOT: tl.constexpr
):
    """Trailing rank-1 update of one flat, contiguous BLK-lane chunk.

    Used for matrices too large for ``_det_step_kernel``; the swap has already
    been applied by ``_det_pivot_swap_kernel`` so chunks never read a row that
    another program is rewriting.
    """
    b = tle.program_id(0).to(tl.int64)
    blk = tle.program_id(1).to(tl.int64)
    base = b * TOT
    e = blk * BLK + tl.arange(0, BLK)
    row = e // LDA
    col = e % LDA
    pivot_row = W + base + K * LDA
    pivot = tl.load(pivot_row + K)
    safe = tl.where(pivot == 0.0, 1.0, pivot)
    urow = tl.where(col > K, tl.load(pivot_row + col), 0.0)
    lcol = tl.load(W + base + row * LDA + K)
    mult = tl.where((row > K) & (row < N), lcol / safe, 0.0)
    tile = tl.load(W + base + e)
    tl.store(W + base + e, tile - mult * urow)


@libentry()
@triton.jit
def _det_reduce_kernel(DG, OUT, N, LDA: tl.constexpr):
    b = tle.program_id(0).to(tl.int64)
    cols = tl.arange(0, LDA)
    v = tl.load(DG + b * LDA + cols)
    det = tl.reduce(tl.where(cols < N, v, 1.0), 0, combine_fn=_reduce_mul)
    tl.store(OUT + b, det)


_DET_GRAPH_LIMIT = 32
# One CUDA graph per (n, batch, dtype, device).  The n+2-launch sequence has a
# fixed ~10us per-launch device-time floor that dominates every small-batch
# shape; a captured graph cuts that to a single replay (batch=1 (16,16) went
# from 438us eager to 184us).  Buffers are allocated once outside the capture
# (no graph memory pool is needed) and replayed on the current stream.
_det_graph_cache = {}
_det_graph_order = []


class _DetGraphEntry:
    __slots__ = (
        "batch_count",
        "n",
        "rows",
        "lda",
        "tot",
        "blk",
        "nblk",
        "padded",
        "src",
        "work",
        "dg",
        "out",
        "graph",
    )

    def __init__(self, batch_count, n, dtype, device):
        rows, lda, tot, blk, nblk = _plan(n)
        self.batch_count = batch_count
        self.n = n
        self.rows = rows
        self.lda = lda
        self.tot = tot
        self.blk = blk
        self.nblk = nblk
        self.padded = not (rows == n and lda == n)
        self.src = torch.empty(batch_count * n * n, dtype=dtype, device=device)
        if n == 4:
            # The 4x4 path never touches a working buffer.
            self.work = None
        elif self.padded:
            self.work = torch.empty(batch_count * tot, dtype=dtype, device=device)
        else:
            # rows == n and lda == n: the LU runs in place on the input copy.
            self.work = self.src
        self.dg = torch.zeros(batch_count * lda, dtype=dtype, device=device)
        self.out = torch.empty(batch_count, dtype=dtype, device=device)
        self.graph = None

    def launch(self, src):
        b = self.batch_count
        n = self.n
        if n == 4:
            with torch_device_fn.device(src.device):
                _det4_kernel[(b,)](src, self.out, 16, num_warps=1)
            return
        work = self.work
        dg = self.dg
        with torch_device_fn.device(src.device):
            if self.nblk > 1:
                if self.padded:
                    _det_pack_kernel[(b, self.nblk)](
                        src, work, n, LDA=self.lda, BLK=self.blk, TOT=self.tot,
                        num_warps=1,
                    )
                for k in range(n):
                    _det_pivot_swap_kernel[(b,)](
                        work, dg, n, k, LDA=self.lda, ROWS=self.rows,
                        num_warps=1,
                    )
                    if k + 1 < n:
                        _det_update_kernel[(b, self.nblk)](
                            work, n, k, LDA=self.lda, BLK=self.blk, TOT=self.tot,
                            num_warps=1,
                        )
            else:
                # Step 0 with the pack fused in: SRC is read only (a separate
                # padded W is written), so this is safe where the in-place
                # layout would hit the backend's store->load reordering.
                if self.padded:
                    _det_step0_kernel[(b,)](
                        src, work, dg, n, LDA=self.lda, TOT=self.tot, num_warps=1
                    )
                    start = 1
                else:
                    start = 0
                for k in range(start, n):
                    _det_step_kernel[(b,)](
                        work, dg, n, k, LDA=self.lda, TOT=self.tot, num_warps=1
                    )
            _det_reduce_kernel[(b,)](dg, self.out, n, LDA=self.lda, num_warps=1)

    def __call__(self, src):
        """Stage the input, then replay (or warm up + capture) the sequence."""
        src_flat = src.reshape(-1)
        if self.graph is None:
            # Warm up the kernels, then capture.  On this backend the capture
            # context executes the launches eagerly *and* records them, so the
            # in-place working buffer is left decomposed when capture returns;
            # staging the input again right before the first replay restores it.
            self.src.copy_(src_flat)
            with torch_device_fn.device(src.device):
                self.launch(self.src)
            torch.cuda.synchronize()
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                self.launch(self.src)
            self.graph = g
        # Replay re-runs the recorded sequence on the staged input.
        self.src.copy_(src_flat)
        self.graph.replay()
        return self.out


def _linalg_det_impl(A, out=None):
    if A.dtype not in (torch.float32, torch.float64):
        raise ValueError(f"linalg_det only supports float32 and float64, got {A.dtype}")

    if A.dim() < 2:
        raise ValueError(
            f"linalg_det: input tensor must be at least 2D, got {A.dim()}D"
        )

    m, n = A.shape[-2], A.shape[-1]
    if m != n:
        raise ValueError(
            f"linalg_det: input tensor must be a square matrix, got {m}x{n}"
        )

    batch_shape = A.shape[:-2]
    if n == 0:
        result = torch.ones(batch_shape, dtype=A.dtype, device=A.device)
        return result if out is None else out.copy_(result)

    batch_count = math.prod(batch_shape)
    if batch_count == 0:
        if out is not None:
            return out
        return torch.empty(batch_shape, dtype=A.dtype, device=A.device)

    key = (n, batch_count, A.dtype, A.device)
    entry = _det_graph_cache.get(key)
    if entry is None:
        entry = _DetGraphEntry(batch_count, n, A.dtype, A.device)
        _det_graph_cache[key] = entry
        _det_graph_order.append(key)
        while len(_det_graph_order) > _DET_GRAPH_LIMIT:
            _det_graph_cache.pop(_det_graph_order.pop(0), None)

    # A is never written: the entry copies it into its own src buffer first.
    flat = entry(A.reshape(batch_count, n, n))
    if out is None:
        return flat.reshape(batch_shape)
    if flat.data_ptr() != out.data_ptr():
        out.copy_(flat.reshape(batch_shape))
    return out


def linalg_det(A):
    logger.debug("GEMS_KUNLUNXIN LINALG_DET")
    return _linalg_det_impl(A)


def linalg_det_out(A, *, out=None):
    logger.debug("GEMS_KUNLUNXIN LINALG_DET_OUT")
    if out is None:
        raise TypeError("linalg_det(): out must be provided for out variant")
    if out.dtype != A.dtype:
        raise RuntimeError(
            f"linalg_det: dtype of out ({out.dtype}) does not match "
            f"dtype of input ({A.dtype})"
        )
    if out.device != A.device:
        raise RuntimeError(
            f"linalg_det: device of out ({out.device}) does not match "
            f"device of input ({A.device})"
        )
    if out.shape != A.shape[:-2]:
        raise RuntimeError(
            f"linalg_det: shape of out {tuple(out.shape)} does not match "
            f"expected shape {tuple(A.shape[:-2])}"
        )
    return _linalg_det_impl(A, out=out)
