import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as tle

logger = logging.getLogger(__name__)

# Row pitch of the working buffer.  The elimination kernel reads whole
# "row-broadcast" vectors (W + row*LDA + col with a [ROWS, LDA]-shaped col
# index), so the pitch must be at least 64 (the smallest vector this backend
# reads/stores contiguously; see linalg_det for the same constraint).
_MIN_LDA = 64
# Upper bound on the leading dimension of the square matrices handled by one
# program.  With the (ROWS, LDA) padded layout the largest program tile is
# next_pow2(n) * 64 lanes: n = 32 -> 2048 lanes, well inside the 4096-lane
# single-program limit that was validated on this backend.
_MAX_MATRIX_SIZE = 32
# 1-D vectors only: TritonXPU refuses to lower a kernel that mixes a [TM]
# vector with a [TM, LDA] tile, and refuses a 2-D reduction inside a runtime
# loop.  Every non-scalar value below therefore has shape [TOT].


@triton.jit
def _reduce_mul(a, b):
    return a * b


@libentry()
@triton.jit
def _slogdet_pack_kernel(SRC, DST, N, LDA: tl.constexpr, TOT: tl.constexpr):
    """Zero-padded pack of a contiguous (batch, N, N) buffer into
    (batch, ROWS, LDA); one program per matrix.

    ``other=`` silently pollutes live lanes and a masked store is not honoured
    at all on this backend, so padding lanes are explicitly zeroed instead.
    """
    b = tle.program_id(0).to(tl.int64)
    e = tl.arange(0, TOT)
    row = e // LDA
    col = e % LDA
    live = (row < N) & (col < N)
    idx = tl.where(live, row * N + col, 0)
    val = tl.load(SRC + b * N * N + idx)
    tl.store(DST + b * TOT + e, tl.where(live, val, 0.0))


@libentry()
@triton.jit
def _slogdet_step_kernel(W, DG, N, K, LDA: tl.constexpr, TOT: tl.constexpr):
    """One complete partial-pivot elimination step, one matrix per program.

    The whole matrix lives in one flat [TOT] vector; K comes from the host so
    there is no runtime loop around the global store/load round trip (an
    in-kernel loop over K corrupts data on this backend, see linalg_det).

    ``tl.argmax`` is unreliable on this backend, so the pivot row is the
    smallest row attaining a plain 1-D tl.max (LAPACK first-strict-maximum
    order), with padded lanes excluded via the row < N condition.  The signed
    pivot (already accounting for the swap parity) is stored to DG[K]; an
    exactly zero pivot (singular column) is stored as +0.0 and detected by
    the post kernel.
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
def _slogdet_post_kernel(DG, SIGN, LOGABS, N, LDA: tl.constexpr):
    """Fuse sign / logabsdet from the signed pivots of one matrix.

    sign = prod(sign(DG[k])) and logabsdet = sum(log|DG[k]|) over k < N; an
    exactly zero (or NaN) pivot means the matrix is singular: sign = 0 and
    logabsdet = -inf, matching torch.linalg.slogdet.
    """
    b = tle.program_id(0).to(tl.int64)
    cols = tl.arange(0, LDA)
    v = tl.load(DG + b * LDA + cols)
    live = cols < N
    v = tl.where(live, v, 0.0)
    neg = tl.sum(tl.where(live & (v < 0.0), 1.0, 0.0), axis=0)
    sg = tl.where(neg % 2.0 == 1.0, -1.0, 1.0)
    la = tl.sum(tl.where(live & (v != 0.0), tl.log(tl.abs(v)), 0.0), axis=0)
    singular = tl.sum(tl.where(live & ((v == 0.0) | (v != v)), 1.0, 0.0), axis=0)
    tl.store(SIGN + b, tl.where(singular > 0.0, 0.0, sg))
    tl.store(LOGABS + b, tl.where(singular > 0.0, float("-inf"), la))


def linalg_slogdet(A):
    logger.debug("GEMS_KUNLUNXIN LINALG_SLOGDET")
    if A.dtype != torch.float32:
        raise NotImplementedError(f"linalg_slogdet: unsupported dtype {A.dtype}")
    if A.dim() < 2 or A.shape[-1] != A.shape[-2]:
        raise RuntimeError("linalg_slogdet: expected batches of square matrices")

    n = A.shape[-1]
    if n == 0 or n > _MAX_MATRIX_SIZE:
        raise NotImplementedError(
            f"linalg_slogdet: matrix size {n} out of supported range "
            f"(1..{_MAX_MATRIX_SIZE})"
        )

    batch_shape = A.shape[:-2]
    batch_size = 1
    for dimension in batch_shape:
        batch_size *= dimension

    sign = torch.empty(batch_shape, dtype=A.dtype, device=A.device)
    logabsdet = torch.empty(batch_shape, dtype=A.dtype, device=A.device)
    if batch_size == 0:
        return torch.zeros_like(sign), torch.full_like(logabsdet, float("-inf"))

    if not A.is_contiguous():
        A = A.contiguous()
    A3 = A.reshape(batch_size, n, n)

    # Padded (batch, ROWS, LDA) working buffer; LDA >= 64 keeps the
    # row-broadcast loads inside the row they address.
    rows = triton.next_power_of_2(n)
    lda = max(_MIN_LDA, rows)
    tot = rows * lda
    work = torch.empty(batch_size * tot, dtype=A.dtype, device=A.device)
    dg = torch.empty(batch_size * lda, dtype=A.dtype, device=A.device)
    sign_flat = sign.reshape(-1)
    logabs_flat = logabsdet.reshape(-1)

    with torch_device_fn.device(A.device):
        _slogdet_pack_kernel[(batch_size,)](
            A3, work, n, LDA=lda, TOT=tot, num_warps=1
        )
        for k in range(n):
            _slogdet_step_kernel[(batch_size,)](
                work, dg, n, k, LDA=lda, TOT=tot, num_warps=1
            )
        _slogdet_post_kernel[(batch_size,)](
            dg, sign_flat, logabs_flat, n, LDA=lda, num_warps=1
        )
    return sign, logabsdet