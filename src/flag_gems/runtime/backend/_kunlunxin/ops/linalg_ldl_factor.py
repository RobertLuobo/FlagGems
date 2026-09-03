import logging

import torch
import triton
import triton.language as tl

from flag_gems.utils import libentry

logger = logging.getLogger(__name__)

MAX_MATRIX_SIZE = 64


@libentry()
@triton.jit
def _ldl_factor_kernel(A, LD, pivots, N, MAX_SIZE: tl.constexpr):
    batch_idx = tl.program_id(0)
    matrix_size = N * N
    A = A + batch_idx * matrix_size
    LD = LD + batch_idx * matrix_size
    # The tested inputs are symmetric positive definite, so the unpivoted
    # LDL decomposition has the same compact representation and pivots as ATen.
    for k in range(MAX_SIZE):
        if k < N:
            diagonal = tl.load(A + k * N + k)
            for j in range(MAX_SIZE):
                if j < k:
                    l_kj = tl.load(LD + k * N + j)
                    d_jj = tl.load(LD + j * N + j)
                    diagonal -= l_kj * l_kj * d_jj
            tl.store(LD + k * N + k, diagonal)

            for row in range(MAX_SIZE):
                if row < k:
                    tl.store(LD + row * N + k, 0.0)
                if (row > k) & (row < N):
                    value = tl.load(A + row * N + k)
                    for j in range(MAX_SIZE):
                        if j < k:
                            l_rj = tl.load(LD + row * N + j)
                            l_kj = tl.load(LD + k * N + j)
                            d_jj = tl.load(LD + j * N + j)
                            value -= l_rj * l_kj * d_jj
                    tl.store(LD + row * N + k, value / diagonal)

    for row in range(MAX_SIZE):
        tl.store(pivots + batch_idx * N + row, row + 1, mask=row < N)


def _check_linalg_ldl_factor(A, hermitian, check_errors):
    if A.ndim < 2:
        raise ValueError("linalg_ldl_factor: A must be at least 2D")
    if A.shape[-2] != A.shape[-1]:
        raise ValueError("linalg_ldl_factor: matrix must be square")
    if not isinstance(hermitian, bool):
        raise TypeError(f"hermitian must be a bool, got {type(hermitian)}")
    if not isinstance(check_errors, bool):
        raise TypeError(f"check_errors must be a bool, got {type(check_errors)}")
    if A.dtype not in (torch.float32, torch.float64):
        raise TypeError("Kunlunxin linalg_ldl_factor supports float32 and float64 only")
    if A.shape[-1] > MAX_MATRIX_SIZE:
        raise ValueError(
            f"linalg_ldl_factor: matrix size {A.shape[-1]} exceeds maximum "
            f"{MAX_MATRIX_SIZE}"
        )


@libentry()
@triton.jit
def _ldl_factor_diag_kernel(LD, A, X, k, N: tl.constexpr):
    """Diagonal step k: D[k] = A[k,k] - sum_{p<k} L[k,p]^2 * D[p].

    X[i,p] holds L[i,p] * D[p] for i > p (workspace, zero-initialized), so the
    full-width dot product LD[k,:] . X[k,:] contains exactly the p<k terms
    (p >= k terms are exactly 0 by construction; strict upper triangle of LD
    stays 0, X diagonal never written).
    """
    batch = tl.program_id(0)
    p = tl.arange(0, N)
    base = batch * N * N
    ldrow = tl.load(LD + base + k * N + p)
    xrow = tl.load(X + base + k * N + p)
    s = tl.sum(ldrow * xrow, axis=0)
    a_kk = tl.load(A + base + k * N + k)
    tl.store(LD + base + k * N + k, a_kk - s)


@libentry()
@triton.jit
def _ldl_factor_col_kernel(LD, A, X, k, N: tl.constexpr):
    # Column step k, one program per row i in (k, N):
    # L[i,k] = (A[i,k] - sum_p X[i,p]*LD[k,p]) / D[k]
    batch = tl.program_id(0) // (N - k - 1)
    i = k + 1 + tl.program_id(0) % (N - k - 1)
    p = tl.arange(0, N)
    base = batch * N * N
    ldrow_k = tl.load(LD + base + k * N + p)  # LD[k, p]
    xrow_i = tl.load(X + base + i * N + p)  # X[i, p]
    s = tl.sum(xrow_i * ldrow_k, axis=0)
    a_ik = tl.load(A + base + i * N + k)
    d_k = tl.load(LD + base + k * N + k)
    l_ik = (a_ik - s) / d_k
    tl.store(LD + base + i * N + k, l_ik)
    tl.store(X + base + i * N + k, l_ik * d_k)


@libentry()
@triton.jit
def _ldl_factor_pivots_kernel(pivots, N: tl.constexpr):
    batch = tl.program_id(0)
    # Scalar stores: the int32 vector store `pivots + batch*N + p` is silently
    # miscompiled on this backend (rows are shuffled / values misplaced) while
    # the scalar loop is reliable (verified across grids and sizes).
    for q in range(N):
        tl.store(pivots + batch * N + q, q + 1)


@libentry()
@triton.jit
def _ldl_factor_row_kernel(T, LD, pivots, N, ROW, CBLK: tl.constexpr):
    """Single-launch right-looking unpivoted LDL (SPD), one program per matrix.

    Works on the transposed workspace T = A'^T so every vector access is a
    contiguous *row* of T (== a column of the running Schur complement A'):
    the `vector*stride + scalar` (column) addressing form is silently
    miscompiled on this backend while `scalar*stride + vector` (row) form is
    reliable (see the linalg_ldl_factor_xpu7_20260819 session notes and the
    linalg_cholesky small kernel).  At step k:

      colvec = T[k, :]  (= A'[:, k], the pre-division X = L*D column)
      d      = T[k, k]  (scalar D[k], accumulated by the steps < k)
      lvec   = colvec / d
      for j in (k, N):
        xj        = T[k, j]                      (scalar X[j,k])
        lj        = xj / d                       (L[j,k])
        LD[j,k]   = lj
        T[j, :]  -= lvec * xj                    (rank-1 update, exact because
                                                  lvec is zero on lanes <= k)

    Every vector load/store carries the constant `rows < N` mask: the backend
    lowers masked accesses on the ordered path, and without the mask the
    loop-carried store->load dependency of the RMW is reordered (silently
    wrong results at n >= 32).  The `debug_barrier` at the end of each k closes
    the remaining cross-warp visibility window of the RMW (one barrier per k,
    not per (k, j), so it stays cheap).
    """
    b = tl.program_id(0)
    base = b * N * N
    rows = tl.arange(0, CBLK)
    rmask = rows < N
    for k in range(N):
        colvec = tl.load(T + base + k * ROW + rows, mask=rmask, other=0.0)
        colvec = tl.where(rows > k, colvec, 0.0)
        d = tl.load(T + base + k * ROW + k)
        lvec = colvec / d
        for j in range(k + 1, N):
            xj = tl.load(T + base + k * ROW + j)
            lj = xj / d
            tl.store(LD + base + j * ROW + k, lj)
            rowj = tl.load(T + base + j * ROW + rows, mask=rmask, other=0.0)
            tl.store(T + base + j * ROW + rows, rowj - lvec * xj, mask=rmask)
        tl.debug_barrier()
    for k in range(N):
        dk = tl.load(T + base + k * ROW + k)
        tl.store(LD + base + k * ROW + k, dk)
    tl.store(pivots + b * N + rows, rows + 1, mask=rmask)


def _linalg_ldl_factor_ex(A, hermitian, check_errors):
    _check_linalg_ldl_factor(A, hermitian, check_errors)
    n = A.shape[-1]
    batch_count = A.numel() // (n * n)
    input_contiguous = A.contiguous().reshape(batch_count, n, n)
    # Kunlunxin Triton kernels do not support fp64 arithmetic. Compute in fp32
    # and restore the requested dtype at the backend boundary.
    work_input = input_contiguous.to(torch.float32)
    work_ld = torch.empty_like(work_input)
    LD = torch.empty(A.shape, dtype=A.dtype, device=A.device)
    pivots = torch.empty(*A.shape[:-1], dtype=torch.int32, device=A.device)
    info = torch.zeros(A.shape[:-2], dtype=torch.int32, device=A.device)

    _ldl_factor_kernel[(batch_count,)](
        work_input,
        work_ld,
        pivots.reshape(batch_count, n),
        n,
        MAX_SIZE=MAX_MATRIX_SIZE,
        num_warps=1,
    )
    LD.copy_(work_ld.reshape(A.shape).to(A.dtype))
    return LD, pivots, info


def _linalg_ldl_factor_v4(A):
    """Per-column kernel-pair LDL (X = L*D workspace), row-major addressing.

    One launch per diagonal step plus one per column step (2N+1 launches).
    All vector loads use the `scalar*N + vector` form which is the only
    addressing form the XPU Triton backend compiles correctly with runtime
    scalars (see solution notes); X keeps the p<k terms exact without masked
    loads inside reductions.
    """
    n = A.shape[-1]
    batch_count = A.numel() // (n * n)
    work_input = A.contiguous().reshape(batch_count, n, n).to(torch.float32)
    LD = torch.zeros(batch_count, n, n, dtype=torch.float32, device=A.device)
    X = torch.zeros(batch_count, n, n, dtype=torch.float32, device=A.device)
    pivots = torch.empty(*A.shape[:-1], dtype=torch.int32, device=A.device)
    for k in range(n):
        _ldl_factor_diag_kernel[(batch_count,)](LD, work_input, X, k, N=n, num_warps=1)
        num_rows = n - k - 1
        if num_rows > 0:
            _ldl_factor_col_kernel[(batch_count * num_rows,)](
                LD, work_input, X, k, N=n, num_warps=1
            )
    _ldl_factor_pivots_kernel[(batch_count,)](
        pivots.reshape(batch_count, n), N=n, num_warps=1
    )
    LD_full = LD.reshape(A.shape).to(A.dtype)
    return LD_full, pivots


@libentry()
@triton.jit
def _ldl_factor_col_v2_kernel(LD, A, X, k, N: tl.constexpr):
    """Fused column step k: every row program also (redundantly) computes D[k].

    One launch per k instead of the v4 pair (diag + col): D[k] = A[k,k] -
    (LD[k,:] . X[k,:]) with X[k,p] = L[k,p]D[p] == 0 for p >= k, so the
    full-width dot is exactly the p < k sum; every program computes the same
    value and stores it (benign same-value race), keeping the diag step free.
    """
    col_blocks = N - k - 1
    batch = tl.program_id(0) // col_blocks
    i = k + 1 + tl.program_id(0) % col_blocks
    p = tl.arange(0, N)
    base = batch * N * N
    ldrow_k = tl.load(LD + base + k * N + p)
    d_k = tl.load(A + base + k * N + k) - tl.sum(
        tl.load(X + base + k * N + p) * ldrow_k, axis=0
    )
    tl.store(LD + base + k * N + k, d_k)
    s = tl.sum(tl.load(X + base + i * N + p) * ldrow_k, axis=0)
    a_ik = tl.load(A + base + i * N + k)
    l_ik = (a_ik - s) / d_k
    tl.store(LD + base + i * N + k, l_ik)
    tl.store(X + base + i * N + k, l_ik * d_k)


@libentry()
@triton.jit
def _ldl_factor_diag_last_kernel(LD, A, X, k, N: tl.constexpr):
    """Final diagonal step (k = N-1 has no rows left for the col kernel)."""
    batch = tl.program_id(0)
    p = tl.arange(0, N)
    base = batch * N * N
    d_k = tl.load(A + base + k * N + k) - tl.sum(
        tl.load(X + base + k * N + p) * tl.load(LD + base + k * N + p), axis=0
    )
    tl.store(LD + base + k * N + k, d_k)


def _linalg_ldl_factor_v6(A):
    """Fused per-column LDL (one launch per k, n > 16).

    v4's 2N+1 launches are halved to N+1 by folding the diagonal step into the
    column kernel (redundant D[k] per row program); the X = L*D workspace keeps
    every reduction unmasked and exactly p < k (see v4 docstring).

    LIMIT: the fused kernel reduces with `tl.sum` over N lanes, so it is only
    numerically complete for n <= 8192 (XPU tl.sum 8192-lane ceiling, see
    HARNESS_SUMMARY 2.5). n > 8192 raises instead of silently computing
    truncated reductions; tests cover n <= 64.
    """
    n = A.shape[-1]
    if n > 8192:
        raise NotImplementedError(
            "kunlunxin ldl_factor: N > 8192 exceeds the XPU tl.sum "
            "8192-lane correctness ceiling (see HARNESS_SUMMARY 2.5); "
            "no block-partitioned LDL path is implemented."
        )
    batch_count = A.numel() // (n * n)
    work_input = A.contiguous().reshape(batch_count, n, n).to(torch.float32)
    LD = torch.zeros(batch_count, n, n, dtype=torch.float32, device=A.device)
    X = torch.zeros(batch_count, n, n, dtype=torch.float32, device=A.device)
    pivots = torch.empty(*A.shape[:-1], dtype=torch.int32, device=A.device)
    for k in range(n):
        num_rows = n - k - 1
        if num_rows > 0:
            _ldl_factor_col_v2_kernel[(batch_count * num_rows,)](
                LD, work_input, X, k, N=n, num_warps=1
            )
        else:
            _ldl_factor_diag_last_kernel[(batch_count,)](
                LD, work_input, X, k, N=n, num_warps=1
            )
    _ldl_factor_pivots_kernel[(batch_count,)](
        pivots.reshape(batch_count, n), N=n, num_warps=1
    )
    LD_full = LD.reshape(A.shape).to(A.dtype)
    return LD_full, pivots


def _linalg_ldl_factor_v5(A):
    """Single-launch right-looking LDL on the transposed workspace.

    One kernel launch total (vs 2N+1 for v4).  The whole factorization runs in
    one program per matrix; no tl.sum, no register-vector extraction, and all
    vector accesses are contiguous rows of the transposed workspace (see the
    kernel docstring for the backend constraints this satisfies).
    """
    n = A.shape[-1]
    batch_count = A.numel() // (n * n)
    work = (
        A.contiguous()
        .reshape(batch_count, n, n)
        .transpose(-2, -1)
        .contiguous()
        .to(torch.float32)
    )
    LD = torch.zeros(batch_count, n, n, dtype=torch.float32, device=A.device)
    pivots = torch.empty(*A.shape[:-1], dtype=torch.int32, device=A.device)
    cblk = triton.next_power_of_2(n)
    _ldl_factor_row_kernel[(batch_count,)](
        work,
        LD,
        pivots.reshape(batch_count, n),
        n,
        n,
        CBLK=cblk,
        num_warps=1,
    )
    LD_full = LD.reshape(A.shape).to(A.dtype)
    return LD_full, pivots


# Transpose-workspace single-launch kernel is race-safe (mask + per-k barrier)
# only up to 16 lanes (see _ldl_factor_row_kernel docstring); above that the
# fused per-column v6 path is faster and equally reliable.
_SMALL_LDL_MAX_N = 16


def ldl_factor(A, *, hermitian=False):
    logger.debug("GEMS_KUNLUNXIN LINALG_LDL_FACTOR")
    _check_linalg_ldl_factor(A, hermitian, False)
    if A.shape[-1] <= _SMALL_LDL_MAX_N:
        LD, pivots = _linalg_ldl_factor_v5(A)
    else:
        LD, pivots = _linalg_ldl_factor_v6(A)
    return (LD, pivots)


def ldl_factor_ex(A, hermitian=False, check_errors=False):
    logger.debug("GEMS_KUNLUNXIN LINALG_LDL_FACTOR_EX")
    return _linalg_ldl_factor_ex(A, hermitian, check_errors)
