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
"""Kunlunxin/XPU implementation of ``torch.linalg.qr`` (``aten::linalg_qr``).

Why this vendor override exists
-------------------------------
The generic ``flag_gems.ops.linalg_qr`` routes every shape through one of
several heavily fused Triton programs (``_qr_fused_kernel`` /
``_geqrt_sram_kernel`` / ``_tsqr_*_kernel``).  On the P800 TritonXPU backend
the very first launch already dies at compile time: ``_qr_fused_kernel``
(generic ``linalg_qr.py:905``) raises

    error: Failures have been detected while processing an MLIR pass pipeline
    note: Pipeline failed while executing [`TritonXPUCoreTiling` ...]

so ``(1, 1)`` -- the smallest input -- never even runs.  These kernels keep
large 2-D register tiles and drive dynamic in-kernel loops that store the Q/R
tile to global and read it back on the next iteration; that structure is the
same class of failure the sibling linalg overrides (``linalg_solve_ex``,
``linalg_matrix_power``) had to sidestep on this backend.

Strategy: vendor decomposition routing (no ATen / native / composite compute
fallback).  We build the classic unblocked Householder QR out of primitives
that are already correct on XPU:

* the Householder reflectors + the upper-triangular ``R`` (i.e. a ``geqrf``)
  are produced by a host-driven column sweep whose only matrix product is the
  trailing-panel update, computed with the vendor ``bmm`` XPU Triton kernel;
  everything else is device elementwise / reduction work;
* ``Q`` is assembled from the packed reflectors by the already-registered
  vendor ``linalg_householder_product`` (orgqr) XPU kernel.

Householder (not Cholesky-QR / Gram-Schmidt) is deliberate: it stays robust
for exactly rank-deficient / zero-column inputs (a zero sub-column simply
yields ``tau = 0`` -> ``H = I``), which the test-suite exercises directly.

Only ``float32`` / ``float64`` are supported, matching the generic op (fp64 is
silently down-cast to fp32 on this backend, so fp64 reference cases are
skipped by the harness).
"""

import logging
from collections import namedtuple

import torch

from .bmm import bmm
from .linalg_householder_product import linalg_householder_product

logger = logging.getLogger(__name__)

LinalgQrResult = namedtuple("LinalgQrResult", ["Q", "R"])

_SUPPORTED_DTYPES = (torch.float32, torch.float64)


def _validate_mode(mode):
    if mode not in ("reduced", "complete", "r"):
        raise ValueError(
            f"linalg_qr: mode must be one of 'reduced', 'complete', 'r', got {mode!r}"
        )


def _out_shapes(batch_shape, m, n, mode):
    """(Q_shape, R_shape) matching torch.linalg.qr for the given mode."""
    k = min(m, n)
    if mode == "r":
        return (0,), (*batch_shape, k, n)
    if mode == "reduced":
        return (*batch_shape, m, k), (*batch_shape, k, n)
    return (*batch_shape, m, m), (*batch_shape, m, n)


def _validate_out(out, dtype, batch_shape, m, n, mode):
    q_shape, r_shape = _out_shapes(batch_shape, m, n, mode)
    for name, t, shape in (("Q", out[0], q_shape), ("R", out[1], r_shape)):
        if t.dtype != dtype:
            raise RuntimeError(
                f"linalg_qr: expected out tensor {name} to have dtype {dtype}, "
                f"but got {t.dtype}"
            )
        if tuple(t.shape) != tuple(shape):
            raise RuntimeError(
                f"linalg_qr: out tensor {name} has shape {tuple(t.shape)}, "
                f"expected {tuple(shape)}"
            )


def _geqrf(af):
    """Unblocked Householder ``geqrf`` with **shape-invariant** kernels.

    Every per-column step operates on full, fixed-shape tensors (never a
    ``[:, j:, ...]`` slice whose extent shrinks with ``j``).  This matters on
    XPU: a Triton kernel is specialised per operand shape, so a shrinking-slice
    loop recompiles once per iteration -- ~1.5 s each, i.e. tens of minutes for
    a ``k`` of ~1000.  Applying each reflector to the *whole* matrix (masking
    the already-finalised leading columns) keeps every launch on the same
    shape, so the kernels compile once and are reused for all ``k`` steps.

    Returns ``(af, tau, V)``: ``af`` upper triangle holds ``R``; ``V`` (B, m, k)
    holds the reflectors column-packed (1 on the diagonal, tail below), the
    exact layout ``linalg_householder_product`` consumes.
    """
    B, m, n = af.shape
    k = min(m, n)
    tau = torch.zeros(B, k, dtype=af.dtype, device=af.device)
    V = torch.zeros(B, m, k, dtype=af.dtype, device=af.device)
    one = torch.ones(B, 1, dtype=af.dtype, device=af.device)
    # Index masks (functions of the loop index j only, not of the data) are
    # built ONCE with triu/tril/eye -- vendor ops that do NOT route through the
    # float-compare 1d_tile kernel.  Doing ``rows == j`` / ``rows >= j`` per
    # iteration instead compiles an eq_func_scalar / ge_func_scalar kernel
    # specialised on m; on this backend the rank-1 "1d_tile" path fails to
    # legalise 'triton_xpu.cmpf' for some m (e.g. m == 4096), so the whole op
    # dies at compile time.  Row tables are (k, m), the column table is (k, n);
    # column j selects the j-th row of each (a cheap view, no kernel):
    #   ge_tab[j, i] = (i >= j)   lt_tab[j, i] = (i < j)
    #   eye_tab[j, i] = (i == j)  gt_tab[j, c] = (c > j)
    ones_km = torch.ones(k, m, dtype=af.dtype, device=af.device)
    ge_tab = torch.triu(ones_km)  # (k, m) upper incl diagonal: (i >= j)
    lt_tab = torch.tril(ones_km, -1)  # (k, m) strict lower: (i < j)
    eye_tab = torch.eye(k, m, dtype=af.dtype, device=af.device)  # (i == j)
    gt_tab = torch.triu(
        torch.ones(k, n, dtype=af.dtype, device=af.device), 1
    )  # (k, n) strict upper: (c > j)
    for j in range(k):
        # Per-batch scalars stay rank-2 (B, 1): the rank-1 "1d_tile" pointwise
        # path miscompiles float comparisons for some tile sizes; B is small.
        af_col = af[:, :, j]  # (B, m) fixed shape
        ge = ge_tab[j].reshape(1, m)  # (1, m) 0/1 float, rows on/below diagonal
        x = af_col * ge  # (B, m), entries above the diagonal zeroed
        normx = torch.sqrt((x * x).sum(dim=1, keepdim=True))  # (B, 1)
        alpha = af_col[:, j : j + 1]  # (B, 1)
        zero = normx == 0  # (B, 1) zero sub-column -> H = I
        # beta = -sign(alpha) * ||x|| ; sign(0) treated as +1.
        s = torch.where(alpha >= 0, one, -one)
        beta = -s * normx  # (B, 1)
        denom = alpha - beta
        safe_denom = torch.where(zero, one, denom)
        # Reflector: v[i]=0 for i<j, v[j]=1, v[i]=x[i]/(alpha-beta) for i>j.
        v = x / safe_denom  # (B, m) / (B, 1) broadcast
        atdiag = eye_tab[j].reshape(1, m)  # (1, m) 0/1 float
        # v[j] = 1, off-diagonal entries unchanged (float select, no cmpf).
        v = v * (1.0 - atdiag) + atdiag
        safe_beta = torch.where(zero, one, beta)
        tau_j = torch.where(zero, torch.zeros_like(beta), (beta - alpha) / safe_beta)
        tau[:, j : j + 1] = tau_j
        V[:, :, j] = v
        # Apply H = I - tau v v^T to the trailing columns (c > j) of the whole
        # matrix; leading columns are masked out (they already hold final R).
        w = bmm(v.unsqueeze(1), af)  # (B, 1, n) = v^T A
        upd = (tau_j * v).unsqueeze(2) * w  # (B, m, n) outer product
        colmask = gt_tab[j].reshape(1, 1, n)  # (1, 1, n)
        af = af - upd * colmask
        # Finalise column j: R entries above the diagonal stay, diagonal = beta,
        # below the diagonal is zeroed.
        ltf = lt_tab[j].reshape(1, m)  # (1, m)
        af[:, :, j] = af_col * ltf + beta * atdiag
    return af, tau, V


def _assemble_q(V, tau, m, k, qcols):
    """Form Q (B, m, qcols) from the packed reflectors via vendor orgqr."""
    B = V.shape[0]
    if qcols <= k:
        Aq = V[:, :, :qcols].contiguous()
    else:
        # complete mode needs qcols == m columns; pad the reflector block with
        # zeros so orgqr applies the k reflectors to the full identity.
        Aq = torch.zeros(B, m, qcols, dtype=V.dtype, device=V.device)
        Aq[:, :, :k] = V[:, :, :k]
    return linalg_householder_product(Aq, tau)


def _linalg_qr(A, mode="reduced", *, out=None):
    _validate_mode(mode)

    if A.dim() < 2:
        raise RuntimeError("linalg_qr: input must have at least 2 dimensions")
    if A.dtype not in _SUPPORTED_DTYPES:
        raise NotImplementedError(
            "FlagGems linalg_qr currently supports float32 and float64 inputs; "
            f"got dtype={A.dtype}"
        )

    batch_shape = A.shape[:-2]
    m, n = A.shape[-2], A.shape[-1]
    k = min(m, n)
    B = 1
    for d in batch_shape:
        B *= d

    if out is not None:
        _validate_out(out, A.dtype, batch_shape, m, n, mode)

    q_shape, r_shape = _out_shapes(batch_shape, m, n, mode)

    # Degenerate (zero-size) input: torch returns empty factors, except
    # complete mode with zero columns where Q = I.
    if m == 0 or n == 0:
        if out is not None:
            Q, R = out
        else:
            Q = A.new_empty(q_shape)
            R = A.new_empty(r_shape)
        if mode == "complete" and n == 0 and m > 0:
            eye = torch.eye(m, dtype=A.dtype, device=A.device)
            Q.copy_(eye.expand(*batch_shape, m, m))
        return LinalgQrResult(Q, R)

    # Factor a contiguous clone so the caller's input is never mutated.
    af = A.reshape(B, m, n).contiguous().clone()
    af, tau, V = _geqrf(af)

    # R = upper-triangular part of the factored matrix.
    rrows = k if mode in ("reduced", "r") else m
    R_flat = af[:, :rrows, :].triu()

    if mode == "r":
        if out is not None:
            out[1].reshape(B, rrows, n).copy_(R_flat)
            return LinalgQrResult(out[0], out[1])
        return LinalgQrResult(A.new_empty(0), R_flat.reshape(r_shape))

    qcols = k if mode == "reduced" else m
    Q_flat = _assemble_q(V, tau, m, k, qcols)

    if out is not None:
        out[0].reshape(B, m, qcols).copy_(Q_flat)
        out[1].reshape(B, rrows, n).copy_(R_flat)
        return LinalgQrResult(out[0], out[1])

    return LinalgQrResult(Q_flat.reshape(q_shape), R_flat.reshape(r_shape))


def linalg_qr(A, mode="reduced", *, out=None):
    logger.debug("GEMS_KUNLUNXIN LINALG_QR")
    logging.getLogger("flag_gems.ops.linalg_qr").debug("GEMS LINALG_QR")
    return _linalg_qr(A, mode, out=out)


def linalg_qr_out(A, mode="reduced", *, Q, R):
    logger.debug("GEMS_KUNLUNXIN LINALG_QR_OUT")
    logging.getLogger("flag_gems.ops.linalg_qr").debug("GEMS LINALG_QR_OUT")
    return _linalg_qr(A, mode, out=(Q, R))
