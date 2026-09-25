import logging
import math
from collections import namedtuple

import torch

from .linalg_lu_factor_ex import linalg_lu_factor_ex
from .linalg_solve_triangular import linalg_solve_triangular
from .lu_unpack import lu_unpack

logger = logging.getLogger(__name__)

LinalgInvExResult = namedtuple("LinalgInvExResult", ["inverse", "info"])


# ---------------------------------------------------------------------------
# Why this vendor override exists
# ---------------------------------------------------------------------------
# The generic ``flag_gems.ops.linalg_inv_ex`` fuses the whole LU factorisation
# + in-kernel forward/backward substitution into a single program that drives
# data-dependent ``while`` loops with repeated read-modify-write to the same
# global LU workspace.  On the P800 TritonXPU backend the generated loop code
# reads stale global memory (the trailing rank-1 update stores are not made
# visible to the loads of later pivot rows), so the ``L`` multipliers
# accumulate error that grows with ``n`` -> ~100% numeric mismatch for larger
# matrices; small blocks additionally trip uni_sram OutOfResources.  This is a
# memory-coherence / codegen problem, not a repairable numerics bug.
#
# This override sidesteps the broken fused kernel by composing already
# registered vendor XPU kernels (same tactic as ``linalg_solve_ex`` and
# ``linalg_matrix_power``).  A^{-1} is obtained by solving ``A X = I``:
#   A = P @ L @ U             (``linalg_lu_factor_ex`` -> ``lu_unpack``)
#   A X = I  ->  L U X = P^T I = P^T
#     Y = L^{-1} P^T          (unit-lower ``linalg_solve_triangular``)
#     X = U^{-1} Y            (upper      ``linalg_solve_triangular``)
# ``info`` is produced directly by ``linalg_lu_factor_ex`` and already carries
# LAPACK ``getrf`` semantics (0 = success, k = zero pivot at 1-based index k),
# matching ``torch.linalg.inv_ex``.  ``torch.linalg.inv`` is a composite that
# dispatches to ``inv_ex`` with ``check_errors=True``, so this override covers
# both markers.  No ATen / native / composite compute fallback is used; the row
# permutation is applied via a pure transpose view of the unit RHS.
# ---------------------------------------------------------------------------


def linalg_inv_ex(A, *, check_errors=False):
    """Compute ``A^{-1}`` returning ``(inverse, info)`` via vendor LU + trsm."""
    logger.debug("GEMS_KUNLUNXIN LINALG_INV_EX")

    assert A.ndim >= 2, "Input must be at least 2D"
    n = A.shape[-1]
    assert A.shape[-2] == n, "Input must be a square matrix"
    assert A.dtype in (
        torch.float32,
        torch.float64,
    ), f"linalg_inv_ex: unsupported dtype {A.dtype}, requires float32 or float64"

    device = A.device
    dtype = A.dtype
    batch_shape = A.shape[:-2]

    if A.numel() == 0:
        inverse = A.clone()
        info = torch.zeros(batch_shape, dtype=torch.int32, device=device)
        return LinalgInvExResult(inverse, info)

    batch = math.prod(batch_shape) if batch_shape else 1
    A_flat = A.reshape(batch, n, n).contiguous()

    # A = P @ L @ U with partial pivoting.
    lu, pivots, info = linalg_lu_factor_ex(A_flat)
    p, l, u = lu_unpack(lu, pivots)

    # A X = I  =>  L U X = P^T I = P^T.  P is a permutation matrix so P^T I is
    # simply its transpose (a pure view / copy, no compute fallback).
    ptb = p.transpose(-2, -1).contiguous()
    # Y = L^{-1} P^T, L is unit lower triangular.
    y = linalg_solve_triangular(l, ptb, upper=False, left=True, unitriangular=True)
    # X = U^{-1} Y, U is upper triangular.
    x = linalg_solve_triangular(u, y, upper=True, left=True)

    inverse = x.reshape(*batch_shape, n, n).to(dtype)
    info = info.reshape(list(batch_shape))

    if check_errors and torch.any(info != 0):
        raise torch.linalg.LinAlgError(
            "torch.linalg.inv_ex: The diagonal element of the LU "
            "decomposition is zero."
        )

    return LinalgInvExResult(inverse, info)
