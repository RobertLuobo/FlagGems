import logging
import math
from collections import namedtuple

import torch

from .bmm import bmm
from .linalg_lu_factor_ex import linalg_lu_factor_ex
from .linalg_solve_triangular import linalg_solve_triangular
from .lu_unpack import lu_unpack

logger = logging.getLogger(__name__)

LinalgSolveExResult = namedtuple("LinalgSolveExResult", ["result", "info"])


# ---------------------------------------------------------------------------
# Why this vendor override exists
# ---------------------------------------------------------------------------
# The generic ``flag_gems.ops.linalg_solve_ex`` fuses the whole LU
# factorisation + forward/backward substitution into a single program that
# drives data-dependent in-kernel ``while`` loops with repeated
# read-modify-write to the same global LU workspace.  On the P800 TritonXPU
# backend the generated loop code reads stale global memory: the trailing
# rank-1 update stores are not made visible to the loads of later pivot rows,
# so the ``L`` multipliers accumulate error that grows with ``n`` (correct for
# ``n<=10``, ~100% mismatch for ``n>=12``).  ``tl.debug_barrier`` reduces but
# never removes the corruption, confirming a memory-coherence problem rather
# than a repairable numerics bug.
#
# This override sidesteps the broken fused kernel entirely by composing the
# already-registered vendor XPU kernels (same tactic as
# ``linalg_matrix_power``):
#   A = P @ L @ U      (``linalg_lu_factor_ex`` -> ``lu_unpack``)
#   A X = B  ->  L U X = P^T B
#     Y = L^{-1} (P^T B)   (unit-lower ``linalg_solve_triangular``)
#     X = U^{-1} Y         (upper      ``linalg_solve_triangular``)
# The ``info`` tensor is produced directly by ``linalg_lu_factor_ex`` and
# already carries LAPACK ``getrf`` semantics (0 = success, k = zero pivot at
# 1-based index k), matching ``torch.linalg.solve_ex``.  No ATen / native /
# composite compute fallback is used; ``bmm`` applies the row permutation.
# ---------------------------------------------------------------------------


def linalg_solve_ex(A, B, *, left=True, check_errors=False):
    """Solve ``AX = B`` returning ``(result, info)`` via vendor LU + trsm."""
    logger.debug("GEMS_KUNLUNXIN LINALG_SOLVE_EX")

    if not left:
        raise NotImplementedError("right=True (XA = B) is not yet supported")
    assert A.dtype in (
        torch.float32,
        torch.float64,
    ), f"linalg_solve_ex requires float32/float64, got {A.dtype}"
    if A.ndim < 2 or B.ndim < 2:
        raise ValueError("A and B must be at least 2D")
    if A.shape[-1] != A.shape[-2]:
        raise ValueError("A must be a square matrix")
    n = A.shape[-1]
    if B.shape[-2] != n:
        raise ValueError("B must have compatible dimensions with A")

    batch_shape = A.shape[:-2]
    if A.numel() == 0 or B.numel() == 0:
        info = torch.zeros(batch_shape, dtype=torch.int32, device=A.device)
        return LinalgSolveExResult(B.clone(), info)

    batch = math.prod(batch_shape) if batch_shape else 1
    nrhs = B.shape[-1]

    A_flat = A.reshape(batch, n, n).contiguous()
    B_flat = B.reshape(batch, n, nrhs).contiguous()

    # A = P @ L @ U with partial pivoting.
    lu, pivots, info = linalg_lu_factor_ex(A_flat)
    p, l, u = lu_unpack(lu, pivots)

    # A X = B  =>  L U X = P^T B
    pt = p.transpose(-2, -1).contiguous()
    ptb = bmm(pt, B_flat)
    # Y = L^{-1} (P^T B), L is unit lower triangular.
    y = linalg_solve_triangular(l, ptb, upper=False, left=True, unitriangular=True)
    # X = U^{-1} Y, U is upper triangular.
    x = linalg_solve_triangular(u, y, upper=True, left=True)

    result = x.reshape(B.shape)
    info = info.reshape(batch_shape) if batch_shape else info.reshape(())

    if check_errors and torch.any(info != 0):
        raise torch.linalg.LinAlgError(
            "linalg.solve_ex: The diagonal element of the LU decomposition is zero."
        )

    return LinalgSolveExResult(result, info)
