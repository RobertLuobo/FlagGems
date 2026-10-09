import logging

from flag_gems.ops.linalg_norm import _parse_ord, _v_norm
from flag_gems.ops.vector_norm import vector_norm

from .linalg_matrix_norm import linalg_matrix_norm

logger = logging.getLogger(__name__)


def linalg_norm(A, ord=None, dim=None, keepdim=False, *, dtype=None):
    """Kunlunxin overlay for ``torch.linalg.norm``.

    The generic ``flag_gems.ops.linalg_norm`` imports the *generic*
    ``linalg_matrix_norm`` at module load time, so its matrix branch bypasses
    the vendor override and runs generic Triton kernels whose ``tl.sum(axis=0)``
    / ``tl.dot`` / SVD paths fail to lower on XPU3 (``axis must not be 0 for 2D+
    shapes``).  This overlay keeps the generic dispatch logic verbatim but sends
    this overlay keeps the generic dispatch logic but (a) sends the matrix
    branch to the vendor ``linalg_matrix_norm`` and (b) routes every vector
    p-norm with ``ord`` outside ``{2, inf, -inf, 0}`` -- including the 1D
    full-reduce case -- through the generic ``_v_norm`` fixed kernel.  The
    generic ``vector_norm``'s dedicated ``l1`` / ``lp`` kernels miscompile on
    XPU3 (``pow`` -> ``undefined symbol: Unsupported``), while ``_v_norm``'s
    kernel lowers cleanly; the remaining ords (2 / +-inf / 0) keep using the
    generic ``vector_norm``, which already lowers and passes on this backend.
    """
    logger.debug("GEMS_KUNLUNXIN LINALG_NORM")
    ord = _parse_ord(ord)
    if dim is not None:
        dim = [dim] if isinstance(dim, int) else list(dim)
        if len(dim) not in (1, 2):
            raise RuntimeError(
                f"linalg.norm: If dim is specified, it must be of length 1 or 2. "
                f"Got {dim}."
            )
    elif ord is not None:
        if A.ndim not in (1, 2):
            raise RuntimeError(
                "linalg.norm: If dim is not specified but ord is, "
                f"the input must be 1D or 2D. Got {A.ndim}D."
            )
    if (
        isinstance(ord, str)
        or (dim is not None and len(dim) == 2)
        or (dim is None and A.ndim == 2)
    ):
        return linalg_matrix_norm(
            A,
            "fro" if ord is None else ord,
            (-2, -1) if dim is None else dim,
            keepdim,
            dtype=dtype,
        )
    ord = 2 if ord is None else ord
    if ord not in (2, float("inf"), float("-inf"), 0):
        reduce_dims = dim if dim is not None else list(range(A.ndim))
        return _v_norm(A, ord, reduce_dims, keepdim, dtype)
    return vector_norm(A, ord, dim, keepdim, dtype=dtype)
