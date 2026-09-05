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
"""Kunlunxin (XPU) ``linalg_norm``.

``flag_gems.ops.linalg_norm`` (the generic implementation the ``SpecOpRegistrar``
replaces for every name that ``_kunlunxin.ops`` exports) imports
``vector_norm`` and ``linalg_matrix_norm`` from ``flag_gems.ops`` at *import
time*, so it always runs the generic Triton kernels.  Those are unusable on
this backend:

* the generic ``vector_norm`` kernels never cast the (possibly integral)
  ``ord`` argument, and the XPU libdevice ``pow`` maps ``(fp32, int32)`` to
  the literal symbol ``"Unsupported"``, so ``ord`` in (1, 3, -1) is a hard
  linker failure on the 1D full-reduction path;
* the generic ``linalg_matrix_norm`` reduces 2-D tiles with ``axis=0`` and
  mixes atomics with masked ``other=`` tails, both rejected / silently
  mis-lowered on this backend.

This override reimplements the ``torch.linalg.norm`` dispatch with the exact
same validation/semantics as the generic module, but binds the local
XPU-safe ``vector_norm`` / ``linalg_matrix_norm`` overrides.  The vector
branch additionally honours a wider ``dtype=`` (e.g. ``torch.float64``) the
XPU kernels cannot compute in: the reduction runs in fp32 and the result is
cast to the requested dtype, matching torch's widen-only semantics to the
precision of an fp32 kernel.
"""

import logging

import torch

from flag_gems.runtime.backend._kunlunxin.ops.linalg_matrix_norm import (
    linalg_matrix_norm,
)
from flag_gems.runtime.backend._kunlunxin.ops.vector_norm import vector_norm

logger = logging.getLogger(__name__)


def _parse_ord(ord):
    """Normalize the ord value arriving from aten.

    The aten ``linalg_norm`` schema declares ``Scalar? ord``, and the dispatcher
    passes numeric orders through as their string form (e.g. "2", "-1", "inf").
    Parse those back to floats; keep "fro"/"nuc" as strings.
    """
    if isinstance(ord, str) and ord not in ("fro", "nuc"):
        return float(ord)
    return ord


def linalg_norm(A, ord=None, dim=None, keepdim=False, *, dtype=None):
    """Mirror ``torch.linalg.norm`` dispatch (torch routes this op to
    ``linalg_matrix_norm`` or ``linalg_vector_norm``):

    - matrix branch when ``ord`` is "fro"/"nuc", ``dim`` is a 2-tuple, or the
      input is 2D with dim=None (numeric ords then use the matrix norm over
      the last two dims).  Reuses the kunlunxin ``linalg_matrix_norm``.
    - vector branch otherwise: ``dim`` as int/1-tuple, or ``dim=None`` which
      flattens the input before applying the vector norm.  Reuses the
      kunlunxin ``vector_norm``.

    The XPU kernels compute in fp16/fp32/bf16 only; a wider requested
    ``dtype`` (torch's widen-only semantics) is honoured by running the
    reduction in fp32 and casting the result.
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
    # Matrix branch: ord='fro'/'nuc', an explicit 2-tuple dim, or a 2D input
    # with dim=None (torch applies the matrix norm over the last two dims
    # there, whether ord is numeric or None).
    if (
        isinstance(ord, str)
        or (dim is not None and len(dim) == 2)
        or (dim is None and A.ndim == 2)
    ):
        # ord=None on the matrix branch means the Frobenius norm.
        return linalg_matrix_norm(
            A,
            "fro" if ord is None else ord,
            (-2, -1) if dim is None else dim,
            keepdim,
            dtype=dtype,
        )
    ord = 2 if ord is None else ord
    if dtype is not None and dtype not in (
        torch.float16,
        torch.float32,
        torch.bfloat16,
    ):
        # XPU vector_norm kernels have no fp64 path.  Widen-only semantics:
        # compute in the (fp32) reduction dtype, then cast to the requested
        # output dtype.  Casting an fp32 result to fp64 is exact, and the
        # fp32 kernel error is well inside the test tolerance.
        return vector_norm(A, ord, dim, keepdim, dtype=None).to(dtype)
    return vector_norm(A, ord, dim, keepdim, dtype=dtype)