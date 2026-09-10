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

"""Kunlunxin backend override for ``linalg_svdvals``.

The generic ``flag_gems.ops.linalg_svdvals`` routes through the generic CUDA
Triton SVD kernels (``_small_jacobi_svals_kernel`` /
``_blocked_jacobi_svals_kernel`` in ``flag_gems/ops/svd.py``), whose
constexpr-tiled kernels do not finish compiling on the Triton-XPU backend
(``xpu.llvm.translate_to_asm`` still running after 900 s for a (16, 16) tile;
``pytest-timeout`` killed the benchmark baseline).  The overload store is also
CPU/ATen-fallback-free in the generic path only for shapes whose Triton kernels
compile, which does not hold on XPU.

This override reuses the Kunlunxin ``linalg_svd`` one-sided Jacobi pipeline
(``_osj_pipeline`` in ``linalg_svd.py``), which is built from *runtime* loops
(nothing constexpr-unrolled beyond ``tl.arange`` tiles) and therefore compiles
fast on XPU, and returns only the singular values.  The ``U``/``Vh`` factors of
the full SVD are computed and then discarded, which costs one extra
``(k*m*n)`` matmul on the host side, but keeps a single already-validated
kernel path (``tests/test_linalg_svd.py`` 28/28 clean pass).

dtype: float32 only (matching the generic linalg_svdvals contract).
"""

import logging

import torch

from .linalg_svd import _osj_svd_impl

logger = logging.getLogger(__name__)


def linalg_svdvals(A: torch.Tensor, driver: str = None) -> torch.Tensor:
    """Computes the singular values of a matrix (Kunlunxin XPU).

    Args:
        A: Input tensor of shape (*, m, n) where * is zero or more batch dimensions.
        driver: Accepted for API compatibility; the one-sided Jacobi pipeline
            does not use a solver driver selection.

    Returns:
        Singular values in descending order, shape (*, min(m, n)).
    """
    logger.debug("GEMS LINALG_SVDVALS (kunlunxin)")
    if A.dtype != torch.float32:
        raise TypeError(f"linalg_svdvals only supports float32 input, got {A.dtype}")
    if not A.is_contiguous():
        A = A.contiguous()

    if A.dim() not in (2, 3):
        # Flatten extra batch dims (the pipeline loops over one batch axis).
        orig_shape = A.shape
        m, n = orig_shape[-2:]
        k = min(m, n)
        A = A.reshape(-1, m, n)
        _, S, _ = _osj_svd_impl(A, full_matrices=False)
        return S.reshape(*orig_shape[:-2], k)

    # (U, S, Vh) -> keep only S (already sorted in descending order).
    _, S, _ = _osj_svd_impl(A, full_matrices=False)
    return S
