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

import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

from .linalg_solve_ex import linalg_solve_ex

logger = logging.getLogger(__name__)


@libentry()
@triton.jit
def _csr_to_dense_kernel(crow_ptr, col_ptr, val_ptr, dense_ptr, n_cols):
    row = tl.program_id(0)
    start = tl.load(crow_ptr + row).to(tl.int64)
    end = tl.load(crow_ptr + row + 1).to(tl.int64)
    row_base = row.to(tl.int64) * n_cols
    for j in range(start, end):
        col = tl.load(col_ptr + j).to(tl.int64)
        val = tl.load(val_ptr + j)
        tl.store(dense_ptr + row_base + col, val)


def _csr_to_dense(A):
    """Densify a 2-D sparse CSR tensor with a per-row scalar scatter kernel.

    The generic vectorized masked-scatter kernel deterministically drops the
    row-0 single-element write on XPU3, so each nnz is stored individually.
    """
    n_rows, n_cols = A.shape
    crow = A.crow_indices().contiguous()
    col = A.col_indices().contiguous()
    val = A.values().contiguous()
    dense = torch.zeros((n_rows, n_cols), dtype=A.dtype, device=A.device)
    if n_rows == 0 or n_cols == 0 or val.numel() == 0:
        return dense
    with torch_device_fn.device(A.device):
        _csr_to_dense_kernel[(n_rows,)](crow, col, val, dense, n_cols)
    return dense


def spsolve(A, B, *, left=True):
    """Solve the sparse linear system ``A @ X = B`` for a CSR matrix ``A``.

    Mirrors ``torch.ops.aten._spsolve``: only a 1-D right-hand side and
    ``left=True`` are supported. The sparse operand is densified with a Triton
    scatter kernel and the dense system is solved with the vendor LU + triangular
    solve path (``linalg_solve_ex``); the generic single-kernel ``linalg_solve``
    produces NaN on XPU3.
    """
    logger.debug("GEMS_KUNLUNXIN SPSOLVE")

    if A.layout != torch.sparse_csr:
        raise RuntimeError(
            f"spsolve: expected A to have sparse_csr layout, but got {A.layout}"
        )
    if A.dim() != 2 or A.shape[0] != A.shape[1]:
        raise RuntimeError(
            f"spsolve: expected A to be a square 2-D matrix, but got shape {tuple(A.shape)}"
        )
    if not left:
        raise RuntimeError("spsolve: only left=True is supported")
    if B.dim() != 1:
        raise RuntimeError(
            f"spsolve: expected B to be a 1-D tensor, but got shape {tuple(B.shape)}"
        )
    if B.size(0) != A.size(0):
        raise RuntimeError(
            f"spsolve: linear system size mismatch: A is {tuple(A.shape)}, "
            f"B is {tuple(B.shape)}"
        )
    if B.device != A.device:
        raise RuntimeError(
            f"spsolve: expected A and B to be on the same device, "
            f"but got A on {A.device} and B on {B.device}"
        )

    A_dense = _csr_to_dense(A)

    out_dtype = A.dtype
    compute_dtype = (
        torch.float32 if out_dtype in (torch.float16, torch.bfloat16) else out_dtype
    )
    A_solve = A_dense.to(compute_dtype)
    B_solve = B.to(compute_dtype)

    X = linalg_solve_ex(A_solve, B_solve.unsqueeze(-1)).result
    return X.squeeze(-1).to(out_dtype)
