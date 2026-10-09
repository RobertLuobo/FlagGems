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

from .mm import mm

logger = logging.getLogger(__name__)


def smm(self, mat):
    """Sparse (COO) matrix ``self`` multiplied by dense matrix ``mat``.

    Mirrors ``torch.smm``: the result is a sparse COO tensor whose rows are
    exactly the rows that appear in the sparse input's structure, each
    materialized with all ``N`` columns.

    XPU3 overlay rationale: the stock COO SpMM kernel reduces each row with a
    runtime-bounded ``tl.sum(axis=0)`` over a 2-D tile inside a data-dependent
    loop. On XPU3 the ``axis=0`` reduce is a hard compile reject ("axis must
    not be 0 for 2D+ shapes"), and every rewrite that keeps the data-dependent
    reduction (transposed 2-D reduce, scalar 1-D axpy loop, atomic scatter)
    sends the XPU3 compiler into a multi-minute explosion that never returns
    inside a 10-minute budget. So this overlay carries no custom kernel: it
    compacts the sparse operand to the columns actually used (O(nnz)), scatters
    its values into a small dense ``(nrows, n_used)`` block -- pure index
    assembly, no numeric compute -- and does the one numeric step, the matmul,
    through the vendor ``mm`` Triton kernel. The compacted matmul keeps the
    working set at O(nnz + nrows * N + n_used * N), never the full ``M * K``.
    """
    logger.debug("GEMS_KUNLUNXIN SMM")

    if not self.is_sparse:
        raise RuntimeError("tensor.sspaddmm(...) can only be called on sparse tensors")
    if self.ndim != 2:
        raise RuntimeError(
            f"sspaddmm: Argument #2: matrices expected, got {self.ndim}D tensor"
        )
    if mat.is_sparse:
        raise RuntimeError(
            "Cannot access data pointer of Tensor that doesn't have storage"
        )
    if mat.ndim != 2:
        raise RuntimeError(
            f"sspaddmm: Argument #3: matrices expected, got {mat.ndim}D tensor"
        )

    M, K = self.shape
    K2, N = mat.shape
    if K != K2:
        raise RuntimeError(f"sspaddmm: Argument #3: Expected dim 0 size {K}, got {K2}")

    if mat.device != self.device:
        raise RuntimeError(
            "Expected all tensors to be on the same device, but got mat2 is on "
            f"{mat.device}, different from other tensors on {self.device}"
        )

    if self.dtype != torch.float32 or mat.dtype != torch.float32:
        raise RuntimeError(
            "GEMS smm is only implemented for float32 inputs, matching torch.smm"
        )

    if not self.is_coalesced():
        self = self.coalesce()

    nnz = self._nnz()
    if nnz == 0 or N == 0 or M == 0:
        out_indices = torch.empty((2, 0), dtype=torch.int64, device=self.device)
        out_values = torch.empty((0,), dtype=torch.float32, device=self.device)
        return torch.sparse_coo_tensor(
            out_indices, out_values, size=(M, N), device=self.device
        )

    indices = self._indices()
    row_indices = indices[0]
    col_indices = indices[1]
    values = self._values().to(torch.float32)

    # Coalesced -> rows are sorted, so unique_consecutive gives the distinct
    # structural rows and, per entry, its compacted structural-row index.
    row_ids, seg_ids = torch.unique_consecutive(row_indices, return_inverse=True)
    nrows = int(row_ids.numel())

    # Compact the contraction axis to only the columns that actually carry a
    # nonzero. ``col_inv`` maps each entry to its index within ``used_cols``.
    used_cols, col_inv = torch.unique(col_indices, return_inverse=True)
    n_used = int(used_cols.numel())

    # Dense block of the sparse operand over the compacted column set. This is
    # a scatter of the stored values (assembly), not a numeric reduction; the
    # input is coalesced so every (row, col) is unique and the writes race-free.
    block = torch.zeros((nrows, n_used), device=self.device, dtype=torch.float32)
    block[seg_ids, col_inv] = values

    # Gather the dense operand's used rows, then the single numeric step runs
    # through the vendor mm Triton kernel.
    mat_sub = mat.index_select(0, used_cols).contiguous()
    out_dense = mm(block, mat_sub)

    # ``out_dense`` is (nrows, N) row-major, so flattening lays the COO values
    # out in ascending structural-row then ascending column order -- exactly
    # what torch.smm emits.
    out_values = out_dense.reshape(-1).contiguous()
    out_row = row_ids.to(torch.int64).repeat_interleave(N)
    out_col = torch.arange(N, device=self.device, dtype=torch.int64).repeat(nrows)
    out_indices = torch.stack([out_row, out_col], dim=0)

    return torch.sparse_coo_tensor(
        out_indices, out_values, size=(M, N), is_coalesced=True
    )
