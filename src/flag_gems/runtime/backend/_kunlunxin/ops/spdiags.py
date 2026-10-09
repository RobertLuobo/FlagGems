# Copyright 2026, The FlagOS Contributors.
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
from flag_gems.runtime.backend._kunlunxin.ops.index_select import index_select
from flag_gems.runtime.backend._kunlunxin.ops.nonzero import nonzero
from flag_gems.runtime.backend._kunlunxin.ops.unique import _unique2
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)


@libentry()
@triton.jit
def spdiags_kernel(
    diagonals_ptr,
    offsets_ptr,
    row_ptr,
    col_ptr,
    values_ptr,
    numel,
    diag_len: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Flat gather over the ``num_diags * diag_len`` scratch layout.

    Slot ``s`` corresponds to element ``p = s % diag_len`` of diagonal
    ``d = s // diag_len``. Following the native ``_spdiags`` contract, that
    element lands at matrix coordinate ``(p - offset, p)``. The kernel writes
    every slot unconditionally (only the trailing ``s >= numel`` lanes are
    masked), leaving all out-of-matrix filtering to the host; it never performs
    a per-diagonal masked scatter.

    This flat form is deliberate: the previous per-diagonal kernel used masked
    int64 scatter stores plus a ``tl.where`` sentinel, which miscompiled on XPU3
    (``program_id 0`` produced garbage columns and the ``mask`` was not honored,
    and the ``tl.where`` additionally aborted TritonXPUUnrollControl).
    """
    pid = tl.program_id(0)
    s = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    m = s < numel

    d = s // diag_len
    p = s % diag_len

    offset = tl.load(offsets_ptr + d, mask=m, other=0)

    tl.store(row_ptr + s, p - offset, mask=m)
    tl.store(col_ptr + s, p, mask=m)
    tl.store(values_ptr + s, tl.load(diagonals_ptr + s, mask=m, other=0.0), mask=m)


def _to_layout(result_coo, layout):
    if layout == torch.sparse_csr:
        return result_coo.to_sparse_csr()
    if layout == torch.sparse_csc:
        return result_coo.to_sparse_csc()
    return result_coo


def spdiags(diagonals, offsets, shape, layout=None):
    logger.debug("GEMS_KUNLUNXIN SPDIAGS")

    diagonals_2d = diagonals.unsqueeze(0) if diagonals.dim() == 1 else diagonals
    offsets_1d = offsets.unsqueeze(0) if offsets.dim() == 0 else offsets

    if diagonals_2d.dim() != 2:
        raise RuntimeError("Diagonals must be vector or matrix")
    if offsets_1d.dim() != 1:
        raise RuntimeError("Offsets must be scalar or vector")
    if len(shape) != 2:
        raise RuntimeError("Output shape must be 2d")

    if layout is not None and layout not in (
        torch.sparse_coo,
        torch.sparse_csr,
        torch.sparse_csc,
    ):
        raise RuntimeError(
            "Only output layouts (Sparse, SparseCsc, SparseCsr) are supported, "
            f"got {layout}"
        )

    if offsets_1d.dtype != torch.int64:
        raise RuntimeError(
            f"Offset Tensor must have dtype Long but got {offsets_1d.dtype}"
        )

    num_diags = diagonals_2d.shape[0]
    diag_len = diagonals_2d.shape[1]
    nrows, ncols = shape

    if offsets_1d.shape[0] != num_diags:
        raise RuntimeError(
            f"Number of diagonals ({num_diags}) does not match "
            f"the number of offsets ({offsets_1d.shape[0]})"
        )

    # Duplicate detection via the kunlunxin unique overlay. The generic
    # ``flag_gems.ops.unique._unique2`` kernel returns wrong results on XPU
    # (e.g. ``[0, 1, -1] -> [-1]``), which spuriously flagged distinct offsets
    # as duplicates; the vendor overlay deduplicates correctly.
    if offsets_1d.numel() > 0:
        unique_offsets = _unique2(offsets_1d, sorted=False)[0]
        if offsets_1d.numel() != unique_offsets.numel():
            raise RuntimeError("Offset tensor contains duplicate values")

    if num_diags == 0 or diag_len == 0 or nrows == 0 or ncols == 0:
        indices = torch.empty((2, 0), dtype=torch.int64, device=diagonals.device)
        values = torch.empty((0,), dtype=diagonals.dtype, device=diagonals.device)
        result = torch.sparse_coo_tensor(
            indices, values, size=shape, dtype=diagonals.dtype, device=diagonals.device
        )
        return _to_layout(result, layout)

    numel = num_diags * diag_len
    row_buffer = torch.empty((numel,), dtype=torch.int64, device=diagonals.device)
    col_buffer = torch.empty((numel,), dtype=torch.int64, device=diagonals.device)
    values_buffer = torch.empty(
        (numel,), dtype=diagonals.dtype, device=diagonals.device
    )

    BLOCK_SIZE = 1024
    grid = (triton.cdiv(numel, BLOCK_SIZE),)

    diagonals_contig = diagonals_2d.contiguous()
    offsets_contig = offsets_1d.contiguous()

    with torch_device_fn.device(diagonals.device):
        spdiags_kernel[grid](
            diagonals_contig,
            offsets_contig,
            row_buffer,
            col_buffer,
            values_buffer,
            numel,
            diag_len,
            BLOCK_SIZE=BLOCK_SIZE,
        )

    # Keep only coordinates that fall inside the matrix. These are plain index
    # comparisons on the scratch buffers (bookkeeping over the kernel-gathered
    # values), compacted through the FlagGems nonzero/index_select overlays.
    valid = (
        (row_buffer >= 0)
        & (row_buffer < nrows)
        & (col_buffer >= 0)
        & (col_buffer < ncols)
    )
    keep = nonzero(valid).reshape(-1)
    rows = index_select(row_buffer, 0, keep)
    cols = index_select(col_buffer, 0, keep)
    values = index_select(values_buffer, 0, keep)
    indices = torch.stack([rows, cols])

    result = torch.sparse_coo_tensor(
        indices, values, size=shape, dtype=diagonals.dtype, device=diagonals.device
    )
    return _to_layout(result, layout)
