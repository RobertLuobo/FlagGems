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
"""Kunlunxin (XPU) override for torch.block_diag.

``block_diag`` is a pure block-scatter copy: the output is a zero
``(total_rows, total_cols)`` matrix and each input block is written into its
own ``(row_off:row_off+rows, col_off:col_off+cols)`` tile. On this backend
the Triton forms of that scatter are structurally 10-100x slower than the
vendor's native copy engine:

* A discrete (computed per-lane) STORE is ~400x more expensive than a
  contiguous store and a discrete LOAD ~30x; only 1D ``tl.arange``-derived
  (affine, unit inner stride) pointer vectors lower to block-DMA. A
  block-diagonal scatter needs a data-dependent (row_off, col_off) per block,
  so every Triton variant pays discrete loads or stores on at least one side.
* The per-block base-pointer patterns that would recover contiguity are
  themselves miscompiled on this backend: an int->pointer cast of a
  ``scalar-where`` selected address (``_coalesce16_kernel``) and the
  device-pointer-table + unmasked affine vector load (``_coalesce_blocks_kernel``)
  both silently produce garbage on masked-off / repeated lanes
  (nondeterministic, verified bit-level); a masked table load plus a
  per-lane gather (the old ``varlen_general`` kernel) works but caps out at
  ~3 GB/s.

Native ``torch.block_diag`` on this device is a memcpy-class op (fill +
per-block strided copy, ~0.015-0.05 ms for the benchmark shapes), so the
override mirrors the native semantics exactly: ``torch.zeros`` output plus
one ``torch.ops.aten.slice`` view (``out[r0:r1, c0:c1]``) and one
``torch.ops.aten._copy_from(src, view, False)`` per block. gems overrides
``copy_``/``copy``/``cat``/``stack`` but never ``_copy_from``, so every copy
reaches the vendor's native strided-copy engine regardless of
``flag_gems.use_gems`` being active; this is the same pattern already used by
the accepted kunlunxin ``slice_backward``/``resize``/``constant_pad_nd``
overrides. 0D inputs are promoted to (1,1), 1D to (1,K); mixed dtypes are
promoted with ``torch.promote_types`` exactly as ATen does; empty blocks
contribute no copy (the output is already zeroed).
"""

import logging

import torch

logger = logging.getLogger("flag_gems").getChild(__name__.lstrip("."))


def block_diag(*tensors):
    """Block diagonal matrix construction (Kunlunxin override)."""
    logger.debug("GEMS_KUNLUNXIN BLOCK_DIAG")

    # Handle case where tensors is passed as a single list/tuple
    if len(tensors) == 1 and isinstance(tensors[0], (list, tuple)):
        tensors = tuple(tensors[0])

    if len(tensors) == 0:
        return torch.empty((1, 0))

    # Normalize to 2D: 0D -> (1, 1), 1D -> (1, K), 2D as-is.
    tensors_2d = []
    for t in tensors:
        if t.ndim == 0:
            tensors_2d.append(t.unsqueeze(0).unsqueeze(0))
        elif t.ndim == 1:
            tensors_2d.append(t.unsqueeze(0))
        else:
            assert t.ndim == 2, f"Expected 0D, 1D, or 2D tensor, got {t.ndim}D"
            tensors_2d.append(t)

    total_rows = sum(t.shape[0] for t in tensors_2d)
    total_cols = sum(t.shape[1] for t in tensors_2d)

    out_dtype = tensors_2d[0].dtype
    for t in tensors_2d[1:]:
        out_dtype = torch.promote_types(out_dtype, t.dtype)
    device = tensors_2d[0].device

    out = torch.zeros((total_rows, total_cols), dtype=out_dtype, device=device)

    row_off = 0
    col_off = 0
    for t in tensors_2d:
        rows, cols = t.shape
        if rows > 0 and cols > 0:
            src = (
                t
                if (t.is_contiguous() and t.dtype == out_dtype)
                else t.contiguous().to(out_dtype)
            )
            view = torch.ops.aten.slice(
                torch.ops.aten.slice(out, 0, row_off, row_off + rows),
                1,
                col_off,
                col_off + cols,
            )
            torch.ops.aten._copy_from(src, view, False)
        row_off += rows
        col_off += cols

    return out
