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

"""Kunlunxin XPU overlay for ``top_k_per_row_prefill``.

The generic radix/histogram kernel in ``flag_gems.fused.top_k_per_row_prefill``
crashes the XPU3 compiler in the ``TritonXPUCreateGM2LM`` pass. This overlay
reuses the in-house XPU Triton radix sort (``_kunlunxin/ops/sort.py``) to select
the top-K values per row, then maps absolute indices back to row-relative ones.
"""

import logging
import sys

import torch

from flag_gems.runtime.backend._kunlunxin.ops.sort import sort as _xpu_sort

logger = logging.getLogger(__name__)

# Finite sentinel below any realistic logit, within float32 range. Used to mask
# out-of-range positions so they sort to the bottom of the radix sort.
_NEG_SENTINEL = -3.0e38

# Cap on logits rows processed per radix-sort call, to bound the auxiliary int64
# index buffers the sort allocates for large (num_rows x vocab_size) inputs.
_ROW_CHUNK_ELEMS = 32 * 1024 * 1024


def top_k_per_row_prefill(
    logits, row_starts, row_ends, indices, num_rows, stride0, stride1, top_k
):
    """Top-K per row (prefill phase) via XPU Triton radix sort.

    Writes 0-based indices relative to ``row_starts[i]`` into ``indices``,
    padding unused slots with -1. See the generic implementation for the
    full argument contract.
    """
    logger.debug("GEMS_KUNLUNXIN TOP_K_PER_ROW_PREFILL")

    device = logits.device
    vocab_size = logits.shape[1]
    assert num_rows == logits.shape[0]

    starts = row_starts.to(torch.int64).view(num_rows, 1)
    ends = row_ends.to(torch.int64).view(num_rows, 1)
    cols = torch.arange(vocab_size, device=device).view(1, vocab_size)
    valid = (cols >= starts) & (cols < ends)
    masked = torch.where(
        valid, logits, torch.full_like(logits, _NEG_SENTINEL)
    ).contiguous()

    k_eff = min(top_k, vocab_size)

    # Row-chunked radix sort to bound auxiliary int64 buffer size.
    rows_per_chunk = max(1, _ROW_CHUNK_ELEMS // max(1, vocab_size))
    topk_abs = torch.empty((num_rows, k_eff), dtype=torch.int64, device=device)
    for r0 in range(0, num_rows, rows_per_chunk):
        r1 = min(num_rows, r0 + rows_per_chunk)
        _, sorted_idx = _xpu_sort(masked[r0:r1], dim=1, descending=True)
        topk_abs[r0:r1] = sorted_idx[:, :k_eff]

    rel = (topk_abs - starts).to(torch.int32)

    valid_len = (ends - starts).clamp(min=0)
    col_rank = torch.arange(k_eff, device=device).view(1, k_eff)
    keep = col_rank < valid_len
    rel = torch.where(keep, rel, torch.full_like(rel, -1))

    if k_eff < top_k:
        out = torch.full((num_rows, top_k), -1, dtype=torch.int32, device=device)
        out[:, :k_eff] = rel
        indices.copy_(out)
    else:
        indices.copy_(rel)


def _install():
    from flag_gems.fused.top_k_per_row_prefill import (
        top_k_per_row_prefill as _generic,
    )

    fused_pkg = sys.modules.get("flag_gems.fused")
    if fused_pkg is not None:
        if getattr(fused_pkg, "top_k_per_row_prefill", None) is _generic:
            fused_pkg.top_k_per_row_prefill = top_k_per_row_prefill

    sub = sys.modules.get("flag_gems.fused.top_k_per_row_prefill")
    if sub is not None:
        if getattr(sub, "top_k_per_row_prefill", None) is _generic:
            sub.top_k_per_row_prefill = top_k_per_row_prefill

    top = sys.modules.get("flag_gems")
    if top is not None:
        if getattr(top, "top_k_per_row_prefill", None) is _generic:
            top.top_k_per_row_prefill = top_k_per_row_prefill


_install()
