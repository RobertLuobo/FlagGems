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

from flag_gems.ops._nested_tensor_from_mask import (
    _nested_tensor_from_mask_compact_kernel,
    _nested_tensor_from_mask_offsets_kernel,
)
from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as tle

logger = logging.getLogger(__name__)


@libentry()
@triton.jit
def _nested_from_mask_lengths_only_kernel(
    mask,
    lengths,
    L: tl.constexpr,
    BLOCK: tl.constexpr,
):
    n = tle.program_id(0)
    acc = tl.zeros((BLOCK,), dtype=tl.int32)
    for i in range(0, L, BLOCK):
        offs = i + tl.arange(0, BLOCK)
        m = offs < L
        vals = tl.load(mask + n * L + offs, mask=m, other=0).to(tl.int32)
        acc += vals
    tl.store(lengths + n, tl.sum(acc, axis=0).to(tl.int64))


@libentry()
@triton.jit
def _nested_from_mask_gaps_only_kernel(
    mask,
    gaps,
    L: tl.constexpr,
    BLOCK: tl.constexpr,
):
    n = tle.program_id(0)
    gap_acc = tl.zeros((BLOCK,), dtype=tl.int32)
    for i in range(0, L, BLOCK):
        offs = i + tl.arange(0, BLOCK)
        m = offs < L
        vals = tl.load(mask + n * L + offs, mask=m, other=0).to(tl.int32)
        m_next = (offs + 1) < L
        vals_next = tl.load(mask + n * L + offs + 1, mask=m_next, other=0).to(tl.int32)
        gap_acc += ((vals == 0) & (vals_next == 1)).to(tl.int32)
    tl.store(gaps + n, (tl.sum(gap_acc, axis=0) > 0).to(tl.int8))


def _nested_tensor_from_mask(t, mask, mask_check=True):
    logger.debug("GEMS_KUNLUNXIN _NESTED_TENSOR_FROM_MASK")
    assert t.dim() == 3, "Input should be a 3D tensor, N * L * D"
    assert mask.dim() == 2, "Padding mask should be 2D"
    assert mask.dtype == torch.bool, "Expected mask to be a bool tensor"
    N, L, D = t.shape
    assert mask.shape[0] == N and mask.shape[1] == L, "Mask shape should match input"

    t = t.contiguous()
    mask = mask.contiguous()

    # Per-batch valid-row counts and left-alignment gap flags. These are split
    # into two separate kernels: fusing the count accumulator with the gap
    # accumulator (which issues the cross-row ``offs + 1`` boundary load) in a
    # single kernel miscompiles on XPU3 and drops the final valid lane whenever
    # a row is completely full (length == L).
    lengths = torch.empty((N,), dtype=torch.int64, device=t.device)
    gaps = torch.empty((N,), dtype=torch.int8, device=t.device)
    block_len = max(1, min(triton.next_power_of_2(L), 1024))
    _nested_from_mask_lengths_only_kernel[(N,)](mask, lengths, L, BLOCK=block_len)
    _nested_from_mask_gaps_only_kernel[(N,)](mask, gaps, L, BLOCK=block_len)

    offsets = torch.empty((N + 1,), dtype=torch.int64, device=t.device)
    has_gap = torch.empty((1,), dtype=torch.int8, device=t.device)
    if N <= 4096:
        block_scan = max(2, triton.next_power_of_2(N))
        _nested_tensor_from_mask_offsets_kernel[(1,)](
            lengths, gaps, offsets, has_gap, N, BLOCK=block_scan
        )
    else:
        offsets.copy_(
            torch.cat(
                [
                    torch.zeros(1, dtype=torch.int64, device=t.device),
                    torch.cumsum(lengths, 0),
                ]
            )
        )
        has_gap.copy_(gaps.any().to(torch.int8))

    if mask_check:
        if bool(has_gap.item()):
            raise RuntimeError("Mask must be left-aligned without gaps")

    total_valid = int(offsets[N].item())

    values = torch.empty((total_valid * D,), dtype=t.dtype, device=t.device)
    if total_valid > 0:
        block_compact = max(1, min(triton.next_power_of_2(L * D), 1024))
        _nested_tensor_from_mask_compact_kernel[(N,)](
            t, lengths, offsets, values, D, L, BLOCK=block_compact
        )

    lengths_cpu = lengths.to("cpu")
    offsets_cpu = offsets.to("cpu")

    nested_size = torch.stack(
        [lengths_cpu, torch.full((N,), D, dtype=torch.int64)], dim=1
    )
    nested_strides = torch.empty_like(nested_size)
    nested_strides[:, 0] = D
    nested_strides[:, 1] = 1

    storage_offsets = offsets_cpu[:-1] * D

    return torch.ops.aten._nested_view_from_buffer.default(
        values, nested_size, nested_strides, storage_offsets
    )
