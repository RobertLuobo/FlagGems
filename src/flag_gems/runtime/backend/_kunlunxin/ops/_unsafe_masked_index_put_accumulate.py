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
from flag_gems.utils import triton_lang_extension as ext

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Path selection
#
# Two implementations of `unsafe_masked_index_put_accumulate` on this backend:
#
# 1. `_scan_impl` -- the historical bitwise scan: one program per input
#    element, each scanning the whole mask (O(N*M) lane ops).  Correct and
#    unbeatable for tiny shapes (its whole cost is a few scalar loads), but
#    quadratic: N=M=131072 ran at 2.4 s vs torch's 5 ms.
#
# 2. `_pipeline_impl` -- sort-based run-total pipeline, O((N+M) log M):
#    * host (XPU device) auxiliary: flatten/clamp/wrap per-dim indices,
#      gather active (target, value) pairs, `torch.sort` by target, one
#      `torch.cumsum` + two boolean-mask gathers to produce per-lane run
#      totals `w` (w[j] = sum of the values whose target equals st[j]);
#    * Triton kernel 1: `contrib[st[j]] = w[j]` -- an unmasked, non-atomic,
#      IDEMPOTENT store: every lane of a run has the same (st, w), i.e. all
#      writers of one address write identical bytes, so no race/atomicity is
#      needed (this matters because discrete `tl.atomic_add` is broken on
#      this backend: every one of 8192 unique targets was off by ~1e2);
#    * Triton kernel 2: `input[off] += contrib[off]` -- a contiguous
#      read-blend-write (the `_mask_scatter_*` pattern, incl. the
#      `tl.where(off < N, off, 0)` idempotent tail trick).
#
# The only device-side data movement of `input` (read + write) lives in the
# Triton kernels; the torch ops above are index arithmetic only (the same
# role `torch.cumsum` plays for the `_bool_blend` rank in index_put_impl).
#
# The scan path is kept for small shapes where its fixed cost is lower than
# the pipeline's host launch overhead (sort + ~8 small kernels).
# ---------------------------------------------------------------------------

# O(N*M) scan path stays below this lane-op budget; above it the pipeline is
# used (measure on the target shapes: [64] scan 1.03x, [8,128] pipeline vs
# 0.18x scan, [4096] pipeline vs 0.105x scan, [2,1024,64] pipeline vs
# 0.002x scan).
_SCAN_MAX_WORK = 1 << 20

_PIPELINE_BLOCK = 2048   # tl.cumsum-free; only loads/stores, BLOCK free
_BLEND_BLOCK = 4096


@libentry()
@triton.jit(do_not_specialize=["mask_numel"])
def _unsafe_masked_index_put_accumulate_kernel(
    input,
    mask,
    index0,
    index1,
    index2,
    values,
    mask_numel,
    SHAPE0: tl.constexpr,
    SHAPE1: tl.constexpr,
    SHAPE2: tl.constexpr,
    STRIDE0: tl.constexpr,
    STRIDE1: tl.constexpr,
    STRIDE2: tl.constexpr,
    RANK: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    output_offset = ext.program_id(0)

    if RANK == 1:
        coordinate0 = output_offset
        coordinate1 = 0
        coordinate2 = 0
        input_offset = coordinate0 * STRIDE0
    elif RANK == 2:
        coordinate0 = output_offset // SHAPE1
        coordinate1 = output_offset % SHAPE1
        coordinate2 = 0
        input_offset = coordinate0 * STRIDE0 + coordinate1 * STRIDE1
    else:
        coordinate0 = output_offset // (SHAPE1 * SHAPE2)
        remainder = output_offset % (SHAPE1 * SHAPE2)
        coordinate1 = remainder // SHAPE2
        coordinate2 = remainder % SHAPE2
        input_offset = (
            coordinate0 * STRIDE0 + coordinate1 * STRIDE1 + coordinate2 * STRIDE2
        )

    offsets = tl.arange(0, BLOCK_SIZE)
    active = offsets < mask_numel
    selected = tl.load(mask + offsets, mask=active, other=0) != 0
    selected &= (
        tl.load(index0 + offsets, mask=active, other=0).to(tl.int32) == coordinate0
    )
    if RANK >= 2:
        selected &= (
            tl.load(index1 + offsets, mask=active, other=0).to(tl.int32) == coordinate1
        )
    if RANK == 3:
        selected &= (
            tl.load(index2 + offsets, mask=active, other=0).to(tl.int32) == coordinate2
        )

    update_values = tl.load(values + offsets, mask=active, other=0.0).to(tl.float32)
    update = tl.sum(tl.where(selected, update_values, 0.0), axis=0)
    original = tl.load(input + input_offset).to(tl.float32)
    tl.store(input + input_offset, original + update)


def _unsafe_masked_index_put_accumulate(input, mask, indices, values):
    logger.debug("GEMS_KUNLUNXIN _UNSAFE_MASKED_INDEX_PUT_ACCUMULATE")
    rank = input.ndim
    if rank < 1 or rank > 3 or len(indices) != rank:
        raise RuntimeError(
            "Kunlunxin _unsafe_masked_index_put_accumulate supports ranks 1 to 3"
        )
    if input.numel() == 0 or mask.numel() == 0:
        return input

    mask_contiguous = mask.contiguous()
    values_contiguous = values.contiguous()
    contiguous_indices = [index.contiguous() for index in indices]
    while len(contiguous_indices) < 3:
        contiguous_indices.append(contiguous_indices[0])

    shape = list(input.shape) + [1] * (3 - rank)
    strides = list(input.stride()) + [0] * (3 - rank)
    block_size = triton.next_power_of_2(mask.numel())

    with torch_device_fn.device(input.device):
        _unsafe_masked_index_put_accumulate_kernel[(input.numel(),)](
            input,
            mask_contiguous,
            contiguous_indices[0],
            contiguous_indices[1],
            contiguous_indices[2],
            values_contiguous,
            mask.numel(),
            SHAPE0=shape[0],
            SHAPE1=shape[1],
            SHAPE2=shape[2],
            STRIDE0=strides[0],
            STRIDE1=strides[1],
            STRIDE2=strides[2],
            RANK=rank,
            BLOCK_SIZE=block_size,
            isCloseVectorization=True,
            buffer_size_limit=2048,
        )
    return input


def _unsafe_masked_index_put_accumulate(input, mask, indices, values):
    logger.debug("GEMS_KUNLUNXIN _UNSAFE_MASKED_INDEX_PUT_ACCUMULATE")
    rank = input.ndim
    if rank < 1 or rank > 3 or len(indices) != rank:
        raise RuntimeError(
            "Kunlunxin _unsafe_masked_index_put_accumulate supports ranks 1 to 3"
        )
    if input.numel() == 0 or mask.numel() == 0:
        return input
    if input.numel() * mask.numel() <= _SCAN_MAX_WORK:
        return _scan_impl(input, mask, indices, values)
    return _pipeline_impl(input, mask, indices, values)
