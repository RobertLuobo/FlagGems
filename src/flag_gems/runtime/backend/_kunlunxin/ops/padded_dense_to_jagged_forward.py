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

from flag_gems.ops.padded_dense_to_jagged_forward import (
    _MAX_GRID_SIZE,
    _MAX_JAGGED_DIMS,
    _check_inputs,
    _walk_up_offset_tree,
)
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import triton_lang_extension as tle

logger = logging.getLogger(__name__)

_BLOCK_SIZE = 256


@triton.jit
def _single_kernel(
    dense,
    offsets,
    output,
    total_tasks,
    chunks_per_batch,
    max_length,
    inner_size,
    total_L,
    scratch_base,
    USE_INT64_INDEX: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    worker = tle.program_id(0)
    worker_count = tle.num_programs(0)

    for raw_task in tl.range(worker, total_tasks, worker_count):
        if USE_INT64_INDEX:
            task = raw_task.to(tl.int64)
        else:
            task = raw_task.to(tl.int32)
        batch_idx = task // chunks_per_batch
        chunk_idx = task - batch_idx * chunks_per_batch
        sequence_start = tl.load(offsets + batch_idx)
        sequence_end = tl.load(offsets + batch_idx + 1)
        sequence_elements = (sequence_end - sequence_start) * inner_size
        sequence_valid = (
            (sequence_start >= 0)
            & (sequence_end >= sequence_start)
            & (sequence_end <= total_L)
            & (sequence_end - sequence_start <= max_length)
        )

        lane = tl.arange(0, BLOCK_SIZE)
        local_offsets = chunk_idx * BLOCK_SIZE + lane
        mask = (local_offsets < sequence_elements) & sequence_valid
        src = tl.where(
            mask, batch_idx * max_length * inner_size + local_offsets, 0
        )
        dst = tl.where(
            mask, sequence_start * inner_size + local_offsets, scratch_base + lane
        )
        values = tl.load(dense + src, mask=mask, other=0)
        tl.store(output + dst, values, mask=mask)


@triton.jit
def _multi_kernel(
    dense,
    offsets_0,
    offsets_1,
    offsets_2,
    offsets_3,
    offsets_4,
    output,
    deepest_parent_count,
    chunks_per_sequence,
    inner_size,
    prefix_padded_volume,
    final_max_length,
    total_L,
    scratch_base,
    max_length_0,
    max_length_1,
    max_length_2,
    max_length_3,
    max_length_4,
    NUM_JAGGED_DIMS: tl.constexpr,
    OFFSET_SIZE_0: tl.constexpr,
    OFFSET_SIZE_1: tl.constexpr,
    OFFSET_SIZE_2: tl.constexpr,
    OFFSET_SIZE_3: tl.constexpr,
    OFFSET_SIZE_4: tl.constexpr,
    LOG_OFFSET_SIZE_0: tl.constexpr,
    LOG_OFFSET_SIZE_1: tl.constexpr,
    LOG_OFFSET_SIZE_2: tl.constexpr,
    LOG_OFFSET_SIZE_3: tl.constexpr,
    LOG_OFFSET_SIZE_4: tl.constexpr,
    USE_INT64_INDEX: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    worker = tle.program_id(0)
    worker_count = tle.num_programs(0)

    for raw_sequence_idx in tl.range(worker, deepest_parent_count, worker_count):
        if USE_INT64_INDEX:
            sequence_idx = raw_sequence_idx.to(tl.int64)
        else:
            sequence_idx = raw_sequence_idx.to(tl.int32)
        tree_offset = sequence_idx
        jagged_index = tree_offset * 0
        jagged_stride = tree_offset * 0 + 1
        tree_valid = tree_offset >= 0

        if NUM_JAGGED_DIMS == 5:
            sequence_start = tl.load(offsets_4 + sequence_idx)
            sequence_end = tl.load(offsets_4 + sequence_idx + 1)
        elif NUM_JAGGED_DIMS == 4:
            sequence_start = tl.load(offsets_3 + sequence_idx)
            sequence_end = tl.load(offsets_3 + sequence_idx + 1)
        elif NUM_JAGGED_DIMS == 3:
            sequence_start = tl.load(offsets_2 + sequence_idx)
            sequence_end = tl.load(offsets_2 + sequence_idx + 1)
        else:
            sequence_start = tl.load(offsets_1 + sequence_idx)
            sequence_end = tl.load(offsets_1 + sequence_idx + 1)

        sequence_length = sequence_end - sequence_start
        sequence_valid = (
            (sequence_start >= 0)
            & (sequence_end >= sequence_start)
            & (sequence_end <= total_L)
            & (sequence_length <= final_max_length)
        )

        if NUM_JAGGED_DIMS >= 5:
            parent, coordinate, level_valid = _walk_up_offset_tree(
                offsets_3, tree_offset, max_length_3,
                OFFSET_SIZE_3, LOG_OFFSET_SIZE_3, USE_INT64_INDEX,
            )
            tree_valid = tree_valid & level_valid
            jagged_index += coordinate * jagged_stride
            jagged_stride *= max_length_3
            tree_offset = parent

        if NUM_JAGGED_DIMS >= 4:
            parent, coordinate, level_valid = _walk_up_offset_tree(
                offsets_2, tree_offset, max_length_2,
                OFFSET_SIZE_2, LOG_OFFSET_SIZE_2, USE_INT64_INDEX,
            )
            tree_valid = tree_valid & level_valid
            jagged_index += coordinate * jagged_stride
            jagged_stride *= max_length_2
            tree_offset = parent

        if NUM_JAGGED_DIMS >= 3:
            parent, coordinate, level_valid = _walk_up_offset_tree(
                offsets_1, tree_offset, max_length_1,
                OFFSET_SIZE_1, LOG_OFFSET_SIZE_1, USE_INT64_INDEX,
            )
            tree_valid = tree_valid & level_valid
            jagged_index += coordinate * jagged_stride
            jagged_stride *= max_length_1
            tree_offset = parent

        parent, coordinate, level_valid = _walk_up_offset_tree(
            offsets_0, tree_offset, max_length_0,
            OFFSET_SIZE_0, LOG_OFFSET_SIZE_0, USE_INT64_INDEX,
        )
        tree_valid = tree_valid & level_valid
        jagged_index += coordinate * jagged_stride
        batch_idx = parent

        sequence_elements = sequence_length * inner_size
        prefix_dense_row = batch_idx * prefix_padded_volume + jagged_index
        dense_sequence_start = prefix_dense_row * final_max_length * inner_size
        output_sequence_start = sequence_start * inner_size
        for raw_chunk_idx in tl.range(0, chunks_per_sequence):
            if USE_INT64_INDEX:
                chunk_idx = raw_chunk_idx.to(tl.int64)
            else:
                chunk_idx = raw_chunk_idx.to(tl.int32)
            lane = tl.arange(0, BLOCK_SIZE)
            local_offsets = chunk_idx * BLOCK_SIZE + lane
            mask = (local_offsets < sequence_elements) & sequence_valid & tree_valid
            src = tl.where(mask, dense_sequence_start + local_offsets, 0)
            dst = tl.where(
                mask, output_sequence_start + local_offsets, scratch_base + lane
            )
            values = tl.load(dense + src, mask=mask, other=0)
            tl.store(output + dst, values, mask=mask)


_VIEW_INT_DTYPE = {
    torch.float16: torch.int16,
    torch.bfloat16: torch.int16,
    torch.float32: torch.int32,
    torch.float64: torch.int64,
    torch.int64: torch.int64,
}


def _launch_single(dense, offset, storage, scratch_base, total_L, max_length, inner_size):
    if total_L == 0 or max_length == 0 or inner_size == 0:
        return
    chunks_per_batch = triton.cdiv(max_length * inner_size, _BLOCK_SIZE)
    batch_size = offset.numel() - 1
    total_tasks = batch_size * chunks_per_batch
    if total_tasks == 0:
        return
    grid = (min(total_tasks, _MAX_GRID_SIZE),)
    use_int64_index = (storage.numel()) >= 2**31
    _single_kernel[grid](
        dense,
        offset,
        storage,
        total_tasks,
        chunks_per_batch,
        max_length,
        inner_size,
        total_L,
        scratch_base,
        USE_INT64_INDEX=use_int64_index,
        BLOCK_SIZE=_BLOCK_SIZE,
    )


def _launch_multi(
    dense, offsets, storage, scratch_base, total_L, max_lengths, inner_size
):
    deepest_parent_count = offsets[-1].numel() - 1
    if total_L == 0 or deepest_parent_count == 0:
        return
    chunks_per_sequence = triton.cdiv(max_lengths[-1] * inner_size, _BLOCK_SIZE)
    if chunks_per_sequence == 0:
        return
    grid = (min(deepest_parent_count, _MAX_GRID_SIZE),)

    padded_offsets = list(offsets) + [offsets[-1]] * (_MAX_JAGGED_DIMS - len(offsets))
    padded_lengths = list(max_lengths) + [1] * (_MAX_JAGGED_DIMS - len(max_lengths))
    offset_sizes = [o.numel() for o in padded_offsets]
    offset_logs = [s.bit_length() for s in offset_sizes]
    prefix_padded_volume = 1
    for length in max_lengths[:-1]:
        prefix_padded_volume *= length

    use_int64_index = (storage.numel()) >= 2**31
    _multi_kernel[grid](
        dense,
        *padded_offsets,
        storage,
        deepest_parent_count,
        chunks_per_sequence,
        inner_size,
        prefix_padded_volume,
        max_lengths[-1],
        total_L,
        scratch_base,
        *padded_lengths,
        NUM_JAGGED_DIMS=len(offsets),
        OFFSET_SIZE_0=offset_sizes[0],
        OFFSET_SIZE_1=offset_sizes[1],
        OFFSET_SIZE_2=offset_sizes[2],
        OFFSET_SIZE_3=offset_sizes[3],
        OFFSET_SIZE_4=offset_sizes[4],
        LOG_OFFSET_SIZE_0=offset_logs[0],
        LOG_OFFSET_SIZE_1=offset_logs[1],
        LOG_OFFSET_SIZE_2=offset_logs[2],
        LOG_OFFSET_SIZE_3=offset_logs[3],
        LOG_OFFSET_SIZE_4=offset_logs[4],
        USE_INT64_INDEX=use_int64_index,
        BLOCK_SIZE=_BLOCK_SIZE,
    )


def _padded_dense_to_jagged_forward(dense, offsets, total_L=None):
    logger.debug("GEMS_KUNLUNXIN PADDED DENSE TO JAGGED FORWARD")

    num_jagged_dims, max_lengths, total_L = _check_inputs(dense, offsets, total_L)
    inner_size = dense.size(-1)

    if total_L == 0 or inner_size == 0:
        return torch.empty(
            (total_L, inner_size), dtype=dense.dtype, device=dense.device
        )

    dense_contiguous = dense.contiguous()
    offsets_contiguous = [offset.contiguous() for offset in offsets]

    # Copy bits through a same-width integer view with a BLOCK_SIZE scratch tail.
    int_dtype = _VIEW_INT_DTYPE[dense.dtype]
    scratch_base = total_L * inner_size
    storage = torch.empty(
        scratch_base + _BLOCK_SIZE, dtype=dense.dtype, device=dense.device
    )
    output = storage[:scratch_base].view(total_L, inner_size)

    kernel_storage = storage.view(int_dtype)
    kernel_dense = dense_contiguous.view(int_dtype)

    with torch_device_fn.device(dense.device):
        if num_jagged_dims == 1:
            _launch_single(
                kernel_dense,
                offsets_contiguous[0],
                kernel_storage,
                scratch_base,
                total_L,
                max_lengths[0],
                inner_size,
            )
        else:
            _launch_multi(
                kernel_dense,
                offsets_contiguous,
                kernel_storage,
                scratch_base,
                total_L,
                max_lengths,
                inner_size,
            )
    return output
