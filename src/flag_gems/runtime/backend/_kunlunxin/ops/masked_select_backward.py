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

from flag_gems.ops.contiguous import contiguous
from flag_gems.ops.copy import copy_
from flag_gems.ops.masked_select_backward import _broadcast_views
from flag_gems.ops.neg import neg
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils.shape_utils import bracket_next_power_of_2

logger = logging.getLogger(__name__)


@libentry()
@triton.jit
def _masked_select_backward_single_pass_kernel(
    grad_ptr,
    mask_ptr,
    out_ptr,
    total_count_ptr,
    grad_numel,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.arange(0, BLOCK_SIZE)
    valid = offsets < n_elements
    selected = tl.load(mask_ptr + offsets, mask=valid, other=0).to(tl.int1)
    grad_offsets = tl.cumsum(selected.to(tl.int32), axis=0) - 1
    tl.store(total_count_ptr, tl.sum(tl.where(valid, selected.to(tl.int32), 0), axis=0))
    source_valid = valid & selected & (grad_offsets < grad_numel)
    values = tl.load(grad_ptr + grad_offsets, mask=source_valid, other=0)
    tl.store(out_ptr + offsets, tl.where(selected, values, 0), mask=valid)


@libentry()
@triton.jit(do_not_specialize=["n_elements", "num_blocks", "num_blocks_per_row"])
def _masked_select_backward_redundant_prefix_kernel(
    grad_ptr,
    mask_ptr,
    out_ptr,
    total_count_ptr,
    grad_numel,
    n_elements,
    num_blocks,
    num_blocks_per_row,
    BLOCK_SIZE: tl.constexpr,
):
    row_id = tl.program_id(0)
    start_block = row_id * num_blocks_per_row
    last_block_id = min(num_blocks - 1, start_block + num_blocks_per_row - 1)
    lane = tl.arange(0, BLOCK_SIZE)

    grad_advance = 0
    prefix_offsets = lane
    for _ in range(0, start_block):
        grad_advance += tl.sum(tl.load(mask_ptr + prefix_offsets).to(tl.int32), axis=0)
        prefix_offsets += BLOCK_SIZE

    offsets = start_block * BLOCK_SIZE + lane
    for _ in range(start_block, last_block_id):
        selected = tl.load(mask_ptr + offsets).to(tl.int1)
        selected_int = selected.to(tl.int32)
        grad_offsets = grad_advance + tl.cumsum(selected_int, axis=0) - 1
        source_valid = selected & (grad_offsets < grad_numel)
        values = tl.load(grad_ptr + grad_offsets, mask=source_valid, other=0)
        tl.store(out_ptr + offsets, tl.where(selected, values, 0))
        grad_advance += tl.sum(selected_int, axis=0)
        offsets += BLOCK_SIZE

    valid = offsets < n_elements
    selected = tl.load(mask_ptr + offsets, mask=valid, other=0).to(tl.int1)
    selected_int = tl.where(valid, selected.to(tl.int32), 0)
    grad_offsets = grad_advance + tl.cumsum(selected_int, axis=0) - 1
    source_valid = valid & selected & (grad_offsets < grad_numel)
    values = tl.load(grad_ptr + grad_offsets, mask=source_valid, other=0)
    tl.store(out_ptr + offsets, tl.where(selected, values, 0), mask=valid)
    if last_block_id == num_blocks - 1:
        tl.store(total_count_ptr, grad_advance + tl.sum(selected_int, axis=0))


@libentry()
@triton.jit(do_not_specialize=["n_elements", "num_blocks", "num_blocks_per_row"])
def _masked_select_backward_count_kernel(
    mask_ptr,
    part_sums_ptr,
    n_elements,
    num_blocks,
    num_blocks_per_row,
    BLOCK_SIZE: tl.constexpr,
):
    row_id = tl.program_id(0)
    start_block = row_id * num_blocks_per_row
    offsets = start_block * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    last_block_id = min(num_blocks - 1, start_block + num_blocks_per_row - 1)

    count = 0
    for _ in range(start_block, last_block_id):
        count += tl.sum(tl.load(mask_ptr + offsets).to(tl.int32), axis=0)
        offsets += BLOCK_SIZE
    valid = offsets < n_elements
    tail = tl.load(mask_ptr + offsets, mask=valid, other=0).to(tl.int32)
    count += tl.sum(tl.where(valid, tail, 0), axis=0)
    tl.store(part_sums_ptr + row_id, count)


@libentry()
@triton.jit(do_not_specialize=["num_programs"])
def _masked_select_backward_prefix_kernel(
    part_sums_ptr,
    total_count_ptr,
    num_programs,
    NP_BLOCK: tl.constexpr,
):
    offsets = tl.arange(0, NP_BLOCK)
    valid = offsets < num_programs
    counts = tl.load(part_sums_ptr + offsets, mask=valid, other=0)
    tl.store(total_count_ptr, tl.sum(tl.where(valid, counts, 0), axis=0))
    prefixes = tl.cumsum(counts, axis=0) - counts
    tl.store(part_sums_ptr + offsets, prefixes, mask=valid)


@libentry()
@triton.jit(do_not_specialize=["n_elements", "num_blocks", "num_blocks_per_row"])
def _masked_select_backward_scatter_kernel(
    grad_ptr,
    mask_ptr,
    part_sums_ptr,
    out_ptr,
    grad_numel,
    n_elements,
    num_blocks,
    num_blocks_per_row,
    BLOCK_SIZE: tl.constexpr,
):
    row_id = tl.program_id(0)
    start_block = row_id * num_blocks_per_row
    offsets = start_block * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    last_block_id = min(num_blocks - 1, start_block + num_blocks_per_row - 1)

    grad_advance = tl.load(part_sums_ptr + row_id)

    for _ in range(start_block, last_block_id):
        selected = tl.load(mask_ptr + offsets).to(tl.int1)
        selected_int = selected.to(tl.int32)
        grad_offsets = grad_advance + tl.cumsum(selected_int, axis=0) - 1
        source_valid = selected & (grad_offsets < grad_numel)
        values = tl.load(grad_ptr + grad_offsets, mask=source_valid, other=0)
        tl.store(out_ptr + offsets, tl.where(selected, values, 0))
        grad_advance += tl.sum(selected_int, axis=0)
        offsets += BLOCK_SIZE

    valid = offsets < n_elements
    selected = tl.load(mask_ptr + offsets, mask=valid, other=0).to(tl.int1)
    selected_int = tl.where(valid, selected.to(tl.int32), 0)
    grad_offsets = grad_advance + tl.cumsum(selected_int, axis=0) - 1
    source_valid = valid & selected & (grad_offsets < grad_numel)
    values = tl.load(grad_ptr + grad_offsets, mask=source_valid, other=0)
    tl.store(
        out_ptr + offsets,
        tl.where(selected, values, 0),
        mask=valid,
    )


def _count_ones(mask, n_elements):
    # XPU3 mis-reduces masked tl.sum (and large single-block tl.sum) when counting
    # mask ones, so compute the population exactly with the dedicated two-stage
    # count/prefix pass over 4096-wide blocks, independent of the scatter kernels.
    block_size = bracket_next_power_of_2(n_elements, 128, 4096)
    num_warps = min(16, block_size // 32)
    sm_count = torch_device_fn.get_device_properties(mask.device).multi_processor_count
    num_blocks = triton.cdiv(n_elements, block_size)
    num_programs = min(num_blocks, sm_count)
    num_blocks_per_row = triton.cdiv(num_blocks, num_programs)
    num_programs = triton.cdiv(num_blocks, num_blocks_per_row)
    scan_block = triton.next_power_of_2(num_programs)
    index_dtype = torch.int32 if n_elements < 2**31 else torch.int64
    with torch_device_fn.device(mask.device):
        part_sums = torch.empty(num_programs, dtype=index_dtype, device=mask.device)
        total_count = torch.empty((), dtype=torch.int64, device=mask.device)
        _masked_select_backward_count_kernel[(num_programs,)](
            mask,
            part_sums,
            n_elements,
            num_blocks,
            num_blocks_per_row,
            BLOCK_SIZE=block_size,
            num_warps=num_warps,
        )
        _masked_select_backward_prefix_kernel[(1,)](
            part_sums,
            total_count,
            num_programs,
            NP_BLOCK=scan_block,
            num_warps=4,
        )
    return total_count.item()


def _masked_select_backward_real(grad, mask, out, validate=True):
    n_elements = out.numel()
    if n_elements == 0:
        return out

    grad = contiguous(grad)
    total_count = torch.empty((), dtype=torch.int64, device=out.device)
    if n_elements <= 32768:
        block_size = triton.next_power_of_2(n_elements)
        num_warps = 4 if block_size < 2048 else min(16, block_size // 256)
        with torch_device_fn.device(out.device):
            _masked_select_backward_single_pass_kernel[(1,)](
                grad,
                mask,
                out,
                total_count,
                grad.numel(),
                n_elements,
                BLOCK_SIZE=block_size,
                num_warps=num_warps,
            )
        if validate and grad.numel() < n_elements:
            if grad.numel() < _count_ones(mask, n_elements):
                raise RuntimeError(
                    "Number of elements of source < number of ones in mask"
                )
        return out

    block_size = bracket_next_power_of_2(n_elements, 128, 4096)
    num_warps = min(16, block_size // 32)
    sm_count = torch_device_fn.get_device_properties(mask.device).multi_processor_count
    num_blocks = triton.cdiv(n_elements, block_size)
    num_programs = min(num_blocks, sm_count)
    num_blocks_per_row = triton.cdiv(num_blocks, num_programs)
    num_programs = triton.cdiv(num_blocks, num_blocks_per_row)

    if n_elements <= 262144:
        with torch_device_fn.device(out.device):
            _masked_select_backward_redundant_prefix_kernel[(num_programs,)](
                grad,
                mask,
                out,
                total_count,
                grad.numel(),
                n_elements,
                num_blocks,
                num_blocks_per_row,
                BLOCK_SIZE=block_size,
                num_warps=num_warps,
            )
        if validate and grad.numel() < n_elements:
            # The fused kernel's inline total_count is unreliable on XPU3 (the
            # masked tail reduction over a mostly-invalid vector mis-reduces), so
            # recompute the mask population with the dedicated count/prefix pass,
            # which is exact. The scatter output above is already correct.
            if grad.numel() < _count_ones(mask, n_elements):
                raise RuntimeError(
                    "Number of elements of source < number of ones in mask"
                )
        return out

    scan_block = triton.next_power_of_2(num_programs)
    index_dtype = torch.int32 if n_elements < 2**31 else torch.int64

    with torch_device_fn.device(out.device):
        part_sums = torch.empty(num_programs, dtype=index_dtype, device=out.device)
        _masked_select_backward_count_kernel[(num_programs,)](
            mask,
            part_sums,
            n_elements,
            num_blocks,
            num_blocks_per_row,
            BLOCK_SIZE=block_size,
            num_warps=num_warps,
        )
        _masked_select_backward_prefix_kernel[(1,)](
            part_sums,
            total_count,
            num_programs,
            NP_BLOCK=scan_block,
            num_warps=4,
        )
        _masked_select_backward_scatter_kernel[(num_programs,)](
            grad,
            mask,
            part_sums,
            out,
            grad.numel(),
            n_elements,
            num_blocks,
            num_blocks_per_row,
            BLOCK_SIZE=block_size,
            num_warps=num_warps,
        )
    if validate and grad.numel() < n_elements and grad.numel() < total_count.item():
        raise RuntimeError("Number of elements of source < number of ones in mask")
    return out


def masked_select_backward(grad, input, mask):
    logger.debug("GEMS_KUNLUNXIN MASKED_SELECT_BACKWARD")

    if mask.dtype != torch.bool:
        raise RuntimeError(
            f"masked_scatter_ only supports boolean masks, but got {mask.dtype}"
        )
    if grad.dtype != input.dtype:
        raise RuntimeError(
            "masked_scatter: expected self and source to have same dtypes but got "
            f"{input.dtype} and {grad.dtype}"
        )
    if grad.device != input.device or mask.device != input.device:
        raise RuntimeError("grad, input, and mask must be on the same device")

    input_expanded, mask_expanded = _broadcast_views(input, mask)
    mask_contiguous = contiguous(mask_expanded)
    # ATen returns a contiguous tensor of the broadcast shape (self.new_zeros
    # then masked_scatter_), so allocate contiguous rather than preserving the
    # possibly non-contiguous layout of the broadcast input view.
    result = torch.empty(
        input_expanded.shape, dtype=input.dtype, device=input.device
    )
    if result.numel() == 0:
        return result

    if input.is_complex():
        physical_grad = grad.conj() if grad.is_conj() else grad
        grad_parts = torch.view_as_real(physical_grad)
        real = torch.empty(
            input_expanded.shape, dtype=grad_parts.dtype, device=grad.device
        )
        imag = torch.empty_like(real)
        _masked_select_backward_real(grad_parts[..., 0], mask_contiguous, real)
        _masked_select_backward_real(
            neg(grad_parts[..., 1]) if grad.is_conj() else grad_parts[..., 1],
            mask_contiguous,
            imag,
            validate=False,
        )
        result_parts = torch.view_as_real(result)
        copy_(result_parts[..., 0], real)
        copy_(result_parts[..., 1], imag)
        return result

    _masked_select_backward_real(grad, mask_contiguous, result)
    return result


