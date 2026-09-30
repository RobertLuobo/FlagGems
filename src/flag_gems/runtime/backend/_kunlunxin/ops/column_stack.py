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
from typing import List, Optional, Tuple, Union

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)

_BLOCK2D_MIN = 16


@triton.jit
def cs_flat_kernel(out_ptr, in_ptr, block_in, out_row_stride, base_offset,
                   total_elements, BLOCK_X: tl.constexpr):
    pid = tl.program_id(0)
    idx = pid * BLOCK_X + tl.arange(0, BLOCK_X)
    mask = idx < total_elements
    pre_idx = idx // block_in
    within = idx % block_in
    out_idx = pre_idx * out_row_stride + base_offset + within
    tl.store(out_ptr + out_idx, tl.load(in_ptr + idx, mask=mask), mask=mask)


@triton.jit
def cs_kernel2d(out_ptr, in_ptr, block_in, out_row_stride, base_offset,
                BLOCK_X: tl.constexpr):
    p0 = tl.program_id(0)
    p1 = tl.program_id(1)
    off = p1 * BLOCK_X + tl.arange(0, BLOCK_X)
    mask = off < block_in
    in_idx = p0 * block_in + off
    out_idx = p0 * out_row_stride + base_offset + off
    tl.store(out_ptr + out_idx, tl.load(in_ptr + in_idx, mask=mask), mask=mask)


def _reshape_input(t: torch.Tensor) -> torch.Tensor:
    if t.ndim <= 1:
        return t.reshape(t.numel(), 1)
    return t


def _pick_block(n: int) -> int:
    b = 1
    while b < n and b < 2048:
        b <<= 1
    if b < 256:
        b = 256
    return b


def column_stack(
    tensors: Union[Tuple[torch.Tensor, ...], List[torch.Tensor]],
    *,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    logger.debug("GEMS_KUNLUNXIN COLUMN_STACK")

    if len(tensors) == 0:
        raise RuntimeError("column_stack expected a non-empty TensorList")

    reshaped = [_reshape_input(t) for t in tensors]
    inp0_shape = list(reshaped[0].shape)
    dim = 1

    for tensor_num, tensor in enumerate(reshaped[1:]):
        if tensor.ndim != reshaped[0].ndim:
            raise RuntimeError(
                f"Tensors must have same number of dimensions: got "
                f"{reshaped[0].ndim} and {tensor.ndim}"
            )
        inp_shape = list(tensor.shape)
        for i in range(len(inp_shape)):
            if i != dim and inp_shape[i] != inp0_shape[i]:
                raise RuntimeError(
                    f"Sizes of tensors must match except in dimension {dim}. "
                    f"Expected size {inp0_shape[i]} but got size {inp_shape[i]} "
                    f"for tensor number {tensor_num + 1} in the list."
                )

    dtype = reshaped[0].dtype
    for t in reshaped[1:]:
        dtype = torch.promote_types(dtype, t.dtype)
    reshaped = [t.to(dtype) if t.dtype != dtype else t for t in reshaped]
    device = reshaped[0].device

    out_shape = list(inp0_shape)
    out_shape[dim] = sum(int(t.shape[dim]) for t in reshaped)

    if out is None:
        out = torch.empty(out_shape, dtype=dtype, device=device)
    else:
        out = out.view(out_shape)

    dim_prod_post = 1
    for s in inp0_shape[dim + 1:]:
        dim_prod_post *= s

    out_row_stride = out_shape[dim] * dim_prod_post
    dim_offset = 0
    for tensor in reshaped:
        tensor = tensor.contiguous()
        total_elements = tensor.numel()
        dim_size_in = int(tensor.shape[dim])
        block_in = dim_size_in * dim_prod_post
        base_offset = dim_offset * dim_prod_post
        if total_elements > 0:
            if block_in >= _BLOCK2D_MIN:
                pre = total_elements // block_in
                BLOCK = _pick_block(block_in)
                grid = (pre, triton.cdiv(block_in, BLOCK))
                cs_kernel2d[grid](
                    out, tensor, block_in, out_row_stride, base_offset,
                    BLOCK_X=BLOCK,
                )
            else:
                BLOCK = 1024
                grid = (triton.cdiv(total_elements, BLOCK),)
                cs_flat_kernel[grid](
                    out, tensor, block_in, out_row_stride, base_offset,
                    total_elements, BLOCK_X=BLOCK,
                )
        dim_offset += dim_size_in

    return out


def column_stack_out(
    tensors: Union[Tuple[torch.Tensor, ...], List[torch.Tensor]],
    *,
    out: torch.Tensor,
) -> torch.Tensor:
    logger.debug("GEMS_KUNLUNXIN COLUMN_STACK_OUT")
    return column_stack(tensors, out=out)
