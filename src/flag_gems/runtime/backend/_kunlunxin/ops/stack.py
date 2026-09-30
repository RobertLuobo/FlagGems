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
from typing import List, Tuple, Union

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)


@triton.jit
def stk_contig_kernel(
    out_ptr, in_ptr, base_offset, total_elements, BLOCK_X: tl.constexpr,
):
    pid = tl.program_id(0)
    idx = pid.to(tl.int64) * BLOCK_X + tl.arange(0, BLOCK_X).to(tl.int64)
    mask = idx < total_elements
    out_idx = base_offset + idx
    tl.store(out_ptr + out_idx, tl.load(in_ptr + idx, mask=mask), mask=mask)


@triton.jit
def stk_flat_kernel(
    out_ptr, in_ptr, block_in, out_row_stride, base_offset,
    total_elements, BLOCK_X: tl.constexpr,
):
    pid = tl.program_id(0)
    idx = pid.to(tl.int64) * BLOCK_X + tl.arange(0, BLOCK_X).to(tl.int64)
    mask = idx < total_elements
    pre_idx = idx // block_in
    within = idx % block_in
    out_idx = pre_idx * out_row_stride + base_offset + within
    tl.store(out_ptr + out_idx, tl.load(in_ptr + idx, mask=mask), mask=mask)


def _pick_block(numel: int) -> int:
    # A large per-program block is what makes a single copy kernel saturate HBM
    # on XPU3 (num_warps is not a tunable knob here): a small BLOCK leaves the
    # 16M-element copy at ~70 GB/s, while BLOCK>=16384 reaches ~360 GB/s.
    target = numel // 128
    b = 4096
    while b < 16384 and b < target:
        b <<= 1
    return b


def stack(
    tensors: Union[Tuple[torch.Tensor, ...], List[torch.Tensor]], dim: int = 0
) -> torch.Tensor:
    logger.debug("GEMS_KUNLUNXIN STACK")

    if len(tensors) == 0:
        raise RuntimeError("stack expected a non-empty TensorList")

    inp_shapes = [list(_.shape) for _ in tensors]
    inp0_shape = inp_shapes[0]
    for i, s in enumerate(inp_shapes[1:]):
        if (dim < -tensors[i + 1].dim() - 1) or (dim > tensors[i + 1].dim()):
            raise IndexError(
                "Dimension out of range (expected to be in range of [{}, {}], but got {})".format(
                    -tensors[i + 1].dim() - 1, tensors[i + 1].dim(), dim
                )
            )
        if s != inp0_shape:
            raise RuntimeError(
                f"stack expects each tensor to be equal size, but got {inp0_shape} at entry 0 and {s} at entry {i + 1}"
            )

    if dim < 0:
        dim = dim + len(inp0_shape) + 1

    dtype = tensors[0].dtype
    for t in tensors[1:]:
        dtype = torch.promote_types(dtype, t.dtype)
    tensors = [t.to(dtype) if t.dtype != dtype else t for t in tensors]

    n = len(tensors)
    out_shape = inp0_shape[:dim] + [n] + inp0_shape[dim:]
    out0 = torch.empty(out_shape, dtype=dtype, device=tensors[0].device)

    # ``post`` is the contiguous run written per input row; ``pre`` how many such
    # runs. stack writes input i into out[..., i, ...]: a per-row contiguous copy
    # with output row stride n*post and per-input base offset i*post.
    post = 1
    for s in inp0_shape[dim:]:
        post *= s
    pre = 1
    for s in inp0_shape[:dim]:
        pre *= s
    out_row_stride = n * post

    for i, a in enumerate(tensors):
        a = a.contiguous()
        total_elements = a.numel()
        if total_elements == 0:
            continue
        base_offset = i * post
        BLOCK = _pick_block(total_elements)
        grid = (triton.cdiv(total_elements, BLOCK),)
        if pre == 1:
            # out[..., i, ...] is a single contiguous run: plain memcpy.
            stk_contig_kernel[grid](
                out0, a, base_offset, total_elements, BLOCK_X=BLOCK,
            )
        else:
            stk_flat_kernel[grid](
                out0, a, post, out_row_stride, base_offset,
                total_elements, BLOCK_X=BLOCK,
            )

    return out0
