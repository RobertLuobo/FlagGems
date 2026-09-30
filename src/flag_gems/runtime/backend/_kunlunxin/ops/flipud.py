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
import math

import torch
import triton
import triton.language as tl

from flag_gems.ops.flipud import (
    flipud_contiguous_kernel,
    flipud_strided_kernel,
)
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)


@libentry()
@triton.jit
def flipud_row_copy_kernel(
    input,
    output,
    height,
    row_size,
    chunks_per_row,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int64)
    row = pid // chunks_per_row
    chunk = pid % chunks_per_row
    in_base = (height - 1 - row) * row_size
    out_base = row * row_size
    lane = tl.arange(0, BLOCK_SIZE)
    idx = chunk * BLOCK_SIZE + lane
    mask = idx < row_size
    values = tl.load(input + in_base + idx, mask=mask)
    tl.store(output + out_base + idx, values, mask=mask)


def _flipud_impl(self: torch.Tensor) -> torch.Tensor:
    if self.ndim < 1:
        raise RuntimeError("Input must be >= 1-d.")

    if self.is_complex():
        return torch.view_as_complex(_flipud_impl(torch.view_as_real(self)))

    output = torch.empty(self.shape, dtype=self.dtype, device=self.device)
    n_elements = self.numel()
    if n_elements == 0:
        return output

    with torch_device_fn.device(self.device):
        if self.is_contiguous() and output.is_contiguous():
            height = self.shape[0]
            row_size = math.prod(self.shape[1:])
            if row_size > 1 and height > 1:
                block_size = min(4096, max(1024, triton.next_power_of_2(row_size)))
                chunks_per_row = triton.cdiv(row_size, block_size)
                grid = (height * chunks_per_row,)
                flipud_row_copy_kernel[grid](
                    self,
                    output,
                    height,
                    row_size,
                    chunks_per_row,
                    BLOCK_SIZE=block_size,
                )
            else:
                grid = (triton.cdiv(n_elements, 1024),)
                flipud_contiguous_kernel[grid](
                    self,
                    output,
                    n_elements,
                    height,
                    row_size,
                    BLOCK_SIZE=1024,
                )
        else:
            grid = (triton.cdiv(n_elements, 1024),)
            flipud_strided_kernel[grid](
                self,
                output,
                n_elements,
                SHAPE=tuple(self.shape),
                INPUT_STRIDES=tuple(self.stride()),
                OUTPUT_STRIDES=tuple(output.stride()),
                NDIM=self.ndim,
                BLOCK_SIZE=1024,
            )

    return output


class Flipud(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        return _flipud_impl(input)

    @staticmethod
    def backward(ctx, grad_output):
        return _flipud_impl(grad_output)


def flipud(self: torch.Tensor) -> torch.Tensor:
    logger.debug("GEMS_KUNLUNXIN FLIPUD")

    if torch.is_grad_enabled() and self.requires_grad:
        return Flipud.apply(self)
    return _flipud_impl(self)
