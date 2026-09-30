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

from flag_gems.ops.fliplr import fliplr_contiguous_kernel, fliplr_strided_kernel
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)


@libentry()
@triton.jit
def fliplr_block_copy_kernel(
    input,
    output,
    width,
    inner_size,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int64)
    col = pid % width
    row = pid // width
    in_base = (row * width + (width - 1 - col)) * inner_size
    out_base = (row * width + col) * inner_size
    lane = tl.arange(0, BLOCK_SIZE)
    n_chunks = tl.cdiv(inner_size, BLOCK_SIZE)
    for chunk in range(0, n_chunks):
        idx = chunk * BLOCK_SIZE + lane
        mask = idx < inner_size
        values = tl.load(input + in_base + idx, mask=mask)
        tl.store(output + out_base + idx, values, mask=mask)


def fliplr(self: torch.Tensor) -> torch.Tensor:
    logger.debug("GEMS_KUNLUNXIN FLIPLR")

    if self.ndim < 2:
        raise RuntimeError("Input must be >= 2-d.")

    output = torch.empty(self.shape, dtype=self.dtype, device=self.device)
    n_elements = self.numel()
    if n_elements == 0:
        return output

    width = self.shape[1]
    inner_size = math.prod(self.shape[2:])
    with torch_device_fn.device(self.device):
        if self.is_contiguous():
            if inner_size > 1 and width > 1:
                block_size = min(4096, max(1024, triton.next_power_of_2(inner_size)))
                grid = (self.shape[0] * width,)
                fliplr_block_copy_kernel[grid](
                    self,
                    output,
                    width,
                    inner_size,
                    BLOCK_SIZE=block_size,
                )
            else:
                grid = (triton.cdiv(n_elements, 1024),)
                fliplr_contiguous_kernel[grid](
                    self,
                    output,
                    n_elements,
                    width,
                    inner_size,
                    BLOCK_SIZE=1024,
                )
        else:
            grid = (triton.cdiv(n_elements, 1024),)
            fliplr_strided_kernel[grid](
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
