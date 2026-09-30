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

logger = logging.getLogger(__name__)

BLOCK_SIZE = 1024


@triton.jit
def _iand_tensor_kernel(a_ptr, b_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    a = tl.load(a_ptr + offsets, mask=mask)
    b = tl.load(b_ptr + offsets, mask=mask)
    tl.store(a_ptr + offsets, a & b, mask=mask)


@triton.jit
def _iand_scalar_kernel(a_ptr, scalar, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    a = tl.load(a_ptr + offsets, mask=mask)
    tl.store(a_ptr + offsets, a & scalar, mask=mask)


def __iand___tensor(self, other):
    logger.debug("GEMS_KUNLUNXIN IAND_TENSOR")
    if not isinstance(other, torch.Tensor):
        return __iand___scalar(self, other)

    a = self if self.is_contiguous() else self.contiguous()

    # In-place bitwise-and writes back into self with self's dtype; broadcast
    # and dtype-align other the way the generic DEFAULT-promotion path does.
    if other.dtype != a.dtype:
        other = other.to(a.dtype)
    if other.shape != a.shape:
        other = other.broadcast_to(a.shape)
    b = other if other.is_contiguous() else other.contiguous()

    n_elements = a.numel()
    if n_elements == 0:
        if a is not self:
            self.copy_(a)
        return self

    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    with torch_device_fn.device(a.device):
        _iand_tensor_kernel[grid](a, b, n_elements, BLOCK_SIZE=BLOCK_SIZE)

    if a is not self:
        self.copy_(a)
    return self


def __iand___scalar(self, other):
    logger.debug("GEMS_KUNLUNXIN IAND_SCALAR")
    a = self if self.is_contiguous() else self.contiguous()

    scalar = int(other)

    n_elements = a.numel()
    if n_elements == 0:
        if a is not self:
            self.copy_(a)
        return self

    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    with torch_device_fn.device(a.device):
        _iand_scalar_kernel[grid](a, scalar, n_elements, BLOCK_SIZE=BLOCK_SIZE)

    if a is not self:
        self.copy_(a)
    return self
