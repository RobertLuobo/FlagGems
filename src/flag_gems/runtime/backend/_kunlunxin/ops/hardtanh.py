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

from flag_gems.ops.hardtanh import hardtanh_kernel
from flag_gems.runtime import torch_device_fn

logger = logging.getLogger("flag_gems.ops.hardtanh")

BLOCK_SIZE = 1024


def hardtanh(A, min_val=-1.0, max_val=1.0):
    logger.debug("GEMS_KUNLUNXIN HARDTANH")
    if not A.is_contiguous():
        A = A.contiguous()
    out = torch.empty_like(A)
    n_elements = A.numel()
    if n_elements == 0:
        return out
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    with torch_device_fn.device(A.device):
        hardtanh_kernel[grid](
            A, out, n_elements, float(min_val), float(max_val), BLOCK_SIZE=BLOCK_SIZE
        )
    return out


def hardtanh_out(A, min_val=-1.0, max_val=1.0, *, out=None):
    logger.debug("GEMS_KUNLUNXIN HARDTANH_OUT")
    if out is None:
        return hardtanh(A, min_val, max_val)
    if not A.is_contiguous():
        A = A.contiguous()
    n_elements = A.numel()
    if n_elements == 0:
        return out
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    with torch_device_fn.device(A.device):
        hardtanh_kernel[grid](
            A, out, n_elements, float(min_val), float(max_val), BLOCK_SIZE=BLOCK_SIZE
        )
    return out
