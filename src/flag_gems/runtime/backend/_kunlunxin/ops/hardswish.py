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

from flag_gems.ops.hardswish import hardswish_kernel
from flag_gems.runtime import torch_device_fn

logger = logging.getLogger(__name__)


def hardswish(x: torch.Tensor):
    logger.debug("GEMS HARDSWISH")
    x_contig = x.contiguous()
    out = torch.empty_like(x_contig)
    n_elements = out.numel()
    BLOCK_SIZE = 1024
    # Constant-tuple grid: on XPU3 a grid=lambda is baked into the Triton
    # compile-cache key and forces a full recompile on every call.
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    with torch_device_fn.device(x_contig.device):
        hardswish_kernel[grid](x_contig, out, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return out


def hardswish_out(x: torch.Tensor, out: torch.Tensor):
    logger.debug("GEMS HARDSWISH_OUT")
    assert x.numel() == out.numel()
    x_contig = x.contiguous()
    n_elements = out.numel()
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    with torch_device_fn.device(x_contig.device):
        hardswish_kernel[grid](x_contig, out, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return out
