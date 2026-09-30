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

from flag_gems.ops.fake_quantize_per_tensor_affine_cachemask_backward import (
    fake_quantize_per_tensor_affine_cachemask_backward_kernel,
)
from flag_gems.runtime import torch_device_fn

logger = logging.getLogger(__name__)

_BLOCK_SIZE = 8192


def fake_quantize_per_tensor_affine_cachemask_backward(grad, mask):
    logger.debug("GEMS_KUNLUNXIN FAKE_QUANTIZE_PER_TENSOR_AFFINE_CACHEMASK_BACKWARD")
    grad = grad.contiguous()
    mask = mask.contiguous()
    output = torch.empty_like(grad)
    n_elements = grad.numel()
    if n_elements == 0:
        return output

    block_size = _BLOCK_SIZE
    grid = (triton.cdiv(n_elements, block_size),)
    with torch_device_fn.device(grad.device):
        fake_quantize_per_tensor_affine_cachemask_backward_kernel[grid](
            grad, mask, output, n_elements, BLOCK_SIZE=block_size
        )
    return output
