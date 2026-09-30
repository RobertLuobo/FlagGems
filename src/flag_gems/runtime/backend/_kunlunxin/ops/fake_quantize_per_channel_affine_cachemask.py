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

from flag_gems.ops.fake_quantize_per_channel_affine_cachemask import (
    fake_quantize_per_channel_affine_cachemask_kernel,
)
from flag_gems.runtime import torch_device_fn

logger = logging.getLogger(__name__)


def _fake_quantize_per_channel_affine_cachemask_impl(
    input,
    scale,
    zero_point,
    axis,
    quant_min,
    quant_max,
    output=None,
    cachemask=None,
):
    input = input.contiguous()
    scale = scale.contiguous()
    zero_point = zero_point.contiguous()

    if output is None:
        output = torch.empty_like(input)
    if cachemask is None:
        cachemask = torch.empty_like(input, dtype=torch.bool)

    n_elements = input.numel()
    if n_elements == 0:
        return output, cachemask

    n_channels = input.shape[axis]
    channel_stride = 1
    for size in input.shape[axis + 1 :]:
        channel_stride *= size

    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    with torch_device_fn.device(input.device):
        fake_quantize_per_channel_affine_cachemask_kernel[grid](
            input,
            scale,
            zero_point,
            output,
            cachemask,
            n_elements,
            n_channels,
            channel_stride,
            quant_min,
            quant_max,
            BLOCK_SIZE=BLOCK_SIZE,
        )
    return output, cachemask


def fake_quantize_per_channel_affine_cachemask(
    input, scale, zero_point, axis, quant_min, quant_max
):
    logger.debug("GEMS_KUNLUNXIN FAKE_QUANTIZE_PER_CHANNEL_AFFINE_CACHEMASK")
    return _fake_quantize_per_channel_affine_cachemask_impl(
        input, scale, zero_point, axis, quant_min, quant_max
    )


def fake_quantize_per_channel_affine_cachemask_out(
    input, scale, zero_point, axis, quant_min, quant_max, *, out0, out1
):
    logger.debug("GEMS_KUNLUNXIN FAKE_QUANTIZE_PER_CHANNEL_AFFINE_CACHEMASK_OUT")
    return _fake_quantize_per_channel_affine_cachemask_impl(
        input,
        scale,
        zero_point,
        axis,
        quant_min,
        quant_max,
        output=out0,
        cachemask=out1,
    )
