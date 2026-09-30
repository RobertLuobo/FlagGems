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

from flag_gems.ops.fake_quantize_per_channel_affine import _round_half_to_even
from flag_gems.runtime import torch_device_fn

logger = logging.getLogger(__name__)


@triton.jit
def _fake_quantize_learnable_per_channel_affine_kernel(
    input_ptr,
    scale_ptr,
    zero_point_ptr,
    output_ptr,
    n_elements,
    n_channels,
    channel_stride,
    quant_min,
    quant_max,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    channel_idx = (offsets // channel_stride) % n_channels
    scale = tl.load(scale_ptr + channel_idx, mask=mask, other=1.0)
    zero_point = tl.load(zero_point_ptr + channel_idx, mask=mask, other=0.0)

    x_fp32 = x.to(tl.float32)
    scale_fp32 = scale.to(tl.float32)
    # Learnable zero-point is a float parameter: round-half-to-even then clamp
    # to the integer quant range, mirroring native _get_rounded_zero_point.
    zp_rounded = _round_half_to_even(zero_point.to(tl.float32))
    zp_fp32 = tl.minimum(tl.maximum(zp_rounded, quant_min), quant_max)

    # Match native: qval = nearbyint(x * (1.0f / scale)) + zp. Using the fp32
    # reciprocal-multiply (not a true x/scale division) avoids one-quant-level
    # round-half-to-even flips at exact half-way quotients on large tensors.
    inv_scale = 1.0 / scale_fp32
    x_quantized = _round_half_to_even(x_fp32 * inv_scale) + zp_fp32
    x_clamped = tl.minimum(tl.maximum(x_quantized, quant_min), quant_max)
    output = (x_clamped - zp_fp32) * scale_fp32

    tl.store(output_ptr + offsets, output.to(x.dtype), mask=mask)


def _fake_quantize_learnable_per_channel_affine(
    self, scale, zero_point, axis, quant_min, quant_max, grad_factor=1.0
):
    logger.debug("GEMS_KUNLUNXIN _FAKE_QUANTIZE_LEARNABLE_PER_CHANNEL_AFFINE")

    if self.dtype not in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
        raise TypeError(
            f"unsupported dtype {self.dtype}; only floating-point dtypes are supported"
        )

    if axis < 0:
        axis = self.dim() + axis

    input = self.contiguous()
    scale = scale.contiguous()
    zero_point = zero_point.contiguous()

    output = torch.empty_like(input)

    n_elements = input.numel()
    if n_elements == 0:
        return output

    shape = input.shape
    n_channels = shape[axis]

    channel_stride = 1
    for i in range(axis + 1, len(shape)):
        channel_stride *= shape[i]

    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)

    with torch_device_fn.device(input.device):
        _fake_quantize_learnable_per_channel_affine_kernel[grid](
            input,
            scale,
            zero_point,
            output,
            n_elements,
            n_channels,
            channel_stride,
            quant_min,
            quant_max,
            BLOCK_SIZE=BLOCK_SIZE,
        )

    return output
