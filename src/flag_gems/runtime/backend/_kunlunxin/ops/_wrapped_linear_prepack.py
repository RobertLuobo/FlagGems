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

from flag_gems.ops._wrapped_linear_prepack import (
    _HEADER_BYTES,
    _PACK_MAGIC,
    _PACK_VERSION,
    _wrapped_linear_prepack_kernel,
)
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)


@libentry()
@triton.jit
def _wrapped_linear_prepack_empty_kernel(
    weight_scale,
    weight_zero_point,
    bias,
    metadata,
    packed_bias,
    N,
    stride_bias,
    PACK_MAGIC: tl.constexpr,
    PACK_VERSION: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    scale = tl.load(weight_scale).to(tl.float32)
    zero_point = tl.load(weight_zero_point).to(tl.float32)
    metadata_values = tl.where(
        offsets == 0,
        scale,
        tl.where(
            offsets == 1,
            zero_point,
            tl.where(offsets == 2, PACK_MAGIC, PACK_VERSION),
        ),
    )
    tl.store(metadata + offsets, metadata_values, mask=offsets < 4)

    bias_mask = offsets < N
    bias_values = tl.load(bias + offsets * stride_bias, mask=bias_mask, other=0.0).to(
        tl.float32
    )
    tl.store(packed_bias + offsets, bias_values, mask=bias_mask)


def _wrapped_linear_prepack(weight, weight_scale, weight_zero_point, bias):
    logger.debug("GEMS_KUNLUNXIN _WRAPPED_LINEAR_PREPACK")
    if weight.dtype != torch.float32:
        raise RuntimeError(f"Quantize only works on Float Tensor, got {weight.dtype}")
    if weight.ndim != 2:
        raise RuntimeError("fbgemm weight packing only packs matrices not vectors.")
    N, K = weight.shape
    if weight_scale.numel() != 1 or weight_zero_point.numel() != 1:
        raise RuntimeError("weight scale and zero point must contain one element")
    if bias.dtype != torch.float32 or bias.ndim != 1 or bias.numel() != N:
        raise RuntimeError("bias must be a float32 vector with out_channel elements")
    if not (
        weight.device == weight_scale.device == weight_zero_point.device == bias.device
    ):
        raise RuntimeError("all prepack inputs must be on the same device")

    packed_numel = _HEADER_BYTES + 4 * N + N * K
    packed = torch.empty(packed_numel, dtype=torch.uint8, device=weight.device)
    metadata = packed[:_HEADER_BYTES].view(torch.float32)
    bias_end = _HEADER_BYTES + 4 * N
    packed_bias = packed[_HEADER_BYTES:bias_end].view(torch.float32)
    quantized_weight = packed[bias_end:].view(torch.int8)

    block_size = 1024
    with torch_device_fn.device(weight.device):
        if N == 0 or K == 0:
            grid = (triton.cdiv(max(N, 4), block_size),)
            _wrapped_linear_prepack_empty_kernel[grid](
                weight_scale,
                weight_zero_point,
                bias,
                metadata,
                packed_bias,
                N,
                bias.stride(0),
                PACK_MAGIC=_PACK_MAGIC,
                PACK_VERSION=_PACK_VERSION,
                BLOCK_SIZE=block_size,
            )
        else:
            grid = (triton.cdiv(max(N * K, N), block_size),)
            _wrapped_linear_prepack_kernel[grid](
                weight,
                weight_scale,
                weight_zero_point,
                bias,
                metadata,
                packed_bias,
                quantized_weight,
                N,
                K,
                weight.stride(0),
                weight.stride(1),
                bias.stride(0),
                PACK_MAGIC=_PACK_MAGIC,
                PACK_VERSION=_PACK_VERSION,
                BLOCK_SIZE=block_size,
            )
    return packed
