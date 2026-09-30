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
    _round_half_to_even,
    unpack_linear_weight,
)
from flag_gems.ops._wrapped_quantized_linear_prepacked import (
    _wrapped_quantized_linear_prepacked_empty_k_kernel,
)
from flag_gems.runtime import torch_device_fn

logger = logging.getLogger(__name__)


@triton.jit
def _center_input_kernel(
    input,
    input_scale,
    input_zero_point,
    centered,
    M,
    K,
    stride_im,
    stride_ik,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < M * K
    rows = offsets // K
    cols = offsets % K
    scale = tl.load(input_scale).to(tl.float32)
    zero_point = tl.load(input_zero_point).to(tl.float32)
    values = tl.load(
        input + rows * stride_im + cols * stride_ik, mask=mask, other=0.0
    ).to(tl.float32)
    quantized = _round_half_to_even(values / scale) + zero_point
    quantized = tl.minimum(tl.maximum(quantized, 0.0), 255.0)
    tl.store(centered + offsets, quantized - zero_point, mask=mask)


@triton.jit
def _center_weight_kernel(
    weight,
    weight_metadata,
    centered,
    N,
    K,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < N * K
    n = offsets // K
    k = offsets % K
    zero_point = tl.load(weight_metadata + 1).to(tl.float32)
    values = tl.load(weight + n * K + k, mask=mask, other=0).to(tl.float32)
    # Store transposed to (K, N) so the paired mm consumes it as the rhs.
    tl.store(centered + k * N + n, values - zero_point, mask=mask)


@triton.jit
def _epilogue_kernel(
    accumulator,
    weight_metadata,
    bias,
    input_scale,
    output_scale,
    output_zero_point,
    output,
    M,
    N,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < M * N
    n = offsets % N
    input_scale_value = tl.load(input_scale).to(tl.float32)
    weight_scale_value = tl.load(weight_metadata).to(tl.float32)
    output_scale_value = tl.load(output_scale).to(tl.float32)
    output_zero_point_value = tl.load(output_zero_point).to(tl.float32)
    acc = tl.load(accumulator + offsets, mask=mask, other=0.0).to(tl.float32)
    bias_values = tl.load(bias + n, mask=mask, other=0.0).to(tl.float32)
    real_output = acc * input_scale_value * weight_scale_value + bias_values
    quantized_output = (
        _round_half_to_even(real_output / output_scale_value)
        + output_zero_point_value
    )
    quantized_output = tl.minimum(tl.maximum(quantized_output, 0.0), 255.0)
    dequantized_output = (
        quantized_output - output_zero_point_value
    ) * output_scale_value
    tl.store(output + offsets, dequantized_output, mask=mask)


def _wrapped_quantized_linear_prepacked(
    input,
    input_scale,
    input_zero_point,
    packed_weight,
    output_scale,
    output_zero_point,
    out_channel,
):
    logger.debug("GEMS_KUNLUNXIN _WRAPPED_QUANTIZED_LINEAR_PREPACKED")
    if input.dtype != torch.float32:
        raise RuntimeError(f"Quantize only works on Float Tensor, got {input.dtype}")
    if input.ndim < 2:
        raise RuntimeError(
            "The dimension of input tensor should be larger than or equal to 2"
        )
    if out_channel < 0:
        raise RuntimeError("out_channel must be non-negative")
    for name, parameter in (
        ("input_scale", input_scale),
        ("input_zero_point", input_zero_point),
        ("output_scale", output_scale),
        ("output_zero_point", output_zero_point),
    ):
        if parameter.numel() != 1:
            raise RuntimeError(f"{name} must contain one element")
        if parameter.device != input.device:
            raise RuntimeError(f"{name} must be on the input device")
    if packed_weight.device != input.device:
        raise RuntimeError("packed_weight must be on the input device")

    import flag_gems

    K = input.shape[-1]
    quantized_weight, weight_metadata, bias = unpack_linear_weight(
        packed_weight, out_channel, K
    )

    output_shape = (*input.shape[:-1], out_channel)
    output = torch.empty(output_shape, dtype=torch.float32, device=input.device)
    M = 1
    for dimension in input.shape[:-1]:
        M *= dimension
    if M == 0 or out_channel == 0:
        return output

    if K == 0:
        block_size = 256
        grid = (triton.cdiv(output.numel(), block_size),)
        with torch_device_fn.device(input.device):
            _wrapped_quantized_linear_prepacked_empty_k_kernel[grid](
                output,
                output_scale,
                output_zero_point,
                output.numel(),
                BLOCK_SIZE=block_size,
            )
        return output

    input_2d = input.reshape(M, K)
    N = out_channel

    # XPU3 miscompiles a masked 2D load feeding tl.dot (int8 GEMM produces an
    # illegal-address / TritonSDNNLegalize failure), so we center inputs and
    # weights to fp32 and route the GEMM through the backend fp32 mm. The
    # quantized operands are small exact integers, so this matches the native
    # integer accumulation used by the reference implementation.
    centered_input = torch.empty((M, K), dtype=torch.float32, device=input.device)
    centered_weight = torch.empty((K, N), dtype=torch.float32, device=input.device)
    block = 1024
    with torch_device_fn.device(input.device):
        _center_input_kernel[(triton.cdiv(M * K, block),)](
            input_2d,
            input_scale,
            input_zero_point,
            centered_input,
            M,
            K,
            input_2d.stride(0),
            input_2d.stride(1),
            BLOCK=block,
        )
        _center_weight_kernel[(triton.cdiv(N * K, block),)](
            quantized_weight,
            weight_metadata,
            centered_weight,
            N,
            K,
            BLOCK=block,
        )

    accumulator = flag_gems.mm(centered_input, centered_weight)

    with torch_device_fn.device(input.device):
        _epilogue_kernel[(triton.cdiv(M * N, block),)](
            accumulator,
            weight_metadata,
            bias,
            input_scale,
            output_scale,
            output_zero_point,
            output,
            M,
            N,
            BLOCK=block,
        )
    return output
