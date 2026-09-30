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
from functools import lru_cache
from typing import Optional, Sequence

import numpy as np
import torch
import triton
import triton.language as tl

from flag_gems.runtime import device, torch_device_fn
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1024)
def _boundary_indices(input_w, output_w, scale):
    ratio = (
        np.float32(input_w / output_w)
        if scale is None or scale <= 0.0
        else np.float32(1.0 / scale)
    )

    def src(o):
        return min(int((np.float32(o) + np.float32(0.5)) * ratio), input_w - 1)

    bounds = [0] * (input_w + 1)
    o = 0
    for i in range(input_w + 1):
        while o < output_w and src(o) < i:
            o += 1
        bounds[i] = o
    return tuple(bounds)


_boundary_tensor_cache = {}


def _boundary_tensor(input_w, output_w, scale, device_):
    bounds = _boundary_indices(input_w, output_w, scale)
    key = (bounds, device_.type, device_.index)
    tensor = _boundary_tensor_cache.get(key)
    if tensor is None:
        tensor = torch.tensor(bounds, dtype=torch.int32, device=device_)
        _boundary_tensor_cache[key] = tensor
    return tensor


@libentry()
@triton.jit
def _upsample_nearest_exact1d_backward_kernel(
    grad_output,
    grad_input,
    WB,
    numel,
    channels,
    input_w,
    grad_output_stride_n,
    grad_output_stride_c,
    grad_output_stride_w,
    grad_input_stride_n,
    grad_input_stride_c,
    grad_input_stride_w,
    IS_FP64: tl.constexpr,
    IS_UINT8: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < numel
    offsets = tl.where(mask, offsets, 0)
    input_x = offsets % input_w
    nc = offsets // input_w
    channel = nc % channels
    batch = nc // channels

    w0 = tl.load(WB + input_x, mask, other=0)
    w1 = tl.load(WB + input_x + 1, mask, other=0)

    if IS_UINT8:
        accumulator = tl.zeros((BLOCK_SIZE,), dtype=tl.int32)
    elif IS_FP64:
        accumulator = tl.zeros((BLOCK_SIZE,), dtype=tl.float64)
    else:
        accumulator = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)

    output_base = batch * grad_output_stride_n + channel * grad_output_stride_c
    count = tl.max(tl.where(mask, w1 - w0, 0), 0)
    for contributor in range(count):
        output_x = w0 + contributor
        in_range = mask & (output_x < w1)
        safe_x = tl.where(in_range, output_x, 0)
        value = tl.load(grad_output + output_base + safe_x * grad_output_stride_w)
        accumulator += tl.where(in_range, value.to(accumulator.dtype), 0)

    input_offset = (
        batch * grad_input_stride_n
        + channel * grad_input_stride_c
        + input_x * grad_input_stride_w
    )
    tl.store(grad_input + input_offset, accumulator, mask=mask)


@libentry()
@triton.jit
def _upsample_nearest_exact1d_backward_scalar_kernel(
    grad_output,
    grad_input,
    WB,
    numel,
    channels,
    input_w,
    grad_output_stride_n,
    grad_output_stride_c,
    grad_output_stride_w,
    grad_input_stride_n,
    grad_input_stride_c,
    grad_input_stride_w,
    IS_FP64: tl.constexpr,
    IS_UINT8: tl.constexpr,
):
    for index in range(tl.program_id(0).to(tl.int64), numel, tl.num_programs(0)):
        input_x = index % input_w
        nc = index // input_w
        channel = nc % channels
        batch = nc // channels

        w0 = tl.load(WB + input_x).to(tl.int64)
        w1 = tl.load(WB + input_x + 1).to(tl.int64)

        if IS_UINT8:
            value = tl.full((), 0, tl.int32)
        elif IS_FP64:
            value = tl.full((), 0, tl.float64)
        else:
            value = tl.full((), 0, tl.float32)

        output_base = (
            grad_output + batch * grad_output_stride_n + channel * grad_output_stride_c
        )
        for output_x in range(w0, w1):
            value += tl.load(output_base + output_x * grad_output_stride_w).to(
                value.dtype
            )
        input_offset = (
            batch * grad_input_stride_n
            + channel * grad_input_stride_c
            + input_x * grad_input_stride_w
        )
        tl.store(grad_input + input_offset, value)


def _validate_args(
    grad_output: torch.Tensor,
    output_size: Sequence[int],
    input_size: Sequence[int],
) -> tuple[int, int, int, int]:
    if grad_output.device.type != device.name:
        raise RuntimeError(
            f"Expected grad_output on {device.name}, but got {grad_output.device.type}"
        )
    if grad_output.ndim != 3:
        raise RuntimeError("Expected grad_output to be a 3D tensor")
    if len(output_size) != 1:
        raise RuntimeError("Expected output_size to contain one element")
    if len(input_size) != 3:
        raise RuntimeError("Expected input_size to contain three elements")
    if not grad_output.is_floating_point() and grad_output.dtype != torch.uint8:
        raise RuntimeError(
            f'"upsample_nearest1d_backward_out_frame" not implemented for '
            f"'{grad_output.dtype}'"
        )

    output_w = int(output_size[0])
    batch, channels, input_w = (int(value) for value in input_size)
    if input_w <= 0 or output_w <= 0:
        raise RuntimeError("Input and output sizes should be greater than 0")
    if tuple(grad_output.shape) != (batch, channels, output_w):
        raise RuntimeError(
            f"Expected grad_output shape {(batch, channels, output_w)}, "
            f"but got {tuple(grad_output.shape)}"
        )
    return batch, channels, input_w, output_w


def _upsample_nearest_exact1d_backward_impl(
    grad_output: torch.Tensor,
    output_size: Sequence[int],
    input_size: Sequence[int],
    scales: Optional[float],
    grad_input: Optional[torch.Tensor],
) -> torch.Tensor:
    batch, channels, input_w, output_w = _validate_args(
        grad_output, output_size, input_size
    )
    if grad_input is None:
        grad_input = torch.empty(
            (batch, channels, input_w),
            dtype=grad_output.dtype,
            device=grad_output.device,
        )
    else:
        if grad_input.device != grad_output.device:
            raise RuntimeError(
                f"Expected grad_input on {grad_output.device}, "
                f"but got {grad_input.device}"
            )
        if grad_input.dtype != grad_output.dtype:
            raise RuntimeError(
                f"Expected grad_input dtype {grad_output.dtype}, "
                f"but got {grad_input.dtype}"
            )
        grad_input.resize_((batch, channels, input_w))

    if grad_input.numel() == 0:
        return grad_input

    wb = _boundary_tensor(input_w, output_w, scales, grad_output.device)
    numel = grad_input.numel()
    block_size = 256
    is_fp64 = grad_output.dtype == torch.float64
    is_uint8 = grad_output.dtype == torch.uint8

    int64_index = any(
        sum((size - 1) * stride for size, stride in zip(t.shape, t.stride()))
        > torch.iinfo(torch.int32).max
        for t in (grad_output, grad_input)
    )

    with torch_device_fn.device(grad_output.device):
        if int64_index:
            _upsample_nearest_exact1d_backward_scalar_kernel[(min(numel, 65535),)](
                grad_output,
                grad_input,
                wb,
                numel,
                channels,
                input_w,
                *grad_output.stride(),
                *grad_input.stride(),
                IS_FP64=is_fp64,
                IS_UINT8=is_uint8,
            )
            return grad_input
        _upsample_nearest_exact1d_backward_kernel[
            (triton.cdiv(numel, block_size),)
        ](
            grad_output,
            grad_input,
            wb,
            numel,
            channels,
            input_w,
            *grad_output.stride(),
            *grad_input.stride(),
            IS_FP64=is_fp64,
            IS_UINT8=is_uint8,
            BLOCK_SIZE=block_size,
        )
    return grad_input


def _upsample_nearest_exact1d_backward(
    grad_output: torch.Tensor,
    output_size: Sequence[int],
    input_size: Sequence[int],
    scales: Optional[float] = None,
) -> torch.Tensor:
    logger.debug("GEMS_KUNLUNXIN _UPSAMPLE_NEAREST_EXACT1D_BACKWARD")
    return _upsample_nearest_exact1d_backward_impl(
        grad_output, output_size, input_size, scales, None
    )


def _upsample_nearest_exact1d_backward_grad_input(
    grad_output: torch.Tensor,
    output_size: Sequence[int],
    input_size: Sequence[int],
    scales: Optional[float] = None,
    *,
    grad_input: torch.Tensor,
) -> torch.Tensor:
    logger.debug("GEMS_KUNLUNXIN _UPSAMPLE_NEAREST_EXACT1D_BACKWARD.GRAD_INPUT")
    return _upsample_nearest_exact1d_backward_impl(
        grad_output, output_size, input_size, scales, grad_input
    )
