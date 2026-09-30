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

# from flag_gems import runtime
from flag_gems.utils import libentry

from .conv2d import conv2d_output_size

logger = logging.getLogger(__name__)


def conv3d_output_size(
    in_size: int,
    kernel_size: int,
    stride: int,
    padding: int,
    dilation: int,
) -> int:
    """
    Determines the output size of a 3D convolution operation.

    Args:
        in_size: Input size.
        kernel_size: Kernel size.
        stride: Stride.
        padding: Padding.
        dilation: Dilation.

    Returns:
        Output size of 3D convolution.
    """
    return conv2d_output_size(in_size, kernel_size, stride, padding, dilation)


@libentry()
@triton.jit
def conv3d_forward_kernel(
    input_pointer,
    weight_pointer,
    output_pointer,
    bias_pointer,
    in_n,
    input_depth,
    input_height,
    input_width,
    out_c,
    out_depth,
    out_height,
    out_width,
    input_n_stride,
    input_c_stride,
    input_depth_stride,
    input_height_stride,
    input_width_stride,
    weight_n_stride,
    weight_c_stride,
    weight_depth_stride,
    weight_height_stride,
    weight_width_stride,
    output_n_stride,
    output_c_stride,
    output_depth_stride,
    output_height_stride,
    output_width_stride,
    weight_c: tl.constexpr,
    weight_depth: tl.constexpr,
    weight_height: tl.constexpr,
    weight_width: tl.constexpr,
    stride_depth: tl.constexpr,
    stride_height: tl.constexpr,
    stride_width: tl.constexpr,
    padding_depth: tl.constexpr,
    padding_height: tl.constexpr,
    padding_width: tl.constexpr,
    dilation_depth: tl.constexpr,
    dilation_height: tl.constexpr,
    dilation_width: tl.constexpr,
    out_per_group_c: tl.constexpr,
    BLOCK: tl.constexpr,
):
    # Flat 1D program over (in_n, out_c, out_depth, out_height, out_width).
    # Scalar spatial/channel loops with tl.where post-load zeroing mirror the
    # kunlunxin conv2d forward kernel: masked-load `other=0.0` does NOT reliably
    # zero out-of-range lanes on XPU, so invalid taps must be zeroed explicitly.
    total = in_n * out_c * out_depth * out_height * out_width
    m = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = m < total

    ow = m % out_width
    q = m // out_width
    oh = q % out_height
    q = q // out_height
    od = q % out_depth
    q = q // out_depth
    oc = q % out_c
    ni = q // out_c

    group = oc // out_per_group_c
    ci_base = group * weight_c

    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    for kd in range(weight_depth):
        idd = od * stride_depth - padding_depth + kd * dilation_depth
        for kh in range(weight_height):
            ih = oh * stride_height - padding_height + kh * dilation_height
            for kw in range(weight_width):
                iw = ow * stride_width - padding_width + kw * dilation_width
                valid = (
                    mask
                    & (idd >= 0)
                    & (idd < input_depth)
                    & (ih >= 0)
                    & (ih < input_height)
                    & (iw >= 0)
                    & (iw < input_width)
                )
                safe_id = tl.where(valid, idd, 0)
                safe_ih = tl.where(valid, ih, 0)
                safe_iw = tl.where(valid, iw, 0)
                for ci in range(weight_c):
                    xv = tl.load(
                        input_pointer
                        + ni * input_n_stride
                        + (ci_base + ci) * input_c_stride
                        + safe_id * input_depth_stride
                        + safe_ih * input_height_stride
                        + safe_iw * input_width_stride,
                        mask=mask,
                        other=0.0,
                    )
                    xv = tl.where(valid, xv, 0.0)
                    wv = tl.load(
                        weight_pointer
                        + oc * weight_n_stride
                        + ci * weight_c_stride
                        + kd * weight_depth_stride
                        + kh * weight_height_stride
                        + kw * weight_width_stride
                    )
                    acc += xv.to(tl.float32) * wv.to(tl.float32)

    acc += tl.load(bias_pointer + oc, mask=mask, other=0.0).to(tl.float32)

    tl.store(
        output_pointer
        + ni * output_n_stride
        + oc * output_c_stride
        + od * output_depth_stride
        + oh * output_height_stride
        + ow * output_width_stride,
        acc,
        mask=mask,
    )


# class Conv3d(torch.autograd.Function):
#     @staticmethod
#     def forward(ctx, input, weight, bias, stride, padding, dilation, groups):
#         pass


def conv3d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
    logger.debug("GEMS_KUNLUNXIN CONV3D")

    assert weight.ndim == 5, "Weights must be 5D, received shape {weight.shape}"
    assert (
        bias is None or bias.ndim == 1
    ), "Bias must be 1D, received shape {bias.shape}"

    assert (
        input.shape[1] == groups * weight.shape[1]
    ), "Incompatible input ({input.shape}) and weights ({weight.shape}) shape with {groups} groups"
    assert (
        bias is None or weight.shape[0] == bias.shape[0]
    ), "Incompatible weights ({weight.shape}) and bias ({bias.shape}) shape"

    if isinstance(stride, (list, tuple)):
        stride_depth, stride_height, stride_width = stride
    else:
        stride_depth = stride_height = stride_width = stride

    if isinstance(padding, (list, tuple)):
        padding_depth, padding_height, padding_width = padding
    else:
        padding_depth = padding_height = padding_width = padding

    if isinstance(dilation, (list, tuple)):
        dilation_depth, dilation_height, dilation_width = dilation
    else:
        dilation_depth = dilation_height = dilation_width = dilation

    in_n, _, input_depth, input_height, input_width = input.shape
    out_c, weight_c, weight_depth, weight_height, weight_width = weight.shape
    out_depth = conv3d_output_size(
        input_depth, weight_depth, stride_depth, padding_depth, dilation_depth
    )

    out_height = conv3d_output_size(
        input_height, weight_height, stride_height, padding_height, dilation_height
    )
    out_width = conv3d_output_size(
        input_width, weight_width, stride_width, padding_width, dilation_width
    )

    output_dtype = input.dtype

    # For float16 inputs, promote to float32 for computation to prevent overflow
    # Same issue as conv2d: float16 max value ~65504, 3D convolution with large
    # channels/kernels easily overflows causing NaN propagation
    use_fp32_compute = input.dtype == torch.float16
    if use_fp32_compute:
        input = input.to(torch.float32)
        weight = weight.to(torch.float32)
        if bias is not None:
            bias = bias.to(torch.float32)
        compute_dtype = torch.float32
    else:
        compute_dtype = output_dtype

    output = torch.empty(
        (in_n, out_c, out_depth, out_height, out_width),
        device=input.device,
        dtype=compute_dtype,
    )

    # Flat 1D grid over every output element (in_n, out_c, out_depth, out_height, out_width).
    BLOCK = 32
    grid = lambda META: (
        triton.cdiv(
            in_n * out_c * out_depth * out_height * out_width, META["BLOCK"]
        ),
    )

    if bias is None:
        bias_pointer = torch.zeros(out_c, device=input.device, dtype=torch.float)
    else:
        bias_pointer = bias.to(torch.float)

    conv3d_forward_kernel[grid](
        input,
        weight,
        output,
        bias_pointer,
        in_n,
        input_depth,
        input_height,
        input_width,
        out_c,
        out_depth,
        out_height,
        out_width,
        *input.stride(),
        *weight.stride(),
        *output.stride(),
        weight_c,
        weight_depth,
        weight_height,
        weight_width,
        stride_depth,
        stride_height,
        stride_width,
        padding_depth,
        padding_height,
        padding_width,
        dilation_depth,
        dilation_height,
        dilation_width,
        out_per_group_c=out_c // groups,
        BLOCK=BLOCK,
    )

    # Convert back to original dtype if we promoted to fp32
    if use_fp32_compute:
        output = output.to(output_dtype)

    return output
