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

from flag_gems.runtime.backend._kunlunxin.ops.add import add as _xpu_add
from flag_gems.runtime.backend._kunlunxin.ops.conv1d import conv1d as _xpu_conv1d
from flag_gems.runtime.backend._kunlunxin.ops.conv2d import Conv2d as _XpuConv2d
from flag_gems.runtime.backend._kunlunxin.ops.conv2d import conv2d as _xpu_conv2d
from flag_gems.runtime.backend._kunlunxin.ops.conv3d import conv3d as _xpu_conv3d
from flag_gems.runtime.backend._kunlunxin.ops.conv_transpose1d import (
    conv_transpose1d as _xpu_conv_transpose1d,
)

logger = logging.getLogger(__name__)


def _pair(value):
    if isinstance(value, (list, tuple)):
        if len(value) == 1:
            return int(value[0]), int(value[0])
        return int(value[0]), int(value[1])
    return int(value), int(value)


def _conv_transpose2d_via_grad_input(
    input, weight, bias, stride, padding, dilation, output_padding, groups
): 
    stride_h, stride_w = _pair(stride)
    pad_h, pad_w = _pair(padding)
    dil_h, dil_w = _pair(dilation)
    opad_h, opad_w = _pair(output_padding)

    kH, kW = weight.shape[2], weight.shape[3]
    out_c = weight.shape[1] * groups
    out_h = (
        (input.shape[2] - 1) * stride_h - 2 * pad_h + dil_h * (kH - 1) + opad_h + 1
    )
    out_w = (
        (input.shape[3] - 1) * stride_w - 2 * pad_w + dil_w * (kW - 1) + opad_w + 1
    )
    target_shape = (input.shape[0], out_c, out_h, out_w)

    with torch.enable_grad():
        x = torch.zeros(
            target_shape,
            device=input.device,
            dtype=input.dtype,
            requires_grad=True,
        )
        out = _XpuConv2d.apply(
            x,
            weight.detach(),
            None,
            stride,
            padding,
            dilation,
            groups,
        )
        out.backward(input)
    result = x.grad.detach()

    if bias is not None:
        result = _xpu_add(result, bias.view(1, -1, 1, 1))
    return result


def _convolution_overrideable_impl(
    input,
    weight,
    bias,
    stride,
    padding,
    dilation,
    transposed,
    output_padding,
    groups,
): 
    spatial_dims = weight.ndim - 2
    assert spatial_dims in (1, 2, 3), (
        f"convolution_overrideable only supports 1D/2D/3D convolutions, "
        f"received weight with shape {tuple(weight.shape)}"
    )

    if transposed:
        if spatial_dims == 1:
            return _xpu_conv_transpose1d(
                input, weight, bias, stride, padding, output_padding, groups, dilation
            )
        if spatial_dims == 2:
            return _conv_transpose2d_via_grad_input(
                input, weight, bias, stride, padding, dilation, output_padding, groups
            )
        raise NotImplementedError(
            "convolution_overrideable does not support 3D transposed convolution."
        )

    if spatial_dims == 1:
        stride_w = _pair(stride)[0]
        padding_w = _pair(padding)[0]
        dilation_w = _pair(dilation)[0]
        return _xpu_conv1d(
            input, weight, bias, stride_w, padding_w, dilation_w, groups
        )
    if spatial_dims == 2:
        return _xpu_conv2d(input, weight, bias, stride, padding, dilation, groups)
    return _xpu_conv3d(input, weight, bias, stride, padding, dilation, groups)


def convolution_overrideable(
    input,
    weight,
    bias,
    stride,
    padding,
    dilation,
    transposed,
    output_padding,
    groups,
):
    logger.debug("GEMS_KUNLUNXIN CONVOLUTION_OVERRIDEABLE")
    return _convolution_overrideable_impl(
        input,
        weight,
        bias,
        stride,
        padding,
        dilation,
        transposed,
        output_padding,
        groups,
    )


def convolution_overrideable_out(
    input,
    weight,
    bias,
    stride,
    padding,
    dilation,
    transposed,
    output_padding,
    groups,
    *,
    out,
):
    logger.debug("GEMS_KUNLUNXIN CONVOLUTION_OVERRIDEABLE_OUT")
    result = _convolution_overrideable_impl(
        input,
        weight,
        bias,
        stride,
        padding,
        dilation,
        transposed,
        output_padding,
        groups,
    )
    out.copy_(result)
    return out
