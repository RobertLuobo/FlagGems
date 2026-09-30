# Copyright 2026, The FlagOS Contributors.
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
import importlib
import logging

import torch

from flag_gems.runtime.backend._kunlunxin.ops.conv2d import Conv2d as _XpuConv2d
from flag_gems.runtime.backend._kunlunxin.ops.conv2d import conv2d as _xpu_conv2d

logger = logging.getLogger(__name__)

# The submodule is shadowed by a same-named function in ``flag_gems.ops``; use importlib to fetch the real module.
_generic = importlib.import_module("flag_gems.ops._convolution_double_backward")

_generic.conv2d = _xpu_conv2d


def _grad_weight_correlation(
    input, out_grad, weight_shape, stride, padding, dilation, groups
):
    """XPU3-safe weight-shaped gradient via the Kunlunxin conv2d autograd.

    Computes ``weight[c_out, c_in, kh, kw] = sum_{n,h,w} input[n, c_in, ...] *
    out_grad[n, c_out, h, w]`` by recognising it as the weight gradient of a
    forward conv2d(input, weight) backpropagated with ``out_grad``. Delegates to
    the validated Kunlunxin conv2d (whose backward-weight kernel is correct on
    XPU3), avoiding the generic correlation kernel that mis-computes on XPU3.
    """
    with torch.enable_grad():
        weight = torch.zeros(
            weight_shape,
            device=input.device,
            dtype=input.dtype,
            requires_grad=True,
        )
        out = _xpu_conv2d(
            input.detach(),
            weight,
            None,
            stride,
            padding,
            dilation,
            groups,
        )
        out.backward(out_grad)
    return weight.grad.detach()


_generic._grad_weight_correlation = _grad_weight_correlation


def _conv2d_grad_input(
    grad_output, weight, input_shape, stride, padding, dilation, groups
):
    """XPU3-safe input-shaped gradient via the Kunlunxin conv2d autograd.

    Computes the adjoint of a forward conv2d w.r.t. its input, i.e. the value a
    ``conv_transpose2d(grad_output, weight)`` would produce. Recognised as the
    input gradient of ``conv2d(x, weight)`` backpropagated with ``grad_output``;
    delegates to the validated Kunlunxin conv2d autograd, avoiding the broken
    XPU3 conv_transpose2d op. Uses ``Conv2d.apply`` directly rather than the
    ``conv2d`` wrapper because the wrapper's non-square square-padding fold is
    not autograd-aware and would sever the input's grad path (the fold is
    irrelevant here: the input gradient of a linear conv is independent of the
    forward output values, so the direct-apply backward is exact).
    """
    with torch.enable_grad():
        x = torch.zeros(
            input_shape,
            device=grad_output.device,
            dtype=grad_output.dtype,
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
        out.backward(grad_output)
    return x.grad.detach()


def _convolution_double_backward_2d(
    ggI,
    ggW,
    ggb,
    gO,
    weight,
    self,
    stride,
    padding,
    dilation,
    transposed,
    output_padding,
    groups,
    output_mask,
):
    """XPU3-safe 2D second-order convolution backward.

    Mirrors the generic ``_convolution_double_backward_2d`` composition but
    expresses every transpose-shaped term (``grad_self`` for the plain conv, the
    ``ggI`` contribution to ``grad_ggO`` for the transposed conv) through the
    validated conv2d autograd instead of the broken XPU3 conv_transpose2d.
    """
    _pair = _generic._pair
    add = _generic.add

    stride_h, stride_w = _pair(stride)
    pad_h, pad_w = _pair(padding)
    dil_h, dil_w = _pair(dilation)
    opad_h, opad_w = _pair(output_padding)

    gO_h, gO_w = gO.shape[2], gO.shape[3]
    kH, kW = weight.shape[2], weight.shape[3]

    grad_ggO = None
    grad_self = None
    grad_weight = None

    parts = []
    if ggI is not None:
        if not transposed:
            parts.append(
                _xpu_conv2d(
                    ggI, weight, None, stride, padding, dilation, groups
                )
            )
        else:
            # transposed ggI term = input grad of conv2d at the forward-output spatial size.
            in_c = weight.shape[1] * groups
            out_h = (
                (ggI.shape[2] - 1) * stride_h
                - 2 * pad_h
                + dil_h * (kH - 1)
                + opad_h
                + 1
            )
            out_w = (
                (ggI.shape[3] - 1) * stride_w
                - 2 * pad_w
                + dil_w * (kW - 1)
                + opad_w
                + 1
            )
            target_shape = (ggI.shape[0], in_c, out_h, out_w)
            parts.append(
                _conv2d_grad_input(
                    ggI, weight, target_shape, stride, padding, dilation, groups
                )
            )
    if ggW is not None:
        if not transposed:
            parts.append(
                _xpu_conv2d(self, ggW, None, stride, padding, dilation, groups)
            )
        else:
            in_c = ggW.shape[1] * groups
            out_h = (
                (self.shape[2] - 1) * stride_h
                - 2 * pad_h
                + dil_h * (kH - 1)
                + opad_h
                + 1
            )
            out_w = (
                (self.shape[3] - 1) * stride_w
                - 2 * pad_w
                + dil_w * (kW - 1)
                + opad_w
                + 1
            )
            target_shape = (self.shape[0], in_c, out_h, out_w)
            parts.append(
                _conv2d_grad_input(
                    self, ggW, target_shape, stride, padding, dilation, groups
                )
            )
    if ggb is not None:
        bias_part = ggb.view(1, -1, 1, 1).expand(1, ggb.shape[0], gO_h, gO_w)
        bias_part = bias_part.expand(gO.shape[0], -1, -1, -1)
        parts.append(bias_part)

    if parts:
        grad_ggO = parts[0]
        for p in parts[1:]:
            grad_ggO = add(grad_ggO, p)

    if ggW is not None:
        if not transposed:
            # input-gradient form of conv2d(self, ggW) backpropped with gO.
            grad_self = _conv2d_grad_input(
                gO, ggW, self.shape, stride, padding, dilation, groups
            )
        else:
            grad_self = _xpu_conv2d(
                gO, ggW, None, stride, padding, dilation, groups
            )

    if ggI is not None:
        if not transposed:
            grad_weight = _grad_weight_correlation(
                ggI, gO, weight.shape, stride, padding, dilation, groups
            )
        else:
            grad_weight = _grad_weight_correlation(
                gO, ggI, weight.shape, stride, padding, dilation, groups
            )

    return grad_ggO, grad_self, grad_weight


_generic._convolution_double_backward_2d = _convolution_double_backward_2d


def _convolution_double_backward(
    ggI,
    ggW,
    ggb,
    gO,
    weight,
    self,
    stride,
    padding,
    dilation,
    transposed,
    output_padding,
    groups,
    output_mask,
):
    logger.debug("GEMS_KUNLUNXIN _CONVOLUTION_DOUBLE_BACKWARD")
    return _generic._convolution_double_backward(
        ggI,
        ggW,
        ggb,
        gO,
        weight,
        self,
        stride,
        padding,
        dilation,
        transposed,
        output_padding,
        groups,
        output_mask,
    )
