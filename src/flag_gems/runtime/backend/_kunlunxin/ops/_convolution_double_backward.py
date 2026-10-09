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

"""Kunlunxin XPU override for ``aten::_convolution_double_backward``.

The generic FlagGems implementation drives the second-order gradients through
``conv_transpose2d`` and a dedicated im2col correlation kernel. Both use
``tl.dot`` which the P800 TritonXPU backend lowers to the ``tf32x3`` precision
it rejects (``input_precision must be one of ('ieee','tf32')``), so 210/252
cases fail to compile or mis-compute.

This override re-expresses every term through the validated device-resident
Kunlunxin ``conv2d`` kernels (scalar fp32 accumulation, no ``tl.dot``):

  * forward convolutions -> the vendor ``conv2d`` forward kernel;
  * ``conv_transpose(a, w)`` terms -> the input-gradient of a forward conv2d,
    computed through the vendor ``Conv2d`` autograd backward kernel;
  * weight-correlation terms -> the weight-gradient of a forward conv2d, again
    via the vendor backward kernel.

No ``conv_transpose2d`` and no ``tl.dot`` are touched. 1D convolutions are
handled by promoting the lone spatial axis to a degenerate 2D one.
"""

import logging

import torch

from .add import add
from .conv2d import Conv2d, conv2d

logger = logging.getLogger(__name__)


def _grad_input(out_grad, weight, input_shape, stride, padding, dilation, groups):
    """``conv_transpose(out_grad, weight)``: the input-gradient of the forward
    conv2d that maps ``input_shape`` to ``out_grad.shape`` with ``weight``.

    Runs through the vendor ``Conv2d`` autograd Function, whose backward uses
    the scalar ``_input_grad`` kernel (no ``tl.dot``). Only tensor shape of the
    throwaway forward matters; the linear input-gradient is independent of the
    forward output values.
    """
    with torch.enable_grad():
        u = torch.zeros(
            tuple(input_shape),
            device=out_grad.device,
            dtype=out_grad.dtype,
            requires_grad=True,
        )
        v = Conv2d.apply(u, weight, None, stride, padding, dilation, groups)
        (gu,) = torch.autograd.grad(v, u, grad_outputs=out_grad.detach())
    return gu


def _grad_weight(inp, out_grad, weight_shape, stride, padding, dilation, groups):
    """Weight-correlation term: the weight-gradient of the forward conv2d
    ``conv2d(inp, w)`` whose output matches ``out_grad``.

    Uses the vendor ``Conv2d`` backward ``_weight_grad`` kernel (no ``tl.dot``).
    """
    with torch.enable_grad():
        wd = torch.zeros(
            tuple(weight_shape),
            device=inp.device,
            dtype=inp.dtype,
            requires_grad=True,
        )
        v = Conv2d.apply(inp.detach(), wd, None, stride, padding, dilation, groups)
        (gw,) = torch.autograd.grad(v, wd, grad_outputs=out_grad.detach())
    return gw


def _dbw_2d(
    ggI,
    ggW,
    ggb,
    gO,
    weight,
    self_,
    stride,
    padding,
    dilation,
    transposed,
    output_padding,
    groups,
):
    gO_h, gO_w = gO.shape[2], gO.shape[3]
    self_shape = self_.shape

    grad_ggO = None
    grad_self = None
    grad_weight = None

    # grad_ggO = sum of the ggI, ggW and ggb contributions.
    parts = []
    if ggI is not None:
        if not transposed:
            parts.append(conv2d(ggI, weight, None, stride, padding, dilation, groups))
        else:
            parts.append(
                _grad_input(ggI, weight, gO.shape, stride, padding, dilation, groups)
            )
    if ggW is not None:
        if not transposed:
            parts.append(conv2d(self_, ggW, None, stride, padding, dilation, groups))
        else:
            parts.append(
                _grad_input(self_, ggW, gO.shape, stride, padding, dilation, groups)
            )
    if ggb is not None:
        bias_part = (
            ggb.view(1, -1, 1, 1)
            .expand(gO.shape[0], ggb.shape[0], gO_h, gO_w)
            .contiguous()
        )
        parts.append(bias_part)

    if parts:
        grad_ggO = parts[0]
        for p in parts[1:]:
            grad_ggO = add(grad_ggO, p)

    # grad_self depends only on ggW.
    if ggW is not None:
        if not transposed:
            grad_self = _grad_input(
                gO, ggW, self_shape, stride, padding, dilation, groups
            )
        else:
            grad_self = conv2d(gO, ggW, None, stride, padding, dilation, groups)

    # grad_weight depends only on ggI.
    if ggI is not None:
        if not transposed:
            grad_weight = _grad_weight(
                ggI, gO, weight.shape, stride, padding, dilation, groups
            )
        else:
            grad_weight = _grad_weight(
                gO, ggI, weight.shape, stride, padding, dilation, groups
            )

    return grad_ggO, grad_self, grad_weight


def _widen(param, fill):
    if param is None:
        return None
    if isinstance(param, (list, tuple)):
        return [int(param[0]), fill]
    return [int(param), fill]


def _dbw_1d(
    ggI,
    ggW,
    ggb,
    gO,
    weight,
    self_,
    stride,
    padding,
    dilation,
    transposed,
    output_padding,
    groups,
):
    def _unsqueeze(t):
        return t.unsqueeze(-1) if t is not None else None

    def _squeeze(t):
        return t.squeeze(-1) if t is not None else None

    res_2d = _dbw_2d(
        _unsqueeze(ggI),
        _unsqueeze(ggW),
        ggb,
        _unsqueeze(gO),
        _unsqueeze(weight),
        _unsqueeze(self_),
        _widen(stride, 1),
        _widen(padding, 0),
        _widen(dilation, 1),
        transposed,
        _widen(output_padding, 0),
        groups,
    )
    return tuple(_squeeze(r) for r in res_2d)


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
    spatial_ndim = weight.ndim - 2
    if spatial_ndim == 2:
        return _dbw_2d(
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
        )
    if spatial_ndim == 1:
        return _dbw_1d(
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
        )
    raise NotImplementedError(
        "_convolution_double_backward: Kunlunxin supports 1D (3-D operands) and "
        "2D (4-D operands) convolutions only; got "
        f"self={self.ndim}, weight={weight.ndim}, gO={gO.ndim}."
    )
