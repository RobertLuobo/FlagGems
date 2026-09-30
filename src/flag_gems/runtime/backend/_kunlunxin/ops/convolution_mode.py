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

from flag_gems.runtime.backend._kunlunxin.ops.conv1d import conv1d as _xpu_conv1d
from flag_gems.runtime.backend._kunlunxin.ops.conv2d import conv2d as _xpu_conv2d
from flag_gems.runtime.backend._kunlunxin.ops.conv3d import conv3d as _xpu_conv3d

logger = logging.getLogger(__name__)

# The submodule is shadowed by a same-named function in ``flag_gems.ops``; use importlib to fetch the real module.
_generic = importlib.import_module("flag_gems.ops._convolution_mode")


def _valid_to_zero(padding):
    """Normalise the string ``"valid"`` to integer zero padding.

    The generic dispatcher passes the raw ``"valid"`` string through to the
    convolution kernel. The Kunlunxin conv1d/conv2d handle that string, but
    conv3d only accepts integer / tuple padding; ``"valid"`` is exactly zero
    padding, so map it uniformly before delegating.
    """
    if isinstance(padding, str) and padding == "valid":
        return 0
    return padding


def _conv1d_mode(input, weight, bias, stride, padding, dilation, groups):
    return _xpu_conv1d(
        input, weight, bias, stride, _valid_to_zero(padding), dilation, groups
    )


def _conv2d_mode(input, weight, bias, stride, padding, dilation, groups):
    return _xpu_conv2d(
        input, weight, bias, stride, _valid_to_zero(padding), dilation, groups
    )


def _conv3d_mode(input, weight, bias, stride, padding, dilation, groups):
    return _xpu_conv3d(
        input, weight, bias, stride, _valid_to_zero(padding), dilation, groups
    )


# Redirect the generic dispatcher's convolution calls to the validated
# Kunlunxin conv kernels (the generic Triton conv kernels mis-compute on XPU3).
# The generic module keeps its exact PyTorch-matching 'same' asymmetric padding
# (F.pad) and validation; only the underlying conv kernel is swapped out.
_generic.conv1d = _conv1d_mode
_generic.conv2d = _conv2d_mode
_generic.conv3d = _conv3d_mode


def _convolution_mode(input, weight, bias, stride, padding, dilation, groups):
    logger.debug("GEMS_KUNLUNXIN _CONVOLUTION_MODE")
    return _generic._convolution_mode(
        input, weight, bias, stride, padding, dilation, groups
    )
