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

import torch.nn.functional as F

from .conv1d import conv1d
from .conv2d import conv2d
from .conv3d import conv3d

logger = logging.getLogger(__name__)


def _same_padding(input, weight, dilation):
    spatial_ndim = weight.ndim - 2
    kernel_size = weight.shape[-spatial_ndim:]
    pad = []
    for k, d in zip(reversed(kernel_size), reversed(dilation)):
        total = d * (k - 1)
        pad.append(total // 2)
        pad.append(total - total // 2)
    return F.pad(input, pad)


def _convolution_mode(input, weight, bias, stride, padding, dilation, groups):
    logger.debug("GEMS_KUNLUNXIN _CONVOLUTION_MODE")
    if padding == "same":
        if any(s != 1 for s in stride):
            raise ValueError("padding='same' is not supported for strided convolutions")
    elif padding == "valid":
        pass
    else:
        raise ValueError(
            f"Invalid padding string: {padding!r}, should be one of {{'same', 'valid'}}"
        )

    spatial_ndim = weight.ndim - 2

    if padding == "same":
        input = _same_padding(input, weight, dilation)
    # After "same" pad (or for "valid"), run the kunlunxin conv overlays with
    # explicit zero padding; this reuses the working XPU scalar-accum kernels.
    if spatial_ndim == 1:
        return conv1d(input, weight, bias, stride, 0, dilation, groups)
    elif spatial_ndim == 2:
        return conv2d(input, weight, bias, stride, 0, dilation, groups)
    elif spatial_ndim == 3:
        return conv3d(input, weight, bias, stride, 0, dilation, groups)

    raise ValueError(f"Unsupported convolution with {spatial_ndim} spatial dimensions")
