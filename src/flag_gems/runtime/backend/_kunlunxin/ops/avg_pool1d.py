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

from flag_gems.runtime.backend._kunlunxin.ops.avg_pool2d import avg_pool2d

logger = logging.getLogger(__name__)


def avg_pool1d(
    input: torch.Tensor,
    kernel_size,
    stride=None,
    padding=0,
    ceil_mode=False,
    count_include_pad=True,
):
    """Average pooling over 1D input (Kunlunxin/XPU implementation).

    The generic implementation reshapes to 2D and calls the generic
    avg_pool2d, whose 2D (BLOCK_H, BLOCK_W) tiled + autotuned forward kernel
    fails TritonXPULegalize on XPU3.  This override keeps the same reshape
    logic but dispatches to the Kunlunxin avg_pool2d override, which uses a
    flat 1D constexpr kernel that lowers cleanly on XPU.
    """
    logger.debug("GEMS_KUNLUNXIN AVG_POOL1D")

    assert input.ndim == 3, f"avg_pool1d expects 3D input, got {input.ndim}D"

    input_4d = input.unsqueeze(2)  # (N, C, L) -> (N, C, 1, L)

    if isinstance(kernel_size, int):
        kernel_size_1d = kernel_size
        kernel_size_2d = (1, kernel_size)
    else:
        kernel_size_1d = kernel_size[0]
        kernel_size_2d = (1, kernel_size[0])

    if stride is None or (isinstance(stride, list) and len(stride) == 0):
        stride_2d = (1, kernel_size_1d)
    elif isinstance(stride, int):
        stride_2d = (1, stride)
    else:
        stride_2d = (1, stride[0])

    if isinstance(padding, int):
        padding_2d = (0, padding)
    else:
        padding_2d = (0, padding[0])

    output_4d = avg_pool2d(
        input_4d,
        kernel_size=kernel_size_2d,
        stride=stride_2d,
        padding=padding_2d,
        ceil_mode=ceil_mode,
        count_include_pad=count_include_pad,
        divisor_override=None,
    )

    return output_4d.squeeze(2)
