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

import triton
import triton.language as tl

from flag_gems.utils import tl_extra_shim

from ..utils.pointwise_dynamic import pointwise_dynamic

logger = logging.getLogger(__name__)
erf = tl_extra_shim.erf


@pointwise_dynamic(promotion_methods=[(0, 1, "DEFAULT")])
@triton.jit
def infinitely_differentiable_gelu_backward_kernel(grad, self_input):
    scale1: tl.constexpr = 0.7071067811  # 1 / math.sqrt(2)
    scale2: tl.constexpr = 0.3989422803  # 1 / math.sqrt(2 * math.pi)
    x_fp32 = self_input.to(tl.float32)
    scaled_x = scale1 * x_fp32
    dydx = scale2 * x_fp32 * tl.exp(-scaled_x * scaled_x) + 0.5 * erf(scaled_x) + 0.5
    dx = dydx * grad
    return dx


def infinitely_differentiable_gelu_backward(grad, self_input):
    logger.debug("GEMS_KUNLUNXIN INFINITELY_DIFFERENTIABLE_GELU_BACKWARD")
    return infinitely_differentiable_gelu_backward_kernel(grad, self_input)
