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

# Kunlunxin (XPU) override of hardsigmoid_backward.
#
# The generic `flag_gems.ops.hardsigmoid_backward` runs through pointwise_dynamic
# without an explicit CodeGenConfig, which on XPU generates a kernel with
# runtime (non-constexpr) strides/num_tasks and no codegen knobs, forcing the
# slow masked/scalarized memory path (measured: 53.9 ms for 16.7M fp16 vs
# 0.056 ms torch, 0.001x).
#
# Two XPU-specific points, both measured on XPU 4 (2026-09-03, 16.7M fp16):
#   1. The 1D-tile codegen with kunlunAutoGrid=True (12 CTAs, pow2 tile) is
#      the established fast grid (mse_loss_backward / lt_ / hardsigmoid
#      forward all use it; 0.089 ms here).
#   2. isCloseVectorization must stay at its default (False). Passing True
#      makes the XPU backend drop ALL vector loads/stores: the same kernel
#      regresses 0.089 ms -> 5.34 ms (60x). The i1-compare form
#      (`(x > -3) & (x < 3)` multiplied into a float) also lowers to the
#      per-lane slow path (0.65 ms), so the derivative is written with
#      saturating fp arithmetic only, exactly like the proven lt_/less_
#      recipe: p = max(0,(x+3)*1e30), q = max(0,(3-x)*1e30), 1e30 saturates
#      every fp16/bf16/fp32 gap around +-3 (min gap 1.2e-7 -> 1.2e23), so
#      min(1,p)*min(1,q) is exactly 1.0 on (-3,3), 0.0 elsewhere, strict at
#      +-3 (matches the CPU reference); NaN inputs resolve to 0.0 through
#      the backend's min/max non-NaN preference, matching the CPU reference
#      (the vendor XPU kernel's NaN->1/6 differs, but tests use --ref cpu).
#      maxabsdiff vs torch CPU reference: 0.0 on 16.7M random fp16/fp32/bf16.
import logging

import triton
import triton.language as tl
from _kunlunxin.utils.codegen_config_utils import CodeGenConfig

from ..utils.pointwise_dynamic import pointwise_dynamic

logger = logging.getLogger(__name__)

config_ = CodeGenConfig(
    512,
    (65536, 65536, 65536),
    32,
    True,
    prefer_1d_tile=True,
    buffer_size_limit=4096,
    kunlunAutoGrid=True,
    unroll_num=8,
)


@pointwise_dynamic(promotion_methods=[(0, 1, "DEFAULT")], config=config_)
@triton.jit
def hardsigmoid_backward_func(grad_output, self):
    # hardsigmoid: y = clamp(x/6 + 0.5, 0, 1)
    # gradient: dy/dx = 1/6 when -3 < x < 3, else 0
    # => grad_input = grad_output * (|self| < 3) / 6
    grad_output_fp32 = grad_output.to(tl.float32)
    self_fp32 = self.to(tl.float32)
    # `0.0 if x <= -3, `+inf (->1.0) if x > -3`; and the mirrored bound.
    p = tl.maximum(0.0, (self_fp32 + 3.0) * 1.0e30)
    q = tl.maximum(0.0, (3.0 - self_fp32) * 1.0e30)
    in_range = tl.minimum(1.0, p) * tl.minimum(1.0, q)
    result = grad_output_fp32 * in_range * (1.0 / 6.0)
    return result.to(grad_output.dtype)


def hardsigmoid_backward(grad_output, self):
    logger.debug("GEMS_KUNLUNXIN HARDSIGMOID_BACKWARD")
    return hardsigmoid_backward_func(grad_output, self)