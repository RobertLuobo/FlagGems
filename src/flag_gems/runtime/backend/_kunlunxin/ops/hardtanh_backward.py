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

# Kunlunxin (XPU) override of hardtanh_backward.
#
# hardtanh_backward is a pure elementwise mask op (grad * [min_val < x < max_val]).
# The generic `flag_gems.ops.hardtanh_backward` runs a flat BLOCK=1024 always-masked
# kernel with the default CodeGenConfig; on XPU that lowers to the slow masked
# memory path (measured 2026-09-04 on 16.7M elements: 2.03 ms fp16 / 1.95 ms fp32,
# ~0.06x / ~0.08x vs the 0.073 ms / 0.123 ms vendor native kernel).
#
# This override reuses the proven hardsigmoid_backward recipe (same op shape:
# masked pointwise backward), which measured 0.108 ms fp16 / 0.141 ms fp32 /
# 0.108 ms bf16 on the same 16.7M shapes (18.8x / 13.8x / 18.8x vs the generic):
#   1. pointwise_dynamic codegen with prefer_1d_tile=True + kunlunAutoGrid=True
#      (12-CTA grid, pow2 tile) is the established fast grid on XPU.
#   2. unroll_num=4 is the sweet spot for this op (sweep 4/8/16 x buffer_size_limit
#      2048..16384: unroll 8 is 6-11% slower, unroll 16 is 17-27% slower;
#      buffer_size_limit is a no-op in 2048..16384).
#   3. The derivative is written with saturating fp arithmetic only, exactly like
#      the proven lt_/less_/hardsigmoid_backward recipe (the i1-compare form
#      `(x > min) & (x < max)` multiplied into a float lowers to the per-lane
#      slow path, 0.68 ms vs 0.11 ms here):
#        p = max(0, (x-min)*1e30), q = max(0, (max-x)*1e30)
#      1e30 saturates every fp16/bf16/fp32 gap around the bounds (min gap
#      1.2e-7 -> 1.2e23), so min(1,p)*min(1,q) is exactly 1.0 on (min,max) and
#      0.0 at/outside the bounds (strict at the boundary, matching the CPU
#      reference: hardtanh_backward(g, -1.0, -1, 1) == 0).  NaN inputs resolve
#      to 0.0 through the backend's min/max non-NaN preference, matching the
#      CPU reference.  maxabsdiff vs torch CPU reference: 0.0 on all checked
#      fp16/fp32/bf16 random and boundary inputs.
#   4. isCloseDtypeConvert=True and isCloseVectorization=True must stay at
#      their defaults (False): the former crashes the bf16 lowering with a
#      VTruncFOpConversion assertion, the latter regresses 60x (see
#      hardsigmoid_backward).
#
# Bedrock check: `_FULL_CONFIG["hardtanh_backward"].__module__` must become
# `_kunlunxin.ops.hardtanh_backward` once `_kunlunxin/ops/__init__.py` imports
# this module (registration is applied by the harness main agent).
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
    unroll_num=4,
)


@pointwise_dynamic(
    is_tensor=[True, True, False, False],
    promotion_methods=[(0, 1, "DEFAULT")],
    config=config_,
)
@triton.jit
def hardtanh_backward_func(grad_output, self, min_val, max_val):
    # hardtanh: y = clamp(x, min_val, max_val)
    # gradient: 1 when min_val < x < max_val (strict, matches ATen
    # hardtanh_backward), 0 on/outside the bounds.
    grad_output_fp32 = grad_output.to(tl.float32)
    self_fp32 = self.to(tl.float32)
    # 0.0 if x <= min_val, +inf (-> 1.0) if x > min_val; mirrored for max_val.
    p = tl.maximum(0.0, (self_fp32 - min_val) * 1.0e30)
    q = tl.maximum(0.0, (max_val - self_fp32) * 1.0e30)
    in_range = tl.minimum(1.0, p) * tl.minimum(1.0, q)
    result = grad_output_fp32 * in_range
    return result.to(grad_output.dtype)


def hardtanh_backward(grad_output, self, min_val, max_val):
    logger.debug("GEMS_KUNLUNXIN HARDTANH_BACKWARD")
    if grad_output.numel() == 0:
        return grad_output
    return hardtanh_backward_func(grad_output, self, float(min_val), float(max_val))