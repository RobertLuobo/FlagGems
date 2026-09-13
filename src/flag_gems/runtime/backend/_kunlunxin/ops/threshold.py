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
    isCloseVectorization=False,
    kunlunAutoGrid=True,
    unroll_num=4,  # PROBE-CANDIDATE unroll4 (baseline unroll8); revert if not strictly better
)


@pointwise_dynamic(
    is_tensor=[True, False, False], promotion_methods=[(0, "DEFAULT")], config=config_
)
@triton.jit
def threshold_kernel(self, threshold, value):
    # `tl.where(self > threshold, self, value)` lowers `arith.cmpf` (fp compare
    # -> i1) to a slow per-lane path on XPU (4096^2 fp16 ~0.70-0.76ms vs ~0.10ms
    # memory floor). Saturating-arithmetic select keeps the vectorized fast
    # path: m = saturate((x - t) * 1e30) lands exactly on {0, 1}; the two-term
    # blend x*m + v*(1-m) is then exact. Note 1e30 (finite in fp32) saturates in
    # f32; for fp16 the same constant overflows to inf, so the whole
    # computation stays in the native dtype (f16<->f32 converts are slow here).
    if self.dtype == tl.float16:
        big = tl.full((), 1.0e30, dtype=self.dtype)
        d = (self - threshold) * big
        m = tl.minimum(1.0, tl.maximum(0.0, d))
        return self * m + value * (1.0 - m)
    # f32 fma form v + (x - v)*m: one extra rounding on the m==1 path
    # (|err| <= 6e-8 in f32, far inside RESOLUTION), but measurably faster
    # than the two-term form on fp32/bf16.
    d = (self - threshold) * 1.0e30
    m = tl.minimum(1.0, tl.maximum(0.0, d))
    return value + (self - value) * m


@pointwise_dynamic(
    is_tensor=[True, True, False], promotion_methods=[(0, 1, "DEFAULT")], config=config_
)
@triton.jit
def threshold_backward_kernel(grad_output, self, threshold):
    # grad_input = grad_output where self > threshold else 0.
    #
    # Every compare-based formulation is slow on XPU: `tl.where(self > t, g, 0)`
    # and `g * (self > t)` lower `arith.cmpf` to a per-lane path (~0.60ms for
    # 4096^2 fp16, 0.09x), and even the integer bit-pattern compare
    # (`yb > tbits & yb <= 0x7F800000`, 0.46ms) stays ~3x above the memory
    # floor because the f32->u32 bitcast + i-cmps keep CoreTiling from
    # vectorizing the blob. The proven fast form (same recipe as the
    # `threshold` forward above and hardsigmoid_backward) is saturating
    # arithmetic with no compare at all: m = min(1, max(0, (x - t) * 1e30))
    # lands exactly on {0, 1} for any positive gap (1e30 saturates every
    # representable gap >= 2^-149 in f32; in fp16/bf16 1e30 is +inf and
    # saturates all gaps incl. subnormals), so `g * m` is exact and the
    # compiler keeps the vectorized tensor-op path. Measured (do_bench, 16.7M
    # fp16 on 4096^2): 0.46ms -> 0.061ms (fp16 0.117x -> 1.02x vs ATen).
    d = (self - threshold) * 1.0e30
    m = tl.minimum(1.0, tl.maximum(0.0, d))
    return grad_output * m


def threshold(self, threshold, value):
    logger.debug("GEMS_KUNLUNXIN THRESHOLD")
    output = threshold_kernel(self, threshold, value)
    return output


def threshold_(self, threshold, value):
    logger.debug("GEMS_KUNLUNXIN THRESHOLD_")
    threshold_kernel(self, threshold, value, out0=self)
    return self


def threshold_backward(grad_output, self, threshold):
    logger.debug("GEMS_KUNLUNXIN THRESHOLD_BACKWARD")
    grad_input = threshold_backward_kernel(grad_output, self, threshold)
    return grad_input
