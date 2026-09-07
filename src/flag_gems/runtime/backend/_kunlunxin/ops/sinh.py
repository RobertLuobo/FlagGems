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

# Keep the same XPU-tuned codegen config as the other hyperbolic siblings
# (asinh_/arcsinh/acosh): buffer_size_limit=4096 + kunlunAutoGrid + unroll_num=8
# route memory through the XPU close-vectorized path.  A/B on [4096,4096] fp16
# shows 0.30 ms here vs 55 ms for the generic flag_gems.utils.pointwise_dynamic
# (which lacks the XPU 12-cluster task-partitioning and the _BAD_TILE_SIZE_1D
# guard); isCloseVectorization=True is 1.7-1.9x faster than the cosh-style
# False variant on the large shapes.
config_ = CodeGenConfig(
    512,
    (65536, 65536, 65536),
    32,
    True,
    prefer_1d_tile=True,
    buffer_size_limit=4096,
    isCloseVectorization=True,
    kunlunAutoGrid=True,
    unroll_num=8,
)


# sinh(x) = (exp(x) - exp(-x)) / 2
# Uses float32 intermediate for numerical precision (matching the generic
# implementation); the formula is exact for the large-value test set
# (|x| <= 100) where exp(x) overflows in both fp16/bf16/fp32 alike.
@pointwise_dynamic(promotion_methods=[(0, "INT_TO_FLOAT")], config=config_)
@triton.jit
def sinh_func(x):
    x32 = x.to(tl.float32)
    return ((tl.exp(x32) - tl.exp(-x32)) * 0.5).to(x.dtype)


def sinh(A):
    logger.debug("GEMS_KUNLUNXIN SINH")
    return sinh_func(A)


def sinh_(A):
    logger.debug("GEMS_KUNLUNXIN SINH_")
    sinh_func(A, out0=A)
    return A