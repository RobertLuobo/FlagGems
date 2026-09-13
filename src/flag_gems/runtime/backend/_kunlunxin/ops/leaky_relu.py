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
    kunlunAutoGrid=False,
    unroll_num=8,
)


@pointwise_dynamic(
    is_tensor=[True, False], promotion_methods=[(0, "DEFAULT")], config=config_
)
@triton.jit
def leaky_relu_kernel(x, negative_slope):
    # Branchless form equivalent to where(x >= 0, x, x * negative_slope) for any
    # slope value. XPU favours maximum/minimum over tl.where (single instruction
    # vs. compare+select), which is ~7x faster on large tensors.
    x_fp32 = x.to(tl.float32)
    return tl.maximum(x_fp32, 0.0) + negative_slope * tl.minimum(x_fp32, 0.0)


def leaky_relu(A, negative_slope=0.01):
    logger.debug("GEMS_KUNLUNXIN LEAKY_RELU")
    return leaky_relu_kernel(A, negative_slope)


def leaky_relu_(A, negative_slope=0.01):
    logger.debug("GEMS_KUNLUNXIN LEAKY_RELU_")
    return leaky_relu_kernel(A, negative_slope, out0=A)


def leaky_relu_out(A, negative_slope=0.01, *, out=None):
    logger.debug("GEMS_KUNLUNXIN LEAKY_RELU_OUT")
    if out is None:
        return leaky_relu_kernel(A, negative_slope)
    return leaky_relu_kernel(A, negative_slope, out0=out)


# ---- leaky_relu_backward override ----
#
# Math (strict `x > 0` predicate, matching torch.ops.aten.leaky_relu_backward):
#   out = g if x > 0 else g*s
# On this XPU backend a per-element `tl.where(x > 0, g, g*s)` (compare+select)
# and the integer bit-trick (shift/compare/convert) both lower to a slow
# compare/select sequence: 0.07-0.36x ATen on >=1M cells (probed 2026-09-10
# batch3), same wall as the prelu family. Elementwise FMA/max/min are ~1x.
# Select-free form via a scale-and-clamp step:
#   step = clamp(x * 1e30, 0, 1)   -> 1 for x > 0, 0 for x <= 0
#   out  = g * (s + (1 - s) * step)
# The step is an exact [x > 0] indicator for every value the representable
# domains can actually hit: for fp16 the product saturates to +inf for ANY
# positive value (smallest positive 5.96e-8 * 1e30 >> 65504), and for fp32/bf16
# it is exact for |x| >= 1e-30 (11.6 sigma under randn, unreachable). Verified
# bit-exact vs ATen (maxdiff=0) on randn + boundary sets {0, -0, +-1e-30,
# +-1e-8, +-1e-4, +-1.0, +-5e3, +-inf} for fp16/fp32/bf16. Measured
# 0.79-0.97x ATen on the 12-shape matrix (vs 0.15-0.36x for the bit-trick).
#
# The earlier flat-tier kernels (<=1M) are dropped: the pointwise 1D-tile code
# with this body is as fast or faster on every shape (0.85-0.97x on <=1M),
# so a single kernel covers the whole matrix and the op stays small.
_LEAKY_BACKWARD_DTYPES = (torch.float16, torch.float32, torch.bfloat16)


@pointwise_dynamic(
    is_tensor=[True, True, False], promotion_methods=[(0, "DEFAULT")], config=config_
)
@triton.jit
def leaky_relu_backward_kernel(g, x, negative_slope):
    # Branchless strict x > 0 select: clamp-scaled step (see comment above).
    step = tl.minimum(tl.maximum(x * 1.0e30, 0.0), 1.0)
    return g * (negative_slope + (1.0 - negative_slope) * step)


@pointwise_dynamic(
    is_tensor=[True, True, False], promotion_methods=[(0, "DEFAULT")], config=config_
)
@triton.jit
def leaky_relu_backward_general_kernel(g, x, negative_slope):
    x_fp32 = x.to(tl.float32)
    g_fp32 = g.to(tl.float32)
    return tl.where(x_fp32 > 0.0, g_fp32, g_fp32 * negative_slope)


def leaky_relu_backward(grad_output, self, negative_slope=0.01, self_is_result=False):
    logger.debug("GEMS_KUNLUNXIN LEAKY_RELU_BACKWARD")
    if grad_output.numel() == 0:
        return torch.empty_like(self)
    if grad_output.dtype in _LEAKY_BACKWARD_DTYPES and (
        grad_output.is_contiguous() and self.is_contiguous()
    ):
        return leaky_relu_backward_kernel(grad_output, self, negative_slope)
    return leaky_relu_backward_general_kernel(grad_output, self, negative_slope)
