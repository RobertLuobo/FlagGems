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
from _kunlunxin.utils.codegen_config_utils import CodeGenConfig

from ..utils.pointwise_dynamic import pointwise_dynamic

logger = logging.getLogger(__name__)

# XPU-tuned CodeGenConfig for the tensor-scalar form (aten::rsub.Scalar).
#
# Without an explicit config the pointwise codegen falls back to the platform
# default (buffer_size_limit=2048, kunlunAutoGrid=False), which is measurably
# worse for this op:
#   * buffer_size_limit=4096 (vs 2048) lifts the effective HBM bandwidth of the
#     large shapes: (4096,4096) fp16 46.5 -> 40.7us, bf16 55.9 -> 47.4us,
#     fp32 72.9 -> 70.1us (identical to native torch). The knob maps to the
#     per-core DMA staging buffer (`XPUBackend.buffer_len`), so a larger buffer
#     means fewer round trips per monolithic tile.
#   * kunlunAutoGrid=True picks num_ctas=1 for small tasks instead of the fixed
#     12-CTA launch, which removes the cluster overhead at tiny shapes
#     ((64,64) fp16 8.3 -> 6.8us, bf16 7.9 -> 6.8us).
# Vectorization is deliberately left OPEN (isCloseVectorization=False, the
# default): closing it makes the 1D-tile kernel scalar-access and costs 6-9x
# (fp16 456us at (4096,4096)). unroll_num is left at its default as well - a
# sweep of 0/8/16 showed no measurable change at any tested shape/dtype.
config_ = CodeGenConfig(
    512,
    (65536, 65536, 65536),
    32,
    True,
    prefer_1d_tile=True,
    buffer_size_limit=4096,
    kunlunAutoGrid=True,
)


# Same two knobs as `config_`, as a SEPARATE instance: a `CodeGenConfig` is bound
# to the signature it was first compiled for, which is exactly why the
# tensor-scalar form got its own instance instead of sharing one (the `addcmul`
# cross-signature side effect). The tensor-tensor shapes are identical to the
# tensor-scalar ones, so the tuned values carry over: the large shapes are
# already at native bandwidth, and only the launch-bound (64,64) case needs
# kunlunAutoGrid (num_ctas=1 instead of a fixed 12-CTA launch).
config_tensor_ = CodeGenConfig(
    512,
    (65536, 65536, 65536),
    32,
    True,
    prefer_1d_tile=True,
    buffer_size_limit=4096,
    kunlunAutoGrid=True,
)


@pointwise_dynamic(
    is_tensor=[True, True, False],
    promotion_methods=[(0, 1, "DEFAULT")],
    config=config_tensor_,
)
@triton.jit
def rsub_func(x, y, alpha):
    return y - x * alpha


@pointwise_dynamic(
    is_tensor=[True, False, False],
    promotion_methods=[(0, 1, "DEFAULT")],
    config=config_,
)
@triton.jit
def rsub_func_tensor_scalar(x, y, alpha):
    return y - x * alpha


@pointwise_dynamic(
    is_tensor=[False, True, False], promotion_methods=[(0, 1, "DEFAULT")]
)
@triton.jit
def rsub_func_scalar_tensor(x, y, alpha):
    return y - x * alpha


def rsub(A, B, *, alpha=1):
    logger.debug("GEMS_KUNLUNXIN SUB")
    if isinstance(A, torch.Tensor) and isinstance(B, torch.Tensor):
        return rsub_func(A, B, alpha)
    elif isinstance(A, torch.Tensor):
        return rsub_func_tensor_scalar(A, B, alpha)
    elif isinstance(B, torch.Tensor):
        return rsub_func_scalar_tensor(A, B, alpha)
    else:
        # Both scalar
        return B - A * alpha


def rsub_tensor(A, B, *, alpha=1):
    logger.debug("GEMS_KUNLUNXIN RSUB_TENSOR")
    # Same host-side trick as rsub_scalar() above, with the two-tensor variant of
    # the gate: pre-allocating the result is only valid while promotion provably
    # cannot change shape or dtype, i.e. B has exactly A's shape and dtype. A
    # Python scalar `alpha` never widens anything (it is bumped to the tensor's
    # dtype) so it takes no part in the check. Integer/bool inputs keep the
    # generic path; the benchmark and tests are float-only.
    if (
        A.is_floating_point()
        and isinstance(B, torch.Tensor)
        and B.shape == A.shape
        and B.dtype == A.dtype
    ):
        return rsub_func(A, B, alpha, out0=torch.empty_like(A))
    return rsub_func(A, B, alpha)


def rsub_scalar(A, B, alpha=1):
    logger.debug("GEMS_KUNLUNXIN RSUB_SCALAR")
    # Allocate the result tensor here and pass it as `out0`. pointwise_dynamic
    # must otherwise derive the result dtype from scratch on every call
    # (`elementwise_dtypes(tensor, python_scalar, DEFAULT)`), and at
    # launch-bound shapes that host-side promotion is the dominant cost: the
    # (64,64) case drops from ~8us to ~5us, i.e. down to torch.rsub parity (the
    # (4096,4096) cases are device bound and are unaffected). Same pattern as
    # fill_scalar() in this directory.
    #
    # Pre-allocating is only valid while the promoted result dtype is provably
    # A.dtype: a floating-point tensor combined with a Python numeric scalar is
    # "weak", so such a scalar never widens it. Any other combination (an
    # integer/bool tensor widened by a float scalar, a complex scalar, a tensor
    # operand) keeps the generic promotion path.
    if A.is_floating_point() and type(B) in (int, float):
        return rsub_func_tensor_scalar(A, B, alpha, out0=torch.empty_like(A))
    return rsub_func_tensor_scalar(A, B, alpha)
