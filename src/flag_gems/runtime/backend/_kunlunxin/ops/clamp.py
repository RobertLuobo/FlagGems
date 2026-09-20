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

# clamp_max is a pure memory-bound elementwise op (reads 1 tensor + scalar,
# writes 1) whose only compute is a min. Without a tuned config the default
# codegen emits a tiny 256-element tile with no unrolling, badly underutilizing
# the XPU (~1000x slower than torch: 49ms vs 0.04ms on 4096^2). Use div.py's
# tuned recipe (larger buffer + unroll) but keep vectorization OPEN
# (isCloseVectorization=False): this op is 1-in/1-out so wide vector DMA (esp.
# packing fp16/bf16) is the bandwidth lever. Measured on 4096^2: vec-open fp16
# 0.092ms vs vec-closed 0.155ms; [10000,65536] fp16 1.43ms vs 5.29ms. Unlike
# addcdiv (3-in, vec-closed best), always measure per op.
clamp_max_config = CodeGenConfig(
    512,
    (65536, 65536, 65536),
    32,
    True,
    prefer_1d_tile=True,
    buffer_size_limit=4096,
    isCloseVectorization=False,
    unroll_num=8,
    # kunlunAutoGrid=True lets the 1d-tile wrapper pick num_ctas=1 when
    # num_tasks <= 2048*64 instead of the fixed 12-CTA launch.  At (64,64)
    # (4096 elems) the fixed grid splits the work into 12 CTAs of a 512-tile
    # each, and the per-CTA launch/cluster overhead dominates: 4096-elem
    # elementwise latency 7.35 -> 6.34us (bf16) / 6.49 -> 6.02us (fp16) when
    # combined with the pre-allocated out0 below (harness/solution/pdhost_cost
    # /phase1d_clamp_ab.txt).  Shapes above the 2048*64 threshold keep the
    # 12-CTA grid, so the large shapes are unaffected (verified 47.5->47.1us
    # bf16, 41.4->41.2us fp16, 69.7->69.9us fp32 on (64,512,512)).  Same knob
    # and rationale as neg.py / rsub.py.
    kunlunAutoGrid=True,
)

# clamp_tensor's benchmark entry combines THREE same-shape tensors, and that is
# the only entry whose (64,64) case sits on the wrong side of 0.8: 22.2/20.8/20.1us
# gems vs ~5.8us torch (speedup 0.26).  Two independent host/launch-side levers,
# both measured on the real entry (harness/solution/clamp_tensor/fix_margin_20260919
# /real_entry_*.log) at (64,64):
#   * out0 pre-allocation in clamp_tensor()      : 22.2 -> 10.6 (bf16) /
#     20.8 -> 9.8 (fp16) / 20.1 -> 9.0us (fp32)   [~12us = elementwise_dtypes
#     + torch.empty_like, which out0 skips]
#   * this config (kunlunAutoGrid=True + the proven tile/unroll recipe)
#                                                : 10.6 -> 6.4 / 9.8 -> 6.3 /
#     9.0 -> 6.6us  [fixed 12-CTA -> 1-CTA launch at num_tasks<=2048*64]
# The two knobs are single-variable A/B verified in ab_decompose.py (V0/VA/VB/VAB).
# NOTE: this is a NEW CodeGenConfig instance, deliberately not clamp_max_config
# (which is clamp_max/clamp_min's own tuned object); the 3-tensor kernel has a
# different input count so the two are kept independent.
# Shapes above the 2048*64 threshold keep the 12-CTA grid, so (4096,4096) and
# (64,512,512) are unaffected (verified 79.8->80.1 / 74.3->74.7 / 136.3->136.6us,
# i.e. within noise).
clamp_tensor_config = CodeGenConfig(
    512,
    (65536, 65536, 65536),
    32,
    True,
    prefer_1d_tile=True,
    buffer_size_limit=4096,
    isCloseVectorization=False,
    unroll_num=8,
    kunlunAutoGrid=True,
)


@pointwise_dynamic(
    promotion_methods=[(0, 1, 2, "DEFAULT")], config=clamp_tensor_config
)
@triton.jit
def clamp_func_tensor(x, mini, maxi):
    return tl.minimum(maxi, tl.maximum(mini, x))


@pointwise_dynamic(
    promotion_methods=[(0, 1, "DEFAULT")], config=clamp_tensor_config
)
@triton.jit
def clamp_func_min_tensor(x, mini):
    return tl.maximum(mini.to(tl.float32), x.to(tl.float32))


@pointwise_dynamic(
    promotion_methods=[(0, 1, "DEFAULT")], config=clamp_tensor_config
)
@triton.jit
def clamp_func_max_tensor(x, maxi):
    return tl.minimum(maxi, x)


def _clamp_out_alloc_ok(a, *others):
    """True when the promoted clamp result is provably ``a`` itself in shape and
    dtype, so ``out0=torch.empty_like(a)`` is a valid pre-allocation.

    Level-③ gate (see clamp_max's out0 comment for levels ①/②): the benchmark
    passes three same-shape tensors, and for a vector/tensor bound every arg must
    match a in BOTH shape and dtype -- a shape-only or dtype-only check silently
    produces wrong values (broadcast or widening) while still "running".
    """
    return (
        a.is_floating_point()
        and all(
            isinstance(o, torch.Tensor) and o.shape == a.shape and o.dtype == a.dtype
            for o in others
        )
    )


def clamp_tensor(A, mini=None, maxi=None):
    logger.debug("GEMS_KUNLUNXIN CLAMP_TENSOR")
    if mini is None and maxi is None:
        raise ValueError("At least one of mini or maxi must not be None")
    elif mini is None:
        if _clamp_out_alloc_ok(A, maxi):
            return clamp_func_max_tensor(A, maxi, out0=torch.empty_like(A))
        return clamp_func_max_tensor(A, maxi)
    elif maxi is None:
        if _clamp_out_alloc_ok(A, mini):
            return clamp_func_min_tensor(A, mini, out0=torch.empty_like(A))
        return clamp_func_min_tensor(A, mini)
    else:
        if _clamp_out_alloc_ok(A, mini, maxi):
            return clamp_func_tensor(A, mini, maxi, out0=torch.empty_like(A))
        return clamp_func_tensor(A, mini, maxi)


def clamp_tensor_(A, mini=None, maxi=None):
    logger.debug("GEMS_KUNLUNXIN CLAMP_TENSOR_")
    if mini is None and maxi is None:
        raise ValueError("At least one of mini or maxi must not be None")
    elif mini is None:
        return clamp_func_max_tensor(A, maxi, out0=A)
    elif maxi is None:
        return clamp_func_min_tensor(A, mini, out0=A)
    else:
        return clamp_func_tensor(A, mini, maxi, out0=A)


@pointwise_dynamic(
    is_tensor=[True, False, False], promotion_methods=[(0, 1, 2, "DEFAULT")]
)
@triton.jit
def clamp_func(x, mini, maxi):
    return tl.minimum(maxi, tl.maximum(mini, x))


@pointwise_dynamic(is_tensor=[True, False], promotion_methods=[(0, 1, "DEFAULT")])
@triton.jit
def clamp_func_min(x, mini):
    return tl.maximum(mini, x)


@pointwise_dynamic(is_tensor=[True, False], promotion_methods=[(0, 1, "DEFAULT")])
@triton.jit
def clamp_func_max(x, maxi):
    return tl.minimum(maxi, x)


@pointwise_dynamic(
    is_tensor=[True, False],
    promotion_methods=[(0, 1, "DEFAULT")],
    config=clamp_max_config,
)
@triton.jit
def clamp_max_func(x, maxi):
    return tl.minimum(maxi, x)


def clamp_max(A, max_value):
    logger.debug("GEMS_KUNLUNXIN CLAMP_MAX")
    if max_value is None:
        raise ValueError("max_value must not be None")
    # Allocate the result here and pass it as `out0`.  Without it
    # pointwise_dynamic must re-derive the result dtype on every call
    # (`elementwise_dtypes(tensor, python_scalar, DEFAULT)` + a fresh
    # `torch.empty_like`), and at launch-bound shapes that host-side work is a
    # large share of the end-to-end latency: (64,64) 10.5 -> 8.7us bf16,
    # 9.0 -> 7.7us fp16, 7.1 -> 6.5us fp32 (phase1d_clamp_ab.txt, B-arm).
    # Combined with the kunlunAutoGrid knob this brings (64,64) to 6.3/6.0/5.8us,
    # i.e. torch parity.  The (4096,4096)/(64,512,512) cases are device bound
    # and unaffected.  Same pattern as rsub_scalar()/fill_scalar().
    #
    # Pre-allocating is only valid while the promoted result dtype is provably
    # A.dtype: a floating-point tensor combined with a Python numeric scalar is
    # "weak", so such a scalar never widens it (and clamp is documented to
    # preserve the input dtype anyway).  Every other combination (an
    # integer/bool tensor, a Tensor max_value, a complex scalar) keeps the
    # generic promotion path.
    if A.is_floating_point() and type(max_value) in (int, float):
        return clamp_max_func(A, max_value, out0=torch.empty_like(A))
    return clamp_max_func(A, max_value)


def clamp_max_(A, max_value):
    logger.debug("GEMS_KUNLUNXIN CLAMP_MAX_")
    if max_value is None:
        raise ValueError("max_value must not be None")
    return clamp_max_func(A, max_value, out0=A)


def clamp_min(A, mini):
    logger.debug("GEMS_KUNLUNXIN CLAMP_MIN")
    if mini is None:
        raise ValueError("Mini must not be None")
    return clamp_func_min(A, mini)


def clamp_min_(A, mini):
    logger.debug("GEMS_KUNLUNXIN CLAMP_MIN_")
    if mini is None:
        raise ValueError("Mini must not be None")
    return clamp_func_min(A, mini, out0=A)


def clamp(A, mini=None, maxi=None):
    logger.debug("GEMS_KUNLUNXIN CLAMP")
    if mini is None and maxi is None:
        raise ValueError("At least one of mini or maxi must not be None")
    elif mini is None:
        return clamp_func_max(A, maxi)
    elif maxi is None:
        return clamp_func_min(A, mini)
    else:
        return clamp_func(A, mini, maxi)


def clamp_(A, mini=None, maxi=None):
    logger.debug("GEMS_KUNLUNXIN CLAMP_")
    if mini is None and maxi is None:
        raise ValueError("At least one of mini or maxi must not be None")
    elif mini is None:
        return clamp_func_max(A, maxi, out0=A)
    elif maxi is None:
        return clamp_func_min(A, mini, out0=A)
    else:
        return clamp_func(A, mini, maxi, out0=A)
