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
    isCloseVectorization=True,
    kunlunAutoGrid=True,
    unroll_num=8,
)

# Two launch variants for the non-out special_log1p (probe-verified in the
# sibling log.py): single-CTA (kunlunAutoGrid=True, 1 CTA) is best for small
# / very large tiles, multi-CTA (kunlunAutoGrid=False, 12 CTAs) is best for
# the 8K..128K-element mid range where the single giant tile serializes.
# The XPU pointwise grid strategy has a per-shape optimum; the wrapper picks
# the variant by numel (same window as log.py). The kernel body and the
# isCloseVectorization=True (log-family correctness requirement) are identical.
config_single_cta = CodeGenConfig(
    512,
    (65536, 65536, 65536),
    32,
    True,
    prefer_1d_tile=True,
    buffer_size_limit=8192,
    isCloseVectorization=True,
    kunlunAutoGrid=True,
    unroll_num=16,
)
config_multi_cta = CodeGenConfig(
    512,
    (65536, 65536, 65536),
    32,
    True,
    prefer_1d_tile=True,
    buffer_size_limit=8192,
    isCloseVectorization=True,
    kunlunAutoGrid=False,
    unroll_num=16,
)


@pointwise_dynamic(promotion_methods=[(0, "INT_TO_FLOAT")], config=config_single_cta)
@triton.jit
def special_log1p_func_single(x):
    return tl.log(1.0 + x.to(tl.float32)).to(x.dtype)


# The two scalar fns must have distinct source text so the pointwise codegen
# cache keys do not collide inside one process (two wrappers, same pid).
@pointwise_dynamic(promotion_methods=[(0, "INT_TO_FLOAT")], config=config_multi_cta)
@triton.jit
def special_log1p_func_multi(x):
    return tl.log(1.0000000000000000 + x.to(tl.float32)).to(x.dtype)


# 12-CTA mode is beneficial only inside this numel window (probe: 2026-08-18,
# see log.py); outside it the single-CTA mode is at least as fast.
_MULTI_CTA_MIN_NUMEL = 8192
_MULTI_CTA_MAX_NUMEL = 131072


# non-out special_log1p previously fell to the generic bare pointwise_dynamic
# (no CodeGenConfig) -> discrete access on XPU -> catastrophic latency
# (~40ms on [4096,4096], gems speedup ~0.004; same failure class as log1p
# before its override). Mirror the proven log.py/log1p.py recipe: the tuned
# CodeGenConfig (prefer_1d_tile, isCloseVectorization=True) on a
# pointwise_dynamic kernel. Kernel body (tl.log(1 + x_fp32)) and precision
# are unchanged.
def special_log1p(A):
    logger.debug("GEMS_KUNLUNXIN SPECIAL_LOG1P")
    numel = A.numel()
    if _MULTI_CTA_MIN_NUMEL < numel <= _MULTI_CTA_MAX_NUMEL:
        return special_log1p_func_multi(A)
    return special_log1p_func_single(A)


# out variant (special_log1p.out): same dual-variant recipe as the non-out
# path above.  Before this change the out kernel ran the older single config_
# (buffer_size_limit=4096, unroll_num=8) only, which measured 0.40-0.89 on the
# 8K..128K-element mid range (single-CTA stall, e.g. (1024,16)/(64,64,16)) and
# left ~15-25us on the table for the 16M-element fp32 shapes (8192/16 DMA
# staging vs 4096/8).  The two scalar fns keep distinct source text so the
# pointwise codegen cache keys do not collide with each other / the non-out
# pair inside one process (same pid).
@pointwise_dynamic(promotion_methods=[(0, "INT_TO_FLOAT")], config=config_single_cta)
@triton.jit
def special_log1p_out_func(x):
    return tl.log(1.0 + x.to(tl.float32)).to(x.dtype)


@pointwise_dynamic(promotion_methods=[(0, "INT_TO_FLOAT")], config=config_multi_cta)
@triton.jit
def special_log1p_out_func_multi(x):
    return tl.log(1.0000000000000000 + x.to(tl.float32)).to(x.dtype)


def special_log1p_out(A, out):
    logger.debug("GEMS_KUNLUNXIN SPECIAL_LOG1P_OUT")
    numel = A.numel()
    if _MULTI_CTA_MIN_NUMEL < numel <= _MULTI_CTA_MAX_NUMEL:
        return special_log1p_out_func_multi(A, out0=out)
    return special_log1p_out_func(A, out0=out)
