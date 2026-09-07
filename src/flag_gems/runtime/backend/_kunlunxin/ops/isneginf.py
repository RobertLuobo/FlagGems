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

from ..utils.codegen_config_utils import CodeGenConfig
from ..utils.pointwise_dynamic import pointwise_dynamic

logger = logging.getLogger(__name__)

# A direct comparison with -inf has the same floating-point semantics as
# isinf(x) & (x < 0), but avoids the libdevice isinf extern call.  The
# fp32-widened `== -inf` compare itself still measures ~0.37-1.50 ms on
# numel > 2^16 on this backend; the pure integer bit-pattern equality below
# (x == -inf <=> IEEE bits == {sign=1, exp=all-ones, mant=0}) measures
# ~3.3-3.8x faster for fp16/fp32 (probe see harness/solution/isneginf/
# kunlunxin_performance_fix_20260904.md) while staying exact for every float
# bit pattern (NaN / +inf / +-0 / subnormal all differ from the single -inf
# pattern).
# bf16 must keep the fp32 compare: its integer bit-pattern variant needs an
# fp32 widen whose uint32 compare measures neutral-to-slower, and direct
# int16/uint16 bitcast fails the XPU TritonXPUDtypeConvert pass in the
# vectorized path (same known limitation as the isposinf/signbit families).
_config = CodeGenConfig(
    512,
    (65536, 65536, 65536),
    32,
    True,
    prefer_1d_tile=True,
    isCloseMemoryAsync=False,
    kunlunAutoGrid=True,
    unroll_num=16,
)


@triton.jit
def _isneginf_body(x):
    if tl.constexpr(x.dtype.is_fp32()):
        xi = x.to(tl.int32, bitcast=True)
        return xi == -(1 << 23)  # 0xFF800000 as signed int32 (-inf)
    elif tl.constexpr(x.dtype.is_fp16()):
        xi = x.to(tl.int16, bitcast=True)
        return xi == -(1 << 10)  # 0xFC00 as signed int16 (-inf)
    elif tl.constexpr(x.dtype.is_bf16()):
        return x.to(tl.float32) == -float("inf")
    elif tl.constexpr(x.dtype.is_fp64()):
        xi = x.to(tl.int64, bitcast=True)
        return xi == -(1 << 52)  # 0xFFF0000000000000 as signed int64 (-inf)
    else:
        # integers / other dtypes: widen and compare (finite -> False)
        return x.to(tl.float32) == -float("inf")


@pointwise_dynamic(promotion_methods=[(0, "ALWAYS_BOOL")], config=_config)
@triton.jit
def isneginf_func(x):
    return _isneginf_body(x)


def isneginf(A):
    logger.debug("GEMS_KUNLUNXIN ISNEGINF")
    return isneginf_func(A)


def isneginf_out(A, *, out=None):
    logger.debug("GEMS_KUNLUNXIN ISNEGINF_OUT")
    if out is None:
        return isneginf_func(A)
    isneginf_func(A, out0=out)
    return out
