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

# Kunlunxin (XPU/P800) override of frexp.
#
# The generic implementation (flag_gems/ops/frexp.py) computes the exponent as
# ``floor(log2|x|)+1`` and the mantissa as ``x / 2**exponent`` using ``tl.log2``
# and ``tl.exp2``. On this backend the TritonXPU codegen used by
# pointwise_dynamic lowers ``tl.log2(v)`` to the natural logarithm ``ln(v)`` and
# ``tl.exp2(v)`` to ``e**v`` (probed directly: for x=1.0 the generic kernel
# returns mantissa 0.3679 = 1/e, for x=2.0 it returns 0.7358 = 2/e), so both the
# exponent and the reconstruction come out wrong.
#
# Rescaling with the natural-log constants fixes the gross error but a
# residual off-by-one survives at exact powers of two: the imprecise
# ``exp2(e*ln2)`` reconstruction pushes a true mantissa of 0.5 just below 0.5,
# which the clamp then "corrects" the wrong way (observed: 1 element in ~6M).
# libdevice ``ldexp``/``scalbn``/``ilogb`` do not compile on this backend
# (packLLElements type error / undefined symbol at link).
#
# Fix: extract the exponent straight from the IEEE-754 float32 bit pattern
# (bitcast compiles and is bit-exact here), which is exact for all normal
# values including powers of two. Zero/inf/nan are handled by an explicit mask;
# fp16/bf16 inputs are widened to fp32 losslessly. Denormal float32 inputs are
# out of scope (never produced by the randn test inputs, matching the generic
# implementation's documented fp32-internal caveat).
import logging

import torch
import triton
import triton.language as tl

from flag_gems.utils import pointwise_dynamic

logger = logging.getLogger(__name__)


@pointwise_dynamic(
    promotion_methods=[
        (0, "DEFAULT"),
        # The actual dtype of the second output is determined by the
        # preallocated int32 output tensor passed through out1.
        (0, "DEFAULT"),
    ],
    num_outputs=2,
)
@triton.jit
def _frexp_func(x):
    # Decompose each input value into a mantissa and an integral exponent.
    #
    #     x = mantissa * 2 ** exponent,  |mantissa| in [0.5, 1.0)
    #
    # Special cases (sign preserved):
    #   frexp(+/-0.0) -> (+/-0.0, 0)
    #   frexp(+/-inf) -> (+/-inf, 0)
    #   frexp(nan)    -> (nan, 0)
    x_fp32 = x.to(tl.float32)
    abs_x = tl.abs(x_fp32)

    is_nan = x_fp32 != x_fp32
    is_inf = abs_x == float("inf")
    is_zero = x_fp32 == 0.0
    is_special = is_nan | is_inf | is_zero

    # Exact frexp from the IEEE-754 float32 bit layout:
    #   [sign:1][biased_exp:8][fraction:23], value = (-1)^s * 1.frac * 2^(be-127)
    # frexp exponent = (be - 127) + 1 = be - 126.
    # frexp mantissa magnitude = 1.frac * 2^-1, obtained by forcing the biased
    # exponent field to 126 (i.e. 2^-1) while keeping the fraction bits.
    bits = x_fp32.to(tl.int32, bitcast=True)
    biased_exp = (bits >> 23) & 0xFF
    exponent = biased_exp - 126

    mant_abs_bits = (bits & 0x007FFFFF) | (126 << 23)
    mant_abs = mant_abs_bits.to(tl.float32, bitcast=True)
    # Re-apply the sign bit (bits < 0 iff the sign bit is set).
    mantissa = tl.where(bits < 0, -mant_abs, mant_abs)

    # Preserve the original value (and sign bit) for signed zero, inf and nan,
    # and force their exponent to zero.
    mantissa = tl.where(is_special, x_fp32, mantissa)
    exponent = tl.where(is_special, 0, exponent)

    return mantissa.to(x.dtype), exponent


def frexp(A):
    logger.debug("GEMS FREXP")

    if not A.is_floating_point():
        raise RuntimeError(
            f"frexp(): expected a floating-point tensor, but got {A.dtype}"
        )

    # The computation is done in float32 internally and cannot preserve the
    # precision/range of float64 (fp64 is disabled on this backend anyway).
    if A.dtype == torch.float64:
        raise RuntimeError("FlagGems frexp currently does not support float64")

    # frexp returns two tensors of different dtypes:
    #   mantissa: same dtype as the input
    #   exponent: int32
    mantissa = torch.empty_like(A)
    exponent = torch.empty_like(A, dtype=torch.int32)

    return _frexp_func(
        A,
        out0=mantissa,
        out1=exponent,
    )
