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

import importlib

from flag_gems.utils import pointwise_dynamic

# Use importlib to fetch the real submodules: the flag_gems.ops package binds
# functions named `ldexp`/`ldexp_` in its namespace, which would otherwise
# shadow the submodule attributes of the same name.
_gen = importlib.import_module("flag_gems.ops.ldexp")
_gen_ = importlib.import_module("flag_gems.ops.ldexp_")

logger = logging.getLogger(__name__)

_LN2 = tl.constexpr(0.6931471805599453)
_LN2_FP64 = tl.constexpr(0.6931471805599453094172321214581766)

_P127 = tl.constexpr(2.0**127)
_PM102 = tl.constexpr(2.0**-102)
_P1023 = tl.constexpr(2.0**1023)
_PM969 = tl.constexpr(2.0**-969)

# ---------------------------------------------------------------------------
# XPU3-safe scalbn (2**n scaling by float-exponent bit manipulation).
#
# tl_extra_shim.ldexp lowers to an unsupported libdevice symbol on XPU3
# (`ld.lld: undefined symbol: Unsupported`), and a single `x * tl.exp2(n)`
# overflows/underflows intermediate 2**n. Split-multiply scalbn keeps the
# result finite; sign is reapplied explicitly because XPU3 multiplication
# does not preserve the sign of zero.
# ---------------------------------------------------------------------------


@triton.jit
def _scalbn32(x, n):
    bits = x.to(tl.int32, bitcast=True)
    x_neg = bits < 0
    x_nan = x != x
    mantissa = bits & 0x7FFFFF
    is_sub = ((bits & 0x7F800000) == 0) & (mantissa != 0)
    ax = tl.where(is_sub, mantissa.to(tl.float32), tl.abs(x))
    n = tl.where(is_sub, n - 149, n)
    over1 = n > 127
    ax = tl.where(over1, ax * _P127, ax)
    n = tl.where(over1, n - 127, n)
    over2 = n > 127
    ax = tl.where(over2, ax * _P127, ax)
    n = tl.where(over2, n - 127, n)
    n = tl.where(n > 127, 127, n)
    und1 = n < -126
    ax = tl.where(und1, ax * _PM102, ax)
    n = tl.where(und1, n + 102, n)
    und2 = n < -126
    ax = tl.where(und2, ax * _PM102, ax)
    n = tl.where(und2, n + 102, n)
    n = tl.where(n < -126, -126, n)
    bits = (n + 127) << 23
    scale = bits.to(tl.float32, bitcast=True)
    mag = ax * scale
    signed = tl.where(x_neg, -mag, mag)
    return tl.where(x_nan, x, signed)


@triton.jit
def _scalbn64(x, n):
    bits = x.to(tl.int64, bitcast=True)
    x_neg = bits < 0
    x_nan = x != x
    mantissa = bits & 0xFFFFFFFFFFFFF
    is_sub = ((bits & 0x7FF0000000000000) == 0) & (mantissa != 0)
    n = n.to(tl.int64)
    ax = tl.where(is_sub, mantissa.to(tl.float64), tl.abs(x))
    n = tl.where(is_sub, n - 1074, n)
    over1 = n > 1023
    ax = tl.where(over1, ax * _P1023, ax)
    n = tl.where(over1, n - 1023, n)
    over2 = n > 1023
    ax = tl.where(over2, ax * _P1023, ax)
    n = tl.where(over2, n - 1023, n)
    n = tl.where(n > 1023, 1023, n)
    und1 = n < -1022
    ax = tl.where(und1, ax * _PM969, ax)
    n = tl.where(und1, n + 969, n)
    und2 = n < -1022
    ax = tl.where(und2, ax * _PM969, ax)
    n = tl.where(und2, n + 969, n)
    n = tl.where(n < -1022, -1022, n)
    bits = (n + 1023) << 52
    scale = bits.to(tl.float64, bitcast=True)
    mag = ax * scale
    signed = tl.where(x_neg, -mag, mag)
    return tl.where(x_nan, x, signed)


@triton.jit
def _apply_sign32(res, x):
    x_neg = x.to(tl.int32, bitcast=True) < 0
    res_nan = res != res
    mag = tl.abs(res)
    signed = tl.where(x_neg, -mag, mag)
    return tl.where(res_nan, res, signed)


@triton.jit
def _apply_sign64(res, x):
    x_neg = x.to(tl.int64, bitcast=True) < 0
    res_nan = res != res
    mag = tl.abs(res)
    signed = tl.where(x_neg, -mag, mag)
    return tl.where(res_nan, res, signed)


# ---------------------------------------------------------------------------
# Functional / .out real kernels (module flag_gems.ops.ldexp).
# ---------------------------------------------------------------------------


@pointwise_dynamic(is_tensor=[True, True], promotion_methods=[(0, 1, "INT_TO_FLOAT")])
@triton.jit
def ldexp_func(x, y):
    xf = x.to(tl.float32)
    if y.dtype.is_int() and x.dtype.is_floating():
        exponent = tl.maximum(tl.minimum(y, 2147483647), -2147483648).to(tl.int32)
        return _scalbn32(xf, exponent)
    res = xf * tl.exp(y.to(tl.float32) * _LN2)
    return _apply_sign32(res, xf)


@pointwise_dynamic(is_tensor=[True, True], promotion_methods=[(0, 1, "INT_TO_FLOAT")])
@triton.jit
def ldexp_fp64_func(x, y):
    xf = x.to(tl.float64)
    if y.dtype.is_int() and x.dtype.is_floating():
        exponent = tl.maximum(tl.minimum(y, 2147483647), -2147483648).to(tl.int32)
        return _scalbn64(xf, exponent)
    res = xf * tl.exp(y.to(tl.float64) * _LN2_FP64)
    return _apply_sign64(res, xf)


@pointwise_dynamic(
    is_tensor=[True, True, False], promotion_methods=[(0, 1, "INT_TO_FLOAT")]
)
@triton.jit
def ldexp_integral_func(x, y, wrap_exponent: tl.constexpr):
    if wrap_exponent:
        exponent = y.to(tl.int32)
    else:
        exponent = tl.maximum(tl.minimum(y, 2147483647), -2147483648).to(tl.int32)
    return _scalbn32(x.to(tl.float32), exponent)


@pointwise_dynamic(
    is_tensor=[True, True, False], promotion_methods=[(0, 1, "INT_TO_FLOAT")]
)
@triton.jit
def ldexp_integral_fp64_func(x, y, wrap_exponent: tl.constexpr):
    if wrap_exponent:
        exponent = y.to(tl.int32)
    else:
        exponent = tl.maximum(tl.minimum(y, 2147483647), -2147483648).to(tl.int32)
    return _scalbn64(x.to(tl.float64), exponent)


@pointwise_dynamic(
    is_tensor=[True, True, True, True],
    num_outputs=2,
    promotion_methods=[
        (0, 1, 2, 3, "INT_TO_FLOAT"),
        (0, 1, 2, 3, "INT_TO_FLOAT"),
    ],
)
@triton.jit
def ldexp_complex_func(xr, xi, yr, yi):
    xr = xr.to(tl.float32)
    xi = xi.to(tl.float32)
    yr = yr.to(tl.float32)
    yi = yi.to(tl.float32)
    scale = tl.exp(yr * _LN2)
    angle = yi * _LN2
    cos_angle = tl.cos(angle)
    sin_angle = tl.sin(angle)
    pr = scale * cos_angle
    pi = tl.where(yi == 0, 0.0, scale * sin_angle)
    return xr * pr - xi * pi, xr * pi + xi * pr


@pointwise_dynamic(
    is_tensor=[True, True, True, True],
    num_outputs=2,
    promotion_methods=[
        (0, 1, 2, 3, "INT_TO_FLOAT"),
        (0, 1, 2, 3, "INT_TO_FLOAT"),
    ],
)
@triton.jit
def ldexp_complex_fp64_func(xr, xi, yr, yi):
    xr = xr.to(tl.float64)
    xi = xi.to(tl.float64)
    yr = yr.to(tl.float64)
    yi = yi.to(tl.float64)
    scale = tl.exp(yr * _LN2_FP64)
    angle = yi * _LN2_FP64
    cos_angle = tl.cos(angle)
    sin_angle = tl.sin(angle)
    pr = scale * cos_angle
    pi = tl.where(yi == 0, 0.0, scale * sin_angle)
    return xr * pr - xi * pi, xr * pi + xi * pr


# ---------------------------------------------------------------------------
# In-place kernels (module flag_gems.ops.ldexp_).
# ---------------------------------------------------------------------------


@pointwise_dynamic(promotion_methods=[(0, 1, "DEFAULT")])
@triton.jit
def ldexp_inplace_func(x, other):
    xf = x.to(tl.float32)
    res = xf * tl.exp(other.to(tl.float32) * _LN2)
    return _apply_sign32(res, xf)


@pointwise_dynamic(promotion_methods=[(0, 1, "DEFAULT")])
@triton.jit
def ldexp_inplace_fp64_func(x, other):
    xf = x.to(tl.float64)
    res = xf * tl.exp(other.to(tl.float64) * _LN2_FP64)
    return _apply_sign64(res, xf)


@pointwise_dynamic(is_tensor=[True, True, False], promotion_methods=[(0, 1, "DEFAULT")])
@triton.jit
def ldexp_inplace_integral_func(x, other, wrap_exponent: tl.constexpr):
    if wrap_exponent:
        exponent = other.to(tl.int32)
    else:
        exponent = tl.maximum(tl.minimum(other, 2147483647), -2147483648).to(tl.int32)
    return _scalbn32(x.to(tl.float32), exponent)


@pointwise_dynamic(is_tensor=[True, True, False], promotion_methods=[(0, 1, "DEFAULT")])
@triton.jit
def ldexp_inplace_integral_fp64_func(x, other, wrap_exponent: tl.constexpr):
    if wrap_exponent:
        exponent = other.to(tl.int32)
    else:
        exponent = tl.maximum(tl.minimum(other, 2147483647), -2147483648).to(tl.int32)
    return _scalbn64(x.to(tl.float64), exponent)


@pointwise_dynamic(
    is_tensor=[True, True, True, True],
    num_outputs=2,
    promotion_methods=[
        (0, 1, 2, 3, "INT_TO_FLOAT"),
        (0, 1, 2, 3, "INT_TO_FLOAT"),
    ],
)
@triton.jit
def ldexp_inplace_complex_func(xr, xi, yr, yi):
    xr = xr.to(tl.float32)
    xi = xi.to(tl.float32)
    yr = yr.to(tl.float32)
    yi = yi.to(tl.float32)
    scale = tl.exp(yr * _LN2)
    angle = yi * _LN2
    cos_angle = tl.cos(angle)
    sin_angle = tl.sin(angle)
    pr = scale * cos_angle
    pi = tl.where(yi == 0, 0.0, scale * sin_angle)
    return xr * pr - xi * pi, xr * pi + xi * pr


@pointwise_dynamic(
    is_tensor=[True, True, True, True],
    num_outputs=2,
    promotion_methods=[
        (0, 1, 2, 3, "INT_TO_FLOAT"),
        (0, 1, 2, 3, "INT_TO_FLOAT"),
    ],
)
@triton.jit
def ldexp_inplace_complex_fp64_func(xr, xi, yr, yi):
    xr = xr.to(tl.float64)
    xi = xi.to(tl.float64)
    yr = yr.to(tl.float64)
    yi = yi.to(tl.float64)
    scale = tl.exp(yr * _LN2_FP64)
    angle = yi * _LN2_FP64
    cos_angle = tl.cos(angle)
    sin_angle = tl.sin(angle)
    pr = scale * cos_angle
    pi = tl.where(yi == 0, 0.0, scale * sin_angle)
    return xr * pr - xi * pi, xr * pi + xi * pr


@pointwise_dynamic(
    is_tensor=[True, True, True],
    num_outputs=2,
    promotion_methods=[
        (0, 1, 2, "DEFAULT"),
        (0, 1, 2, "DEFAULT"),
    ],
)
@triton.jit
def ldexp_inplace_complex_integral_func(xr, xi, other):
    exponent = tl.maximum(tl.minimum(other, 2147483647), -2147483648).to(tl.int32)
    ones = (exponent * 0 + 1).to(tl.float32)
    scale = _scalbn32(ones, exponent)
    xr = xr.to(tl.float32)
    xi = xi.to(tl.float32)
    return xr * scale - xi * 0.0, xr * 0.0 + xi * scale


@pointwise_dynamic(
    is_tensor=[True, True, True],
    num_outputs=2,
    promotion_methods=[
        (0, 1, 2, "DEFAULT"),
        (0, 1, 2, "DEFAULT"),
    ],
)
@triton.jit
def ldexp_inplace_complex_integral_fp64_func(xr, xi, other):
    exponent = tl.maximum(tl.minimum(other, 2147483647), -2147483648).to(tl.int32)
    ones = (exponent * 0 + 1).to(tl.float64)
    scale = _scalbn64(ones, exponent)
    xr = xr.to(tl.float64)
    xi = xi.to(tl.float64)
    return xr * scale - xi * 0.0, xr * 0.0 + xi * scale


# ---------------------------------------------------------------------------
# Rebind the XPU3-safe kernels into the generic modules so their Python
# plumbing (_ldexp_impl / ldexp_ / complex helpers) reuses the fixed kernels.
# Names are resolved as module globals at call time, so this takes effect for
# every path (functional, .out, in-place, complex).
# ---------------------------------------------------------------------------

_gen.ldexp_func = ldexp_func
_gen.ldexp_fp64_func = ldexp_fp64_func
_gen.ldexp_integral_func = ldexp_integral_func
_gen.ldexp_integral_fp64_func = ldexp_integral_fp64_func
_gen.ldexp_complex_func = ldexp_complex_func
_gen.ldexp_complex_fp64_func = ldexp_complex_fp64_func

_gen_.ldexp_inplace_func = ldexp_inplace_func
_gen_.ldexp_inplace_fp64_func = ldexp_inplace_fp64_func
_gen_.ldexp_inplace_integral_func = ldexp_inplace_integral_func
_gen_.ldexp_inplace_integral_fp64_func = ldexp_inplace_integral_fp64_func
_gen_.ldexp_inplace_complex_func = ldexp_inplace_complex_func
_gen_.ldexp_inplace_complex_fp64_func = ldexp_inplace_complex_fp64_func
_gen_.ldexp_inplace_complex_integral_func = ldexp_inplace_complex_integral_func
_gen_.ldexp_inplace_complex_integral_fp64_func = ldexp_inplace_complex_integral_fp64_func


def ldexp(self, other):
    logger.debug("GEMS_KUNLUNXIN LDEXP")
    return _gen.ldexp(self, other)


def ldexp_out(self, other, *, out):
    logger.debug("GEMS_KUNLUNXIN LDEXP_OUT")
    return _gen.ldexp_out(self, other, out=out)


def ldexp_(self, other):
    logger.debug("GEMS_KUNLUNXIN LDEXP_")
    return _gen_.ldexp_(self, other)


