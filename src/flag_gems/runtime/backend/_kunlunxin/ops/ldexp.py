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

from flag_gems.ops.ldexp import (
    _WRAP_INTEGER_EXPONENT,
    _complex_parts,
    _copy_result,
    _result_device,
    _result_dtype,
    _validate_out,
)
from flag_gems.ops.ldexp_ import _is_exact_alias
from flag_gems.ops.ldexp_ import _result_dtype as _inplace_result_dtype
from flag_gems.ops.ldexp_ import _write_complex
from flag_gems.utils import pointwise_dynamic
from flag_gems.utils.shape_utils import MemOverlap, has_internal_overlapping

logger = logging.getLogger(__name__)

_LN2 = tl.constexpr(0.6931471805599453)


@triton.jit
def _scalbn_f32(x, n):
    # x * 2**n via IEEE-754 exponent-field manipulation.
    # XPU libdevice ldexp/scalbn resolve to an "Unsupported" symbol, and the
    # naive x * exp2(n) overflows for large |n| and loses the sign of zero.
    # The hardware also flushes subnormal operands to zero in every float ALU
    # op (even x * 1.0), so subnormal inputs must be normalised purely in the
    # integer bit domain -- no float multiply anywhere in this routine.
    xi = x.to(tl.uint32, bitcast=True)
    n = tl.minimum(tl.maximum(n.to(tl.int32), -1024), 1024)  # saturate; avoid int32 overflow in new_e
    sign = xi & 0x80000000
    absb = xi & 0x7FFFFFFF
    exp = (absb >> 23).to(tl.int32)
    man = absb & 0x7FFFFF
    is_zero = absb == 0
    is_special = exp == 0xFF  # inf / nan -> pass through unchanged
    is_sub = (exp == 0) & (man != 0)
    # Bit-domain highest-set-bit index b of the subnormal mantissa (0..22).
    v = man
    b = tl.zeros_like(man)
    t = tl.where(v >= (1 << 16), 16, 0)
    v = v >> t
    b = b + t
    t = tl.where(v >= (1 << 8), 8, 0)
    v = v >> t
    b = b + t
    t = tl.where(v >= (1 << 4), 4, 0)
    v = v >> t
    b = b + t
    t = tl.where(v >= (1 << 2), 2, 0)
    v = v >> t
    b = b + t
    t = tl.where(v >= (1 << 1), 1, 0)
    b = b + t
    sh = 23 - b.to(tl.int32)
    sub_norm_man = (man << sh.to(tl.uint32)) & 0x7FFFFF
    sub_exp = 1 - sh  # biased exponent of the normalised subnormal input
    # Unified normal/subnormal operands.
    exp_before = tl.where(is_sub, sub_exp, exp)
    man_before = tl.where(is_sub, sub_norm_man, man)
    new_e = exp_before + n.to(tl.int32)
    normal_bits = sign | (new_e.to(tl.uint32) << 23) | man_before
    inf_bits = sign | 0x7F800000
    shift = (1 - new_e).to(tl.uint32)
    full = man_before | 0x800000
    sub_man = tl.where((1 - new_e) < 32, full >> shift, 0)
    sub_bits = sign | sub_man
    out_bits = tl.where(
        new_e >= 0xFF,
        inf_bits,
        tl.where(new_e >= 1, normal_bits, sub_bits),
    )
    out = out_bits.to(tl.float32, bitcast=True)
    return tl.where(is_special | is_zero, x, out)


@triton.jit
def _reapply_sign_f32(mag, src):
    # mag is a non-negative magnitude (or NaN); copy the sign bit of src.
    sgn = src.to(tl.uint32, bitcast=True) & 0x80000000
    body = mag.to(tl.uint32, bitcast=True) & 0x7FFFFFFF
    return (sgn | body).to(tl.float32, bitcast=True)


@pointwise_dynamic(is_tensor=[True, True], promotion_methods=[(0, 1, "INT_TO_FLOAT")])
@triton.jit
def ldexp_func(x, y):
    # Float (possibly non-integer) exponent path: 2**y == exp(y*ln2).
    # tl.exp2 is miscompiled on XPU (returns e**x), so use tl.exp.
    xf = x.to(tl.float32)
    scale = tl.exp(y.to(tl.float32) * _LN2)
    mag = tl.abs(xf) * scale
    return _reapply_sign_f32(mag, xf)


@pointwise_dynamic(
    is_tensor=[True, True, False], promotion_methods=[(0, 1, "INT_TO_FLOAT")]
)
@triton.jit
def ldexp_integral_func(x, y, wrap_exponent: tl.constexpr):
    exponent = y.to(tl.int32)
    if not wrap_exponent:
        exponent = tl.maximum(tl.minimum(y, 2147483647), -2147483648).to(tl.int32)
    return _scalbn_f32(x.to(tl.float32), exponent)


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
    pi = tl.where(yi == 0, 0, scale * sin_angle)
    return xr * pr - xi * pi, xr * pi + xi * pr


@pointwise_dynamic(
    is_tensor=[True, True, True, False],
    num_outputs=2,
    promotion_methods=[
        (0, 1, 2, "DEFAULT"),
        (0, 1, 2, "DEFAULT"),
    ],
)
@triton.jit
def ldexp_complex_integral_func(xr, xi, exponent, wrap_exponent: tl.constexpr):
    e = exponent.to(tl.int32)
    if not wrap_exponent:
        e = tl.maximum(tl.minimum(exponent, 2147483647), -2147483648).to(tl.int32)
    return _scalbn_f32(xr.to(tl.float32), e), _scalbn_f32(xi.to(tl.float32), e)


def _complex_ldexp(self, other, result_dtype):
    real_dtype = torch.float64 if result_dtype == torch.complex128 else torch.float32
    xr, xi = _complex_parts(self, real_dtype)
    yr, yi = _complex_parts(other, real_dtype)
    real, imag = ldexp_complex_func(xr, xi, yr, yi)
    result = torch.empty(real.shape, dtype=result_dtype, device=real.device)
    parts = torch.view_as_real(result)
    from flag_gems.ops.copy import copy_

    copy_(parts[..., 0], real)
    copy_(parts[..., 1], imag)
    return result


def _ldexp_impl(self, other, out=None):
    result_dtype = _result_dtype(self, other)
    if out is not None:
        _validate_out(self, other, out, result_dtype)
    device = _result_device(self, other)
    if self.device != device:
        self = torch.tensor(self.item(), dtype=self.dtype, device=device)
    if other.device != device:
        other = torch.tensor(other.item(), dtype=other.dtype, device=device)
    if self.is_complex() or other.is_complex():
        result = _complex_ldexp(self, other, result_dtype)
        if out is None:
            return result
        out.resize_(result.shape)
        return _copy_result(out, result)

    integral_exponent = (
        self.is_floating_point()
        and not other.is_floating_point()
        and not other.is_complex()
    )
    if integral_exponent:
        result = ldexp_integral_func(self, other, _WRAP_INTEGER_EXPONENT)
    else:
        result = ldexp_func(self, other)
    if out is None:
        return result
    out.resize_(result.shape)
    return _copy_result(out, result)


def ldexp(self, other):
    logger.debug("GEMS_KUNLUNXIN LDEXP")
    logger.debug("GEMS LDEXP")
    return _ldexp_impl(self, other)


def ldexp_out(self, other, *, out):
    logger.debug("GEMS_KUNLUNXIN LDEXP_OUT")
    logger.debug("GEMS LDEXP_OUT")
    return _ldexp_impl(self, other, out)


def _ldexp_complex_(self, other):
    real_dtype = torch.float64 if self.dtype == torch.complex128 else torch.float32
    xr, xi = _complex_parts(self, real_dtype)
    yr, yi = _complex_parts(other, real_dtype)
    real, imag = ldexp_complex_func(xr, xi, yr, yi)
    _write_complex(self, real, imag)


def _ldexp_complex_integral_(self, other):
    real_dtype = torch.float64 if self.dtype == torch.complex128 else torch.float32
    xr, xi = _complex_parts(self, real_dtype)
    real, imag = ldexp_complex_integral_func(xr, xi, other, _WRAP_INTEGER_EXPONENT)
    _write_complex(self, real, imag)


def ldexp_(self, other):
    logger.debug("GEMS_KUNLUNXIN LDEXP_")
    logger.debug("GEMS LDEXP_")
    if has_internal_overlapping(self) == MemOverlap.Yes:
        raise RuntimeError(
            "unsupported operation: more than one element of the written-to tensor "
            "refers to a single memory location"
        )
    if (
        self.device == other.device
        and self.untyped_storage().nbytes() > 0
        and self.untyped_storage().data_ptr() == other.untyped_storage().data_ptr()
        and not _is_exact_alias(self, other)
    ):
        from flag_gems.ops.copy import copy_

        temporary = torch.empty_like(other)
        if other.is_complex():
            real, imag = _complex_parts(
                other,
                torch.float64 if other.dtype == torch.complex128 else torch.float32,
            )
            _write_complex(temporary, real, imag)
        else:
            copy_(temporary, other)
        other = temporary

    result_dtype = _inplace_result_dtype(self, other)
    if not torch.can_cast(result_dtype, self.dtype):
        raise RuntimeError(
            f"result type {result_dtype} can't be cast to the desired output "
            f"type {self.dtype}"
        )

    output_shape = torch.broadcast_shapes(self.shape, other.shape)
    if output_shape != self.shape:
        raise RuntimeError(
            f"output with shape {self.shape} doesn't match the broadcast shape "
            f"{output_shape}"
        )

    integral_exponent = not other.is_floating_point() and not other.is_complex()
    if self.is_complex():
        if integral_exponent:
            _ldexp_complex_integral_(self, other)
        else:
            _ldexp_complex_(self, other)
    elif integral_exponent:
        ldexp_integral_func(self, other, _WRAP_INTEGER_EXPONENT, out0=self)
    else:
        ldexp_func(self, other, out0=self)
    return self
