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

from flag_gems import runtime
from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as ext

logger = logging.getLogger(__name__)


@triton.jit
def _i0(x):
    # Pure-Triton modified Bessel I0 (Abramowitz & Stegun 9.8.1/9.8.2); libdevice.cyl_bessel_i0 does not link on XPU3.
    ax = tl.abs(x)
    use_small = ax <= 3.75

    z = ax * (1.0 / 3.75)
    y = z * z
    p_small = tl.fma(y, 0.0045813, 0.0360768)
    p_small = tl.fma(y, p_small, 0.2659732)
    p_small = tl.fma(y, p_small, 1.2067492)
    p_small = tl.fma(y, p_small, 3.0899424)
    p_small = tl.fma(y, p_small, 3.5156229)
    res_small = tl.fma(y, p_small, 1.0)

    safe_ax = tl.where(use_small, 3.75, ax)
    yb = 3.75 / safe_ax
    p_big = tl.fma(yb, 0.00392377, -0.01647633)
    p_big = tl.fma(yb, p_big, 0.02635537)
    p_big = tl.fma(yb, p_big, -0.02057706)
    p_big = tl.fma(yb, p_big, 0.00916281)
    p_big = tl.fma(yb, p_big, -0.00157565)
    p_big = tl.fma(yb, p_big, 0.00225319)
    p_big = tl.fma(yb, p_big, 0.01328592)
    p_big = tl.fma(yb, p_big, 0.39894228)
    res_big = tl.exp(safe_ax) * p_big * tl.rsqrt(safe_ax)

    return tl.where(use_small, res_small, res_big)


@libentry()
@triton.jit
def kaiser_window_kernel(
    out_ptr,
    window_length,
    half_length,
    beta,
    IS_FP16: tl.constexpr,
    IS_BF16: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = ext.program_id(0)
    idx = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = idx < window_length

    out_dtype = out_ptr.dtype.element_ty

    # torch rounds the element index into the output dtype before the arithmetic (visible once idx exceeds the dtype's exact-integer range). The XPU3 Triton int->bfloat16 cast does not drop mantissa bits, so bfloat16 rounding is emulated by round-to-nearest-even on the fp32 bit pattern; the float16 cast rounds correctly.
    if IS_BF16:
        nf = idx.to(tl.float32)
        u = nf.to(tl.int32, bitcast=True)
        lsb = (u >> 16) & 1
        u = (u + 0x7FFF + lsb) & -65536
        n = u.to(tl.float32, bitcast=True)
    elif IS_FP16:
        n = idx.to(tl.float16).to(tl.float32)
    else:
        n = idx.to(tl.float32)
    x = (n - half_length) / half_length
    val = _i0(beta * tl.sqrt(tl.maximum(1.0 - x * x, 0.0)))
    val = val / _i0(beta + 0.0)

    tl.store(out_ptr + idx, val.to(out_dtype), mask=mask)


def kaiser_window(
    window_length,
    periodic=True,
    beta=12.0,
    *,
    dtype=None,
    layout=torch.strided,
    device=None,
    pin_memory=None,
):
    logger.debug("GEMS_KUNLUNXIN KAISER_WINDOW")
    assert window_length >= 0, "window_length must be non-negative"

    if dtype is None:
        dtype = torch.get_default_dtype()
    assert dtype.is_floating_point, "kaiser_window only supports floating point dtype"

    if layout != torch.strided:
        raise ValueError("Only strided layout is supported for kaiser_window.")

    if device is None:
        device = torch.device(runtime.device.name)

    out = torch.empty(
        (window_length,),
        dtype=dtype,
        layout=layout,
        device=device,
        pin_memory=pin_memory,
    )

    if window_length == 0:
        return out
    if window_length == 1:
        return torch.fill(out, 1.0)

    full_length = window_length if periodic else window_length - 1

    BLOCK_SIZE = 1024
    grid = (triton.cdiv(window_length, BLOCK_SIZE),)
    kaiser_window_kernel[grid](
        out,
        window_length,
        full_length / 2.0,
        float(beta),
        dtype == torch.float16,
        dtype == torch.bfloat16,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return out


def kaiser_window_periodic(
    window_length,
    periodic,
    *,
    dtype=None,
    layout=torch.strided,
    device=None,
    pin_memory=None,
):
    return kaiser_window(
        window_length,
        periodic,
        dtype=dtype,
        layout=layout,
        device=device,
        pin_memory=pin_memory,
    )


def kaiser_window_beta(
    window_length,
    periodic,
    beta,
    *,
    dtype=None,
    layout=torch.strided,
    device=None,
    pin_memory=None,
):
    return kaiser_window(
        window_length,
        periodic,
        beta,
        dtype=dtype,
        layout=layout,
        device=device,
        pin_memory=pin_memory,
    )
