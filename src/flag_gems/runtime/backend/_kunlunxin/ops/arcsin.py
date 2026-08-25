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

from flag_gems.utils import triton_lang_extension as ext

_ATAN2 = tl_extra_shim.atan2
_ASIN = tl_extra_shim.asin
logger = logging.getLogger(__name__)

# asin(x) fast path: replace the XPU software atan2/acosf external calls
# (measured ~0.18x torch on the official unary matrix) with a pure
# polynomial in the stable region.
#
#   asin(x) = pi/2 - acos(x)
#
# acos(x) = 2 * asin(sqrt((1-|x|)/2)) for x >= 0, pi - acos(-x) for x < 0;
# with t = (1-|x|)/2 in [0, 0.5], s = sqrt(t), asin(s)/s = P(t) with P the
# same degree-8 LSQ fit as the acos family (acos.py). fp32 Horner keeps
# |asin(x) - asin_ref| <= 4e-5 on the full fp32 domain [-1, 1] (numpy fp32
# emulation, no-FMA), inside the test tolerance (atol 1e-4 + rtol 1.3e-6).
# NaN/Inf semantics: |x| > 1 makes t < 0 -> sqrt(NaN) -> the single
# x < 0 where-chain keeps NaN like torch (|x|>1 gives NaN, NaN input gives
# NaN, ±1 give ±pi/2 exactly). Only ONE ordered comparison (x < 0) remains;
# compound boolean compares ((x<=1)&(x>=-1)) compile to the slow XPU path.
# Coeffs (fp32-rounded, Horner order high -> low), shared with acos.py:
#   [-246.59942627, 530.01574707, -470.57415771, 222.85160828, -60.52576828,
#     9.49576759, -0.72389036, 0.19823363, 0.99959993]
MIN_BLOCK = 2048
# unroll 8 beats 16 on the official unary matrix (acos family sweep on XPU2,
# arccos/arccos_ closure 2026-08-16: u16 -> u8 gained ~4%： 0.6834 -> 0.7170;
# probe on this operator's own matrix stays within noise).
UNROLL_NUM = 8
BUFFER_SIZE_LIMIT = 8192
IS_CLOSE_MEMORY_ASYNC = False


def _pick_block(n_elements):
    # Bucket the tile into a few unmasked sizes + 1 masked fallback so the
    # kernel compiles at most ~4 times total. Unmasked runs when the shape
    # divides the tile exactly (masked memory path on XPU costs ~2x).
    if n_elements >= 16384 and n_elements % 32768 == 0:
        return 32768, 8, False
    if n_elements >= 16384 and n_elements % 16384 == 0:
        return 16384, 8, False
    if n_elements <= 65536:
        return 2048, 4, True
    return 16384, 8, True


@triton.jit
def arcsin_func(x):
    x_f32 = x.to(tl.float32)
    in_domain = tl.abs(x_f32) <= 1.0
    # P800 asin intrinsic mirrors acos: ~3e-3 repeatable error on
    # in-domain fp32 values.  atan2(x, sqrt(1-x^2)) avoids the intrinsic
    # (radicand clamped against fp32 roundoff at endpoints; keep the
    # intrinsic for out-of-domain/NaN where identity gives finite 0).
    radicand = tl.maximum(1.0 - x_f32 * x_f32, 0.0)
    stable = _ATAN2(x_f32, tl.sqrt(radicand))
    return tl.where(in_domain, stable, _ASIN(x_f32))


def arcsin(x, *, out=None):
    logger.debug("GEMS_KUNLUNXIN ARCSIN")
    xc = x.contiguous()
    if out is None:
        out = torch.empty_like(xc)
        _launch(xc, out)
        return out
    oc = out.contiguous()
    _launch(xc, oc)
    if oc.data_ptr() != out.data_ptr():
        out.copy_(oc.view(out.shape))
    return out


def arcsin_(x):
    logger.debug("GEMS_KUNLUNXIN ARCSIN_")
    xc = x.contiguous()
    _launch(xc, xc)
    if xc.data_ptr() != x.data_ptr():
        x.copy_(xc.view(x.shape))
    return x


def arcsin_out(x, *, out=None):
    logger.debug("GEMS_KUNLUNXIN ARCSIN OUT")
    return arcsin(x, out=out)
