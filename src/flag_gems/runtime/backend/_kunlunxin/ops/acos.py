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
import triton.language.extra.xpu.libdevice as xpu

from flag_gems.utils import triton_lang_extension as ext

logger = logging.getLogger(__name__)

# acos(x) fast path: replace the XPU software atan2/acosf external calls
# (measured ~0.12-0.13x torch on the official unary matrix) with a pure
# polynomial in the stable region.
#
#   acos(x)  = 2 * asin(sqrt((1-|x|)/2))          for x >= 0
#            = pi - acos(-x)                       for x < 0
#
# With t = (1-|x|)/2 in [0, 0.5] and s = sqrt(t), asin(s)/s = P(t) with P
# analytic on [0, 0.5]; P is an LSQ fit (degree 8 in t) whose fp32 Horner
# evaluation keeps |acos(x) - acos_ref| <= 3.8e-5 on the full fp32 domain
# [-1, 1] (fp32 simulation, no-FMA assumption), comfortably inside the test
# tolerance (atol 1e-4 + rtol 1.3e-6 * |ref|). NaN/Inf semantics: |x| > 1
# makes t < 0, sqrt(t) yields NaN which propagates through the (single)
# where-chain exactly like torch; NaN input also propagates (comparisons are
# false but the arithmetic stays NaN).
# Coeffs (fp32-rounded, Horner order high -> low):
#   [0.99959993, 0.19823363, -0.72389036, 9.49576759, -60.525768,
#   222.85160828, -470.57415771, 530.01574707, -246.59942627]
MIN_BLOCK = 2048
# unroll 8 beats 16 on the official matrix: (4096,4096) 0.532 vs 0.585 ms,
# [1024,4096] 0.140 vs 0.153 ms, [1024,65536] 2.09 vs 2.36 ms (fp32, XPU2
# wall-clock, same process A/B). Verified in a per-stable subprocess sweep:
# everything else (block/warp/buffer buckets) is within noise.
UNROLL_NUM = 8
# In-place path only (acos_ / arccos_): the read-modify-write aliasing of
# x_ptr == out_ptr makes the deep unroll counter-productive. Measured on the
# official unary matrix (XPU2, same-process A/B, 4096x4096 / [1024,4096] /
# [1024,65536]):
#   fp16 : u2 0.479 / 0.125 / 1.884 ms  vs  u8 0.561 / 0.147 / 2.206 ms
#   fp32 : u2 0.455 / 0.120 / 1.792 ms  vs  u8 0.514 / 0.137 / 2.021 ms
#   bf16 : u4 0.566 / 0.148 / 2.232 ms  vs  u8 0.628 / 0.165 / 2.446 ms
#          (bf16 u2 lands at 0.615 / 0.160 / 2.433 ms, i.e. worse than u4)
# The out-of-place path keeps UNROLL_NUM = 8 because it was tuned with that
# value and is shared with acos / arccos.
INPLACE_UNROLL_NUM = 2
INPLACE_UNROLL_NUM_BF16 = 4
BUFFER_SIZE_LIMIT = 8192
IS_CLOSE_MEMORY_ASYNC = False


def _pick_block(n_elements):
    # Bucket the tile into a few unmasked sizes + 1 masked fallback so the
    # kernel compiles at most ~4 times total. Unmasked runs when the shape
    # divides the tile exactly (masked memory path on XPU costs ~2x).
    # Measured in-place on the official matrix (XPU2, same-process A/B):
    #   n=16384  : 2048/4w masked 5.9us beats 16384/8w unmasked 10.6us (a
    #              single 8-warp CTA is launch/parallelism limited; 8 CTAs x
    #              4 warps wins)
    #   n=65536  : 8192/4w unmasked 7.9us beats 32768/8w unmasked 15.9us
    #   n=262144 : 8192/4w unmasked 14.3us beats 32768/8w unmasked 16.0us
    #   n>=1M    : 32768/8w unmasked remains the sweet spot (32+ CTAs;
    #              8192/4w measured worse on 4194304/16777216/67108864).
    if n_elements <= 16384:
        return 2048, 4, True
    if n_elements <= 262144 and n_elements % 8192 == 0:
        return 8192, 4, False
    if n_elements % 32768 == 0:
        return 32768, 8, False
    if n_elements <= 65536:
        return 2048, 4, True
    return 16384, 8, True


_acos = tl_extra_shim.acos
_atan2 = tl_extra_shim.atan2


@triton.jit
def acos_kernel_unmasked(
    x_ptr,
    out_ptr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = ext.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    x = tl.load(x_ptr + offset).to(tl.float32)
    t = 0.5 - 0.5 * tl.abs(x)
    s = tl.sqrt(t)
    p = -246.59942627
    p = p * t + 530.01574707
    p = p * t + -470.57415771
    p = p * t + 222.85160828
    p = p * t + -60.52576828
    p = p * t + 9.49576759
    p = p * t + -0.72389036
    p = p * t + 0.19823363
    p = p * t + 0.99959993
    y = (s * p) * 2.0
    r = tl.where(x < 0.0, 3.1415927 - y, y)
    tl.store(out_ptr + offset, r.to(out_ptr.dtype.element_ty))

@pointwise_dynamic(promotion_methods=[(0, "INT_TO_FLOAT")], config=config_)
@triton.jit()
def acos_kernel(x):
    x_f32 = x.to(tl.float32)
    in_domain = tl.abs(x_f32) <= 1.0

    # The P800 acos intrinsic has a repeatable error of about 3e-3 on
    # in-domain fp32 values.  atan2(sqrt(1 - x^2), x) avoids that intrinsic;
    # clamp the radicand because fp32 roundoff can make it slightly negative
    # at the endpoints.  Keep the intrinsic for out-of-domain and NaN inputs,
    # where the identity would otherwise return a finite 0 or pi.
    radicand = tl.maximum(1.0 - x_f32 * x_f32, 0.0)
    stable = _atan2(tl.sqrt(radicand), x_f32)
    return tl.where(in_domain, stable, _acos(x_f32))


def acos(x):
    logger.debug("GEMS_KUNLUNXIN ACOS")
    x = x.contiguous()
    out = torch.empty_like(x)
    _launch(x, out)
    return out


def acos_(A):
    logger.debug("GEMS_KUNLUNXIN ACOS_")
    x = A.contiguous()
    _launch(x, x, unroll_num=_inplace_unroll(x.dtype))
    if x.data_ptr() != A.data_ptr():
        A.copy_(x.view(A.shape))
    return A
