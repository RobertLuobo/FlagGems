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

# acos(x) fast path (race: arccos_/acos_, 2026-09-04): the previous kernel
# (tl.sqrt + tl.where sign reconstruction) was select/mask-bound: the x<0
# where-compile emits an ~100-instruction i1->i32 mask extraction
# (llvm.xpu.vvor_f_mh_rn) costing ~0.245 ms at 16.7M elements, and tl.sqrt
# is a software-expanded chain. This rewrite (same recipe as asin, sibling
# operand: acos(x) = pi/2 - asin(x)) uses XPU-friendly ops:
#
#   acos(x) = 2*asin(s) = y        for x >= 0,   s = sqrt(t), t=(1-|x|)/2
#           = pi - y               for x < 0
#   s = t*rsqrt(t+eps) ~ sqrt(t),  P = degree-8 LSQ fit of asin(s)/s
#   m = min(1, max(0, -x*2^126))   (1 iff x<0, no select/compare)
#   r = m*pi + (1-2m)*y
#
#   - rsqrt: xpu.rsqrt lowers to the inline hardware SFU op
#     (tt.extern_elementwise _ZN3xpu6rsqrtfEf, ~0.67x the cost of the
#     software-expanded tl.sqrt chain). The +1e-30 bias keeps s=0 exactly
#     at t=0 (x = +-1): 0*rsqrt(1e-30) = 0, so acos(1) = 0 and
#     acos(-1) = pi like torch. Bit-exact no-op for every other t.
#   - sign: pure min/max/fma, no ordered-compare -> no mask extraction.
#     The 2^126 scale maps x<0 -> m=1, x>0 -> m=0, and the min(1,..)
#     clamps the product; for normal |x| the result is exactly +-1.
#     (min/max on XPU drop NaN, which is fine: y is already NaN for
#     |x|>1, so r = m*pi + (1-2m)*NaN = NaN.)
#   - poly2 = 2*P folds the *2.0 into the coefficients (one mul saved).
#
# Accuracy: fp32 Horner keeps |acos(x)-ref| <= ~4e-5 on [-1,1] (measured
# 3.70e-5 on randn via the shared asin/asin_ eval, |acos| == |asin| error
# by the same polynomial), inside atol 1e-4 + rtol (1.3e-6 / 1e-3 / 1e-3).
# Coeffs (fp32-rounded, Horner order high -> low), shared with asin.py:
#   [-493.19885254, 1060.03149414, -941.14831543, 445.70321655,
#   -121.05153656, 18.99153519, -1.44778073, 0.39646727, 1.99919987]
MIN_BLOCK = 2048
# unroll 8 beats 16 on the official matrix: (4096,4096) 0.532 vs 0.585 ms,
# [1024,4096] 0.140 vs 0.153 ms, [1024,65536] 2.09 vs 2.36 ms (fp32, XPU2
# wall-clock, same process A/B). Verified in a per-stable subprocess sweep:
# everything else (block/warp/buffer buckets) is within noise.
UNROLL_NUM = 8
# In-place path only (acos_ / arccos_): with the previous sqrt+where body
# the read-modify-write aliasing of x_ptr == out_ptr made a deep unroll
# counter-productive (measured on the old body: fp32 u2 0.455ms vs u8
# 0.514ms at 4096x4096). With the rsqrt/min-max body (memory-bound, no
# ~100-instr i1->i32 mask extract) that reverses: full-matrix A/B 2026-09-04
# shows u8 >= u2 on 32/36 cases and >= u4 bf16 everywhere (u8/u2 latency
# ratio median 1.002 / 1.024 / 1.009 for fp16/fp32/bf16, big shapes a
# wash), so the in-place path now shares UNROLL_NUM = 8.
INPLACE_UNROLL_NUM = 8
INPLACE_UNROLL_NUM_BF16 = 8
BUFFER_SIZE_LIMIT = 8192
IS_CLOSE_MEMORY_ASYNC = False


def _pick_block(n_elements):
    # Bucket the tile into a few unmasked sizes + 1 masked fallback so the
    # kernel compiles at most ~6 times total. Unmasked runs when the shape
    # divides the tile exactly (masked memory path on XPU costs ~2x).
    # Larger blocks win once there are >= ~128 programs (16.7M+ elements);
    # mid sizes prefer 32768 (>=16 programs); small sizes 8192; tiny shapes
    # are launch-bound and use the 2048/4w masked kernel.
    if n_elements >= (1 << 24) and n_elements % 131072 == 0:
        return 131072, 8, False
    if n_elements >= (1 << 19) and n_elements % 32768 == 0:
        return 32768, 8, False
    if n_elements >= (1 << 14) and n_elements % 8192 == 0:
        return 8192, 8, False
    if n_elements <= 16384:
        return 2048, 4, True
    if n_elements % 16384 == 0:
        return 16384, 8, False
    return 16384, 8, True


@triton.jit
def _acos_body(x):
    t = 0.5 - 0.5 * tl.abs(x)
    # |x| > 1 makes t < 0 -> rsqrt(NaN) -> NaN propagates out, matching torch.
    s = t * xpu.rsqrt(t + 1e-30)
    p = -493.19885254
    p = p * t + 1060.03149414
    p = p * t + -941.14831543
    p = p * t + 445.70321655
    p = p * t + -121.05153656
    p = p * t + 18.99153519
    p = p * t + -1.44778073
    p = p * t + 0.39646727
    p = p * t + 1.99919987
    y = s * p
    # acos(x) = y (x>=0) / pi - y (x<0); m = 1 iff x<0 (min/max, no select).
    m = tl.minimum(1.0, tl.maximum(0.0, -x * 8.50705917e37))
    return m * 3.1415927 + (1.0 - 2.0 * m) * y


@triton.jit
def acos_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = ext.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < n_elements
    x = tl.load(x_ptr + offset, mask=mask, other=0).to(tl.float32)
    r = _acos_body(x)
    tl.store(out_ptr + offset, r.to(out_ptr.dtype.element_ty), mask=mask)


@triton.jit
def acos_kernel_unmasked(
    x_ptr,
    out_ptr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = ext.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    x = tl.load(x_ptr + offset).to(tl.float32)
    r = _acos_body(x)
    tl.store(out_ptr + offset, r.to(out_ptr.dtype.element_ty))


def _launch(x, out, unroll_num=UNROLL_NUM):
    n_elements = x.numel()
    if n_elements == 0:
        return
    block_size, num_warps, masked = _pick_block(n_elements)
    if masked:
        grid = (triton.cdiv(n_elements, block_size),)
        acos_kernel[grid](
            x,
            out,
            n_elements,
            BLOCK_SIZE=block_size,
            num_warps=num_warps,
            unroll_num=unroll_num,
            buffer_size_limit=BUFFER_SIZE_LIMIT,
            isCloseMemoryAsync=IS_CLOSE_MEMORY_ASYNC,
        )
    else:
        grid = (n_elements // block_size,)
        acos_kernel_unmasked[grid](
            x,
            out,
            BLOCK_SIZE=block_size,
            num_warps=num_warps,
            unroll_num=unroll_num,
            buffer_size_limit=BUFFER_SIZE_LIMIT,
            isCloseMemoryAsync=IS_CLOSE_MEMORY_ASYNC,
        )


def _inplace_unroll(dtype):
    if dtype == torch.bfloat16:
        return INPLACE_UNROLL_NUM_BF16
    return INPLACE_UNROLL_NUM


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
