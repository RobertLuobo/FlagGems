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

from flag_gems.utils import triton_lang_extension as ext
from ..utils.pointwise_dynamic import pointwise_dynamic

logger = logging.getLogger(__name__)


def _unwrap_if_constexpr(o):
    return o.value if isinstance(o, tl.constexpr) else o


@tl.constexpr
def _get_uint_dtype(num_bits):
    num_bits = _unwrap_if_constexpr(num_bits)
    return tl.core.get_int_dtype(num_bits, False)


@tl.constexpr
def _get_sign_bit_mask(num_bits):
    num_bits = _unwrap_if_constexpr(num_bits)
    return 1 << (num_bits - 1)


# Reduced buffer_size_limit: the bf16 path widens `other` to fp32 (doubling the
# temp footprint) and additionally allocates a uint32 view + mask, which
# overflows XPU uni_sram at larger shapes under the default pointwise config.
config_ = CodeGenConfig(
    512,
    (65536, 65536, 65536),
    32,
    True,
    prefer_1d_tile=True,
    buffer_size_limit=2048,
    isCloseVectorization=True,
    kunlunAutoGrid=True,
    unroll_num=8,
)


@pointwise_dynamic(
    is_tensor=[True, True],
    promotion_methods=[(0, 1, "DEFAULT")],
    config=config_,
)
@triton.jit
def copysign_func(input, other):
    # Compute magnitude of input, apply sign of other. Do all work in fp32:
    # bf16 in-place output otherwise miscompiles under TritonXPUDtypeConvert
    # (both the bitcast path and native-bf16 arithmetic + tl.where + negate
    # trip the pass). fp32 intermediate + explicit final cast mirrors log2_.
    inp_f32 = input.to(tl.float32)
    oth_f32 = other.to(tl.float32)
    abs_val = tl.abs(inp_f32)
    signed = tl.where(oth_f32 < 0.0, -abs_val, abs_val)
    return signed.to(input.dtype)


# In-place copysign_ fast path (XPU): integer bit-domain copysign.
# Measured on XPU 3 the float compare+select body is ~8-10x slower than
# ATen on large fp16/bf16 tensors, while a pure 2-op integer body
# ((abs bits of a) ^ (sign bit of b)) costs ~3x less. bf16 must widen to
# fp32 bits first (native u16 bit path overflows uni_sram / trips
# TritonXPUDtypeConvert); fp32 uses the native-width u32 path. The XOR
# form keeps the payload at two ALU ops; measured identical to AND+OR.
@pointwise_dynamic(
    is_tensor=[True, True],
    promotion_methods=[(0, 1, "DEFAULT")],
    config=config_,
)
@triton.jit
def copysign_bit_func(input, other):
    if input.dtype == tl.float16:
        ua = input.to(tl.uint16, bitcast=True)
        ub = other.to(tl.uint16, bitcast=True)
        r = (ua & 0x7FFF) ^ (ub & 0x8000)
        return r.to(input.dtype, bitcast=True)
    elif input.dtype == tl.bfloat16:
        # bf16 widening keeps bf16 bits in the HIGH half of u32, so the
        # fp32 bit pattern of the result is already exact; the final
        # value conversion to bf16 is lossless.
        ua = input.to(tl.float32).to(tl.uint32, bitcast=True)
        ub = other.to(tl.float32).to(tl.uint32, bitcast=True)
        r = (ua & 0x7FFFFFFF) ^ (ub & 0x80000000)
        v = r.to(tl.float32, bitcast=True)
        return v.to(input.dtype)
    else:
        ua = input.to(tl.uint32, bitcast=True)
        ub = other.to(tl.uint32, bitcast=True)
        r = (ua & 0x7FFFFFFF) ^ (ub & 0x80000000)
        return r.to(input.dtype, bitcast=True)


def copysign(input, other, *, out=None):
    logger.debug("GEMS_KUNLUNXIN COPYSIGN")
    return copysign_func(input, other)


def copysign_out(input, other, *, out=None):
    logger.debug("GEMS_KUNLUNXIN COPYSIGN_OUT")
    if out is None:
        return copysign_func(input, other)
    copysign_bit_func(input, other, out0=out)
    return out


def copysign_(input, other):
    logger.debug("GEMS_KUNLUNXIN COPYSIGN_")
    if _copysign_fast_(input, other):
        return input
    copysign_bit_func(input, other, out0=input)
    return input


# ---------------------------------------------------------------------------
# Fast in-place path.  The generic bit body above works on 16-bit lanes, but
# the XPU LLVM backend widens every u16 AND/XOR to u32 sequences (TritonXPU
# DtypeConvert), which costs ~2x over the memory-bound floor (measured 211us
# vs 98us on [4096,4096] fp16).  Processing the SAME bit identity on u32 lanes
# (two 16-bit elements per lane for fp16/bf16, one fp32 per lane) keeps the
# ALU at native width and reaches ~1.0TB/s (sp ~0.95 vs ATen on 16M-elems).
# The u32 bit identity is exactly equivalent: magnitude bits of A plus sign
# bit of B, including ±0/inf/NaN payload/sign bits (identical to the u16
# version above, which the tests already validate bit-for-bit).
# ---------------------------------------------------------------------------

@triton.jit
def _copysign_pair_kernel(a_ptr, b_ptr, o_ptr, n_lanes, BLOCK: tl.constexpr):
    # 16-bit dtypes (fp16/bf16) viewed as int32: two elements per lane.
    pid = ext.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_lanes
    ua = tl.load(a_ptr + offs, mask=mask).to(tl.uint32, bitcast=True)
    ub = tl.load(b_ptr + offs, mask=mask).to(tl.uint32, bitcast=True)
    r = (ua & 0x7FFF7FFF) | (ub & 0x80008000)
    tl.store(o_ptr + offs, r.to(tl.int32, bitcast=True), mask=mask)


@triton.jit
def _copysign_single_kernel(a_ptr, b_ptr, o_ptr, n, BLOCK: tl.constexpr):
    # fp32: one element per u32 lane.
    pid = ext.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    ua = tl.load(a_ptr + offs, mask=mask).to(tl.uint32, bitcast=True)
    ub = tl.load(b_ptr + offs, mask=mask).to(tl.uint32, bitcast=True)
    r = (ua & 0x7FFFFFFF) | (ub & 0x80000000)
    tl.store(o_ptr + offs, r.to(tl.int32, bitcast=True), mask=mask)


# ---------------------------------------------------------------------------
# Size-adaptive launch config.
# Measured on XPU (L = u32 lane count, sweep of {BLOCK x num_warps} on the
# benchmark/test_copysign_.py grid, fp16/bf16 pair + fp32 single): the
# memory-bound floor for large inputs needs BLOCK=65536/num_warps=16, but
# that fixed config over-provisions small inputs (a masked 65536-lane tile
# with 512 threads on <=2K lanes costs ~2us extra vs a 1-4 CTA launch), so
# both BLOCK and num_warps scale down with the lane count.  Every bucket
# below was measured to beat the wrapped u16/u32 bit-body path (the
# pre-fast-path fallback) across the full benchmark shape grid.
# ---------------------------------------------------------------------------
def _pick_lane_cfg(lanes):
    if lanes <= 1024:
        return 1024, 2
    if lanes <= 16384:
        return 4096, 4
    if lanes <= 65536:
        return 8192, 8
    if lanes <= 1048576:
        return 16384, 8
    return 65536, 16


def _copysign_fast_(input, other):
    """Return True if the fast u32-lane kernel handled the in-place op."""
    n = input.numel()
    if n == 0:
        return True
    if (
        other.numel() != n
        or not input.is_contiguous()
        or not other.is_contiguous()
        or input.data_ptr() % 4 != 0
        or other.data_ptr() % 4 != 0
    ):
        return False
    dt = input.dtype
    if dt in (torch.float16, torch.bfloat16):
        if n % 2 != 0:
            return False
        a32 = input.view(-1).view(torch.int32)
        b32 = other.view(-1).view(torch.int32)
        n_lanes = n // 2
        block, warps = _pick_lane_cfg(n_lanes)
        _copysign_pair_kernel[
            (triton.cdiv(n_lanes, block),)
        ](a32, b32, a32, n_lanes, BLOCK=block, num_warps=warps,
          unroll_num=16, isCloseVectorization=True)
        return True
    if dt == torch.float32:
        a32 = input.view(-1).view(torch.int32)
        b32 = other.view(-1).view(torch.int32)
        block, warps = _pick_lane_cfg(n)
        _copysign_single_kernel[
            (triton.cdiv(n, block),)
        ](a32, b32, a32, n, BLOCK=block, num_warps=warps,
          unroll_num=16, isCloseVectorization=True)
        return True
    return False
