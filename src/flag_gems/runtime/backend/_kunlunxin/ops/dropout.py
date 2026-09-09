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

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils.random_utils import (
    philox_backend_seed_offset,
    uint_to_uniform_float,
)

logger = logging.getLogger(__name__)


@libentry()
@triton.jit(do_not_specialize=["p", "philox_seed", "philox_offset"])
def dropout_forward_kernel(
    X,
    Y,
    dropout_mask,
    N,
    p,
    philox_seed,
    philox_offset,
    BLOCK: tl.constexpr,
):
    UNROLL: tl.constexpr = 8
    philox_seed = philox_seed.to(tl.int64)
    philox_offset = philox_offset.to(tl.int64)
    c0 = (philox_offset & 0xFFFFFFFF).to(tl.uint32)
    c1 = ((philox_offset >> 32) & 0xFFFFFFFF).to(tl.uint32)

    # First set of 4 random numbers
    i4_0 = tl.program_id(0) * BLOCK * 2 + tl.arange(0, BLOCK)
    c0_0 = c0 + i4_0
    _O = c0_0 * 0
    r0, r1, r2, r3 = tl.philox(philox_seed, c0_0, c1, _O, _O)
    r0 = uint_to_uniform_float(r0)
    r1 = uint_to_uniform_float(r1)
    r2 = uint_to_uniform_float(r2)
    r3 = uint_to_uniform_float(r3)

    # Second set of 4 random numbers
    i4_1 = tl.program_id(0) * BLOCK * 2 + BLOCK + tl.arange(0, BLOCK)
    c0_1 = c0 + i4_1
    _O1 = c0_1 * 0
    r4, r5, r6, r7 = tl.philox(philox_seed, c0_1, c1, _O1, _O1)
    r4 = uint_to_uniform_float(r4)
    r5 = uint_to_uniform_float(r5)
    r6 = uint_to_uniform_float(r6)
    r7 = uint_to_uniform_float(r7)

    mask0 = r0 > p
    mask1 = r1 > p
    mask2 = r2 > p
    mask3 = r3 > p
    mask4 = r4 > p
    mask5 = r5 > p
    mask6 = r6 > p
    mask7 = r7 > p
    scale = 1.0 / (1.0 - p)

    off_0 = tl.program_id(0) * BLOCK * UNROLL + tl.arange(0, BLOCK)
    off_1 = off_0 + BLOCK
    off_2 = off_1 + BLOCK
    off_3 = off_2 + BLOCK
    off_4 = off_3 + BLOCK
    off_5 = off_4 + BLOCK
    off_6 = off_5 + BLOCK
    off_7 = off_6 + BLOCK

    x0 = tl.load(X + off_0, mask=off_0 < N, other=0.0)
    x1 = tl.load(X + off_1, mask=off_1 < N, other=0.0)
    x2 = tl.load(X + off_2, mask=off_2 < N, other=0.0)
    x3 = tl.load(X + off_3, mask=off_3 < N, other=0.0)
    x4 = tl.load(X + off_4, mask=off_4 < N, other=0.0)
    x5 = tl.load(X + off_5, mask=off_5 < N, other=0.0)
    x6 = tl.load(X + off_6, mask=off_6 < N, other=0.0)
    x7 = tl.load(X + off_7, mask=off_7 < N, other=0.0)

    y0 = tl.where(mask0, x0 * scale, 0.0)
    y1 = tl.where(mask1, x1 * scale, 0.0)
    y2 = tl.where(mask2, x2 * scale, 0.0)
    y3 = tl.where(mask3, x3 * scale, 0.0)
    y4 = tl.where(mask4, x4 * scale, 0.0)
    y5 = tl.where(mask5, x5 * scale, 0.0)
    y6 = tl.where(mask6, x6 * scale, 0.0)
    y7 = tl.where(mask7, x7 * scale, 0.0)

    tl.store(Y + off_0, y0, mask=off_0 < N)
    tl.store(Y + off_1, y1, mask=off_1 < N)
    tl.store(Y + off_2, y2, mask=off_2 < N)
    tl.store(Y + off_3, y3, mask=off_3 < N)
    tl.store(Y + off_4, y4, mask=off_4 < N)
    tl.store(Y + off_5, y5, mask=off_5 < N)
    tl.store(Y + off_6, y6, mask=off_6 < N)
    tl.store(Y + off_7, y7, mask=off_7 < N)
    tl.store(dropout_mask + off_0, mask0, mask=off_0 < N)
    tl.store(dropout_mask + off_1, mask1, mask=off_1 < N)
    tl.store(dropout_mask + off_2, mask2, mask=off_2 < N)
    tl.store(dropout_mask + off_3, mask3, mask=off_3 < N)
    tl.store(dropout_mask + off_4, mask4, mask=off_4 < N)
    tl.store(dropout_mask + off_5, mask5, mask=off_5 < N)
    tl.store(dropout_mask + off_6, mask6, mask=off_6 < N)
    tl.store(dropout_mask + off_7, mask7, mask=off_7 < N)


@libentry()
@triton.jit(do_not_specialize=["scale"])
def dropout_backward_kernel(
    DY,
    DX,
    dropout_mask,
    N,
    scale,
    BLOCK: tl.constexpr,
):
    # 2026-08-15 (perf round): the mask tensor is passed as an int8 view
    # (bool tensor storage is i8). On XPU, loading bool (i1) and multiplying
    # it into the float lane costs ~20-25% more than loading the same bytes
    # as i8 and casting to the grad dtype; `m.to(dy.dtype)` stays vectorized.
    #
    # 2026-09-09 (fix): `m.to(dy.dtype)` directly is a MISCOMPILE on this
    # TritonXPU build for tiles >= 4096 (vector<16xf32> cluster paths):
    # ConvertTritonXPUToLLVM fails with "size mismatch when packing elements
    # for LLVM struct expected 8 but got 4" (BLOCK=8192/32768/131072 all hit
    # it; only tiny tiles below the vectorized path survive). A first cast
    # to int32 also fixes it, but the select form `tl.where(m != 0, ...)` is
    # the cheapest (verified vs the int32 cast at 1M/16M/268M elements).
    offset = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    if NEED_MASK:
        mask = offset < N
        m = tl.load(dropout_mask + offset, mask=mask, other=0)
        dy = tl.load(DY + offset, mask=mask, other=0)
        dx = tl.where(m != 0, dy * scale, 0.0)
        tl.store(DX + offset, dx, mask=mask)
    else:
        m = tl.load(dropout_mask + offset)
        dy = tl.load(DY + offset)
        dx = tl.where(m != 0, dy * scale, 0.0)
        tl.store(DX + offset, dx)


UNROLL = 8


def _dropout_launch_config(N):
    # Compute BLOCK / num_warps in Python and pass them EXPLICITLY. On XPU triton
    # letting @triton.heuristics supply num_warps/BLOCK at launch triggers a
    # per-launch recompile pathology (the same class as bernoulli_/std) that blows
    # the IR up (8M+ lines) and makes do_bench measure 500-2000ms for some shapes.
    if N <= 512:
        return 512, 4
    elif N <= 1024:
        return 1024, 8
    else:
        return 1024, 16


def dropout(input, p, train=True):
    logger.debug("GEMS_KUNLUNXIN NATIVE_DROPOUT_FORWARD")
    if not train or p == 0:
        out = input.clone()
        mask = torch.ones_like(input, dtype=torch.bool)
        return out, mask
    if p == 1:
        out = torch.zeros_like(input)
        mask = torch.zeros_like(input, dtype=torch.bool)
        return out, mask
    assert p > 0.0 and p < 1.0, "p must be in (0, 1)"
    device = input.device
    input = input.contiguous()
    out = torch.empty_like(input)
    mask = torch.empty_like(input, dtype=torch.bool)
    N = input.numel()
    BLOCK, num_warps = _dropout_launch_config(N)
    grid = (triton.cdiv(N, BLOCK * UNROLL),)
    increment = triton.cdiv(N, UNROLL)
    with torch_device_fn.device(device):
        philox_seed, philox_offset = philox_backend_seed_offset(increment)
        dropout_forward_kernel[grid](
            input,
            out,
            mask,
            N,
            p,
            philox_seed,
            philox_offset,
            BLOCK=BLOCK,
            num_warps=num_warps,
        )
    return out, mask


def dropout_backward(grad_output, mask, scale):
    logger.debug("GEMS_KUNLUNXIN DROPOUT_BACKWARD")
    grad_output = grad_output.contiguous()
    grad_input = torch.empty_like(grad_output)
    N = grad_output.numel()
    BLOCK, num_warps = _dropout_launch_config(N)
    grid = (triton.cdiv(N, BLOCK),)
    with torch_device_fn.device(grad_output.device):
        dropout_backward_kernel[grid](
            grad_output,
            grad_input,
            mask,
            N,
            scale,
            BLOCK=BLOCK,
            num_warps=num_warps,
        )
    return grad_input


def native_dropout_backward(grad_output, mask, scale):
    """Kunlunxin override of aten::native_dropout_backward.

    The generic flag_gems.ops.native_dropout_backward adapter calls the generic
    flag_gems.ops.dropout.dropout_backward (bound at import time), which uses
    @triton.heuristics with the vendor launch config. On XPU that path has a
    per-launch rebuild pathology (heuristics-driven constexpr recompile) and
    loads the bool mask as i1, both much slower than the tuned int8-view mask
    kernel below. Registering this wrapper makes the ATen op dispatch to the
    same tuned kernel as dropout_backward.
    """
    logger.debug("GEMS_KUNLUNXIN NATIVE_DROPOUT_BACKWARD")
    return dropout_backward(grad_output, mask, scale)
