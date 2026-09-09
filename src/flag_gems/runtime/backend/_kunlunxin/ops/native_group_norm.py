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
from flag_gems.utils import libentry, tl_extra_shim
from flag_gems.utils import triton_lang_extension as ext

# The accuracy test asserts on the generic logger name; native_layer_norm.py
# does the same thing for the same reason.
logger = logging.getLogger("flag_gems.ops.native_group_norm")
rsqrt = tl_extra_shim.rsqrt

# A group is `group_size` channels x HW CONTIGUOUS elements; one program is
# one (n, group), grid = N*group.  Two kernels:
#
# 1) native_group_norm_small_kernel (num = group_size*HW <= NGN_SMALL_NUM):
#    the group fits a single 256-lane tile, so BOTH the flat 1D reduction and
#    the affine write run in one iteration.  The weight/bias per lane is
#    recovered with `ch = min(idx // HW, group_size-1)` (a small gather over
#    at most NGN_SMALL_NUM/HW <= 256 addresses, all L1-resident).  Measured
#    (2026-09-08, interleaved min-of-7): this is 1.9-2.4x FASTER than the
#    per-channel scalar kernel below on (16,16,64) / (1,8,4,4), because the
#    per-channel path pays the XPU small-tile codegen penalty (64/128-wide
#    masked tiles are ~6x more expensive per lane than 256+ wide ones) with
#    FOUR iterations per program instead of one.  `ch` is clamped so masked
#    lanes never read past the weight tensor (stores stay masked).
#
# 2) native_group_norm_kernel (num > NGN_SMALL_NUM): per-channel loop with a
#    SCALAR weight/bias per channel + contiguous HW block DMA (no per-element
#    div/gather).  BLOCK is min(next_pow2(HW), NGN_BLOCK_MAX): raisng the cap
#    from 1024 to 4096 cuts the iteration count 3-4x on the large-HW cells
#    ((16,16,4098) 452.8->137-166us, (16,8,128,128) 318->76-97us, measured
#    2026-09-08).  NEED_MASK selects an unmasked (exact-divisible) fast path;
#    tiles < 64 lanes are always kept masked (the [0,32] unmasked interval
#    miscompiles on this backend, and 64-wide unmasked measures no better than
#    masked).
#
# This replaced the 2026-08-29 single-fused kernel (flat 1D reduce + per-channel
# scalar normalize, BLOCK capped at 1024, always masked), which measured
# 0.1378-0.1497x dtype-equal-weight; the hybrid measures ~0.23-0.28x on the
# same matrix (see harness/solution/performance/native_group_norm_xpu7_20260908.md).

NGN_SMALL_NUM = 256
NGN_BLOCK_MAX = 4096


@libentry()
@triton.jit(do_not_specialize=["eps"])
def native_group_norm_small_kernel(
    X,
    Y,
    W,
    B,
    Mean,
    Rstd,
    group_size,
    HW,
    num_groups,
    eps,
    BLOCK: tl.constexpr,
):
    pid = ext.program_id(0)
    group = pid % num_groups
    num_elements = group_size * HW
    base = pid * num_elements
    ch_base = group * group_size

    sum_acc = tl.zeros([BLOCK], dtype=tl.float32)
    sumsq_acc = tl.zeros([BLOCK], dtype=tl.float32)
    for off in range(0, num_elements, BLOCK):
        idx = off + tl.arange(0, BLOCK)
        m = idx < num_elements
        x = tl.load(X + base + idx, mask=m, other=0.0).to(tl.float32)
        sum_acc += x
        sumsq_acc += x * x

    mean = tl.sum(sum_acc) / num_elements
    var = tl.sum(sumsq_acc) / num_elements - mean * mean
    rstd = rsqrt(var + eps)
    tl.store(Mean + pid, mean)
    tl.store(Rstd + pid, rstd)

    for off in range(0, num_elements, BLOCK):
        idx = off + tl.arange(0, BLOCK)
        m = idx < num_elements
        x = tl.load(X + base + idx, mask=m, other=0.0).to(tl.float32)
        ch = tl.minimum(idx // HW, group_size - 1)
        if W is None:
            weight = 1.0
        else:
            weight = tl.load(W + ch_base + ch, mask=m, other=0.0).to(tl.float32)
        if B is None:
            bias = 0.0
        else:
            bias = tl.load(B + ch_base + ch, mask=m, other=0.0).to(tl.float32)
        y = (x - mean) * rstd * weight + bias
        tl.store(Y + base + idx, y, mask=m)


@libentry()
@triton.jit(do_not_specialize=["eps"])
def native_group_norm_kernel(
    X,
    Y,
    W,
    B,
    Mean,
    Rstd,
    group_size,
    HW,
    num_groups,
    eps,
    GROUP_SIZE: tl.constexpr,
    BLOCK: tl.constexpr,
    NEED_MASK: tl.constexpr,
):
    pid = ext.program_id(0)
    group = pid % num_groups
    num_elements = group_size * HW
    base = pid * num_elements
    ch_base = group * group_size

    sum_acc = tl.zeros([BLOCK], dtype=tl.float32)
    sumsq_acc = tl.zeros([BLOCK], dtype=tl.float32)
    if NEED_MASK:
        for off in range(0, num_elements, BLOCK):
            idx = off + tl.arange(0, BLOCK)
            m = idx < num_elements
            x = tl.load(X + base + idx, mask=m, other=0.0).to(tl.float32)
            sum_acc += x
            sumsq_acc += x * x
    else:
        for off in range(0, num_elements, BLOCK):
            idx = off + tl.arange(0, BLOCK)
            x = tl.load(X + base + idx).to(tl.float32)
            sum_acc += x
            sumsq_acc += x * x

    mean = tl.sum(sum_acc) / num_elements
    var = tl.sum(sumsq_acc) / num_elements - mean * mean
    rstd = rsqrt(var + eps)
    tl.store(Mean + pid, mean)
    tl.store(Rstd + pid, rstd)

    for c in range(0, GROUP_SIZE):
        cbase = base + c * HW
        if W is None:
            weight = 1.0
        else:
            weight = tl.load(W + ch_base + c).to(tl.float32)
        if B is None:
            bias = 0.0
        else:
            bias = tl.load(B + ch_base + c).to(tl.float32)
        if NEED_MASK:
            for off in range(0, HW, BLOCK):
                idx = off + tl.arange(0, BLOCK)
                m = idx < HW
                x = tl.load(X + cbase + idx, mask=m, other=0.0).to(tl.float32)
                y = (x - mean) * rstd * weight + bias
                tl.store(Y + cbase + idx, y, mask=m)
        else:
            for off in range(0, HW, BLOCK):
                idx = off + tl.arange(0, BLOCK)
                x = tl.load(X + cbase + idx).to(tl.float32)
                y = (x - mean) * rstd * weight + bias
                tl.store(Y + cbase + idx, y)


def native_group_norm(input, weight, bias, N, C, HxW, group, eps=1e-05):
    """aten::native_group_norm on a single fused Kunlunxin kernel.

    The generic flag_gems.ops.native_group_norm binds
    flag_gems.ops.groupnorm.group_norm at import time, so SpecOpRegistrar
    swapping flag_gems.group_norm never reached it and native_group_norm kept
    running the generic single-kernel giant-2D-tile implementation on XPU.
    That path miscompiles on the small tiles used by the accuracy matrix and
    hard-fails with `out of resource: uni_sram` for HxW >= 4096, so bind a
    vendor kernel here explicitly.
    """
    logger.debug("GEMS NATIVE_GROUP_NORM")

    group_size = triton.cdiv(C, group)
    input = input.contiguous()
    weight = None if weight is None else weight.contiguous()
    bias = None if bias is None else bias.contiguous()

    y = torch.empty_like(input)
    mean = torch.empty((N, group), dtype=input.dtype, device=input.device)
    rstd = torch.empty((N, group), dtype=input.dtype, device=input.device)

    num_elements = group_size * HxW
    grid = (N * group,)
    with torch_device_fn.device(input.device):
        if num_elements <= NGN_SMALL_NUM:
            native_group_norm_small_kernel[grid](
                input,
                y,
                weight,
                bias,
                mean,
                rstd,
                group_size,
                HxW,
                group,
                eps,
                BLOCK=NGN_SMALL_NUM,
            )
        else:
            block_hw = min(triton.next_power_of_2(HxW), NGN_BLOCK_MAX)
            need_mask = (
                (num_elements % block_hw != 0)
                or (HxW % block_hw != 0)
                or (block_hw < 64)
            )
            native_group_norm_kernel[grid](
                input,
                y,
                weight,
                bias,
                mean,
                rstd,
                group_size,
                HxW,
                group,
                eps,
                GROUP_SIZE=group_size,
                BLOCK=block_hw,
                NEED_MASK=need_mask,
            )
    return y, mean, rstd