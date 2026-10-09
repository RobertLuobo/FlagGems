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
import os

import torch
import triton
import triton.language as tl
from _kunlunxin.utils.codegen_config_utils import CodeGenConfig

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as ext

from ..utils.block_size_utils import get_block_size_1d
from ..utils.pointwise_dynamic import pointwise_dynamic

logger = logging.getLogger(__name__)

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


@pointwise_dynamic(promotion_methods=[(0, 1, "DEFAULT")], config=config_)
@triton.jit
def _l1_loss_elementwise(input, target):
    return tl.abs(input.to(tl.float32) - target.to(tl.float32))


@libentry()
@triton.jit
def _l1_loss_partial_sum_kernel(
    inp,
    target,
    mid,
    M,
    reduction: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = ext.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < M
    inp_val = tl.load(inp + offset, mask=mask, other=0.0).to(tl.float32)
    target_val = tl.load(target + offset, mask=mask, other=0.0).to(tl.float32)
    diff = tl.abs(inp_val - target_val)
    if reduction == 1:
        sum_val = tl.sum(diff) / M
    else:
        sum_val = tl.sum(diff)
    tl.store(mid + pid, sum_val)


@libentry()
@triton.jit
def _sum_partial_kernel(inp, mid, M, MEAN: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid = ext.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < M
    v = tl.load(inp + offset, mask=mask, other=0.0).to(tl.float32)
    sum_val = tl.sum(v)
    if MEAN:
        sum_val = sum_val / M
    tl.store(mid + pid, sum_val)


@libentry()
@triton.jit
def _l1_loss_final_sum_kernel(mid, out, mid_size, BLOCK_MID: tl.constexpr):
    offset = tl.arange(0, BLOCK_MID)
    mask = offset < mid_size
    mid_val = tl.load(mid + offset, mask=mask, other=0.0).to(tl.float32)
    tl.store(out, tl.sum(mid_val))


_MAX_MID = 32768


def _normalize_reduction(reduction):
    if isinstance(reduction, str):
        mapping = {"none": 0, "mean": 1, "sum": 2}
        r = reduction.lower()
        if r not in mapping:
            raise ValueError(f"Invalid reduction: {reduction}")
        return mapping[r]
    if isinstance(reduction, int):
        if reduction in (0, 1, 2):
            return reduction
        raise ValueError(f"Invalid reduction int: {reduction}")
    raise ValueError(f"Unsupported reduction type: {type(reduction)}")


def _broadcast_inputs(input, target):
    shape = torch.broadcast_shapes(input.shape, target.shape)
    if input.numel() == 0 or target.numel() == 0:
        return shape, None, None
    return shape, input, target


def _l1_loss_reduce_fused(input, target, reduction):
    input = input.contiguous()
    target = target.contiguous()
    M = input.numel()
    dtype = input.dtype

    block_size = get_block_size_1d(M, input.element_size() * 2)

    mid_size = triton.cdiv(M, block_size)
    if mid_size > _MAX_MID:
        block_size = triton.next_power_of_2(triton.cdiv(M, _MAX_MID))
        mid_size = triton.cdiv(M, block_size)
    block_mid = triton.next_power_of_2(mid_size)

    mid = torch.empty((mid_size,), dtype=torch.float32, device=input.device)
    out = torch.empty([], dtype=dtype, device=input.device)

    os.environ["TRITONXPU_OTHER_SIM"] = "1"
    with torch_device_fn.device(input.device):
        _l1_loss_partial_sum_kernel[(mid_size, 1, 1)](
            input,
            target,
            mid,
            M,
            reduction,
            block_size,
            buffer_size_limit=2048,
        )
        if mid_size == 1:
            if "TRITONXPU_OTHER_SIM" in os.environ:
                del os.environ["TRITONXPU_OTHER_SIM"]
            return mid.reshape([]).to(dtype)
        _l1_loss_final_sum_kernel[(1, 1, 1)](
            mid, out, mid_size, block_mid, buffer_size_limit=2048
        )
    if "TRITONXPU_OTHER_SIM" in os.environ:
        del os.environ["TRITONXPU_OTHER_SIM"]

    return out


def _sum_reduce(loss, mean):
    loss = loss.contiguous().reshape(-1)
    M = loss.numel()
    block_size = get_block_size_1d(M, loss.element_size())
    mid_size = triton.cdiv(M, block_size)
    if mid_size > _MAX_MID:
        block_size = triton.next_power_of_2(triton.cdiv(M, _MAX_MID))
        mid_size = triton.cdiv(M, block_size)
    block_mid = triton.next_power_of_2(mid_size)

    mid = torch.empty((mid_size,), dtype=torch.float32, device=loss.device)
    out = torch.empty([], dtype=loss.dtype, device=loss.device)

    os.environ["TRITONXPU_OTHER_SIM"] = "1"
    with torch_device_fn.device(loss.device):
        _sum_partial_kernel[(mid_size, 1, 1)](
            loss, mid, M, mean, block_size, buffer_size_limit=2048
        )
        if mid_size == 1:
            if "TRITONXPU_OTHER_SIM" in os.environ:
                del os.environ["TRITONXPU_OTHER_SIM"]
            return mid.reshape([]).to(loss.dtype)
        _l1_loss_final_sum_kernel[(1, 1, 1)](
            mid, out, mid_size, block_mid, buffer_size_limit=2048
        )
    if "TRITONXPU_OTHER_SIM" in os.environ:
        del os.environ["TRITONXPU_OTHER_SIM"]

    return out


def l1_loss(input, target, reduction=1):
    logger.debug("GEMS_KUNLUNXIN L1_LOSS")
    reduction = _normalize_reduction(reduction)

    shape, input_e, target_e = _broadcast_inputs(input, target)
    if input_e is None:
        if reduction == 0:
            return torch.empty(shape, device=input.device, dtype=input.dtype)
        if reduction == 1:
            return torch.full((), float("nan"), device=input.device, dtype=input.dtype)
        return torch.zeros((), device=input.device, dtype=input.dtype)

    if reduction == 0:
        return _l1_loss_elementwise(input_e.contiguous(), target_e.contiguous())

    if input_e.shape == target_e.shape:
        return _l1_loss_reduce_fused(input_e, target_e, reduction)

    loss = _l1_loss_elementwise(input_e, target_e)
    return _sum_reduce(loss, mean=(reduction == 1))
