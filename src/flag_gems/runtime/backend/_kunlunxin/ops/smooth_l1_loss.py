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
# Must use the vendor (XPU) pointwise_dynamic codegen: the generic
# flag_gems.utils.pointwise_dynamic reads `self.config.balance_grid`, an
# attribute the _kunlunxin CodeGenConfig does not define, so every case
# died at JIT-instantiation time (same fix as igammac_ / lgamma).
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
def _l1_loss(input, target):
    return tl.abs(input.to(tl.float32) - target.to(tl.float32))


@pointwise_dynamic(
    is_tensor=[True, True, False],
    promotion_methods=[(0, 1, "DEFAULT")],
    config=config_,
)
@triton.jit
def _smooth_loss(input, target, beta):
    diff = tl.abs(input.to(tl.float32) - target.to(tl.float32))
    return tl.where(diff < beta, 0.5 * diff * diff / beta, diff - 0.5 * beta)


@pointwise_dynamic(
    is_tensor=[True, True, True],
    promotion_methods=[(0, 1, 2, "DEFAULT")],
    config=config_,
)
@triton.jit
def _l1_backward(input, target, grad_output):
    diff = input.to(tl.float32) - target.to(tl.float32)
    grad = tl.where(diff > 0.0, 1.0, tl.where(diff < 0.0, -1.0, 0.0))
    return grad * grad_output.to(tl.float32)


@pointwise_dynamic(
    is_tensor=[True, True, False],
    promotion_methods=[(0, 1, "DEFAULT")],
    config=config_,
)
@triton.jit
def _l1_backward_scalar(input, target, grad_output):
    diff = input.to(tl.float32) - target.to(tl.float32)
    grad = tl.where(diff > 0.0, 1.0, tl.where(diff < 0.0, -1.0, 0.0))
    return grad * grad_output


@pointwise_dynamic(
    is_tensor=[True, True, True, False],
    promotion_methods=[(0, 1, 2, "DEFAULT")],
    config=config_,
)
@triton.jit
def _smooth_backward(input, target, grad_output, beta):
    diff = input.to(tl.float32) - target.to(tl.float32)
    sign = tl.where(diff > 0.0, 1.0, tl.where(diff < 0.0, -1.0, 0.0))
    grad = tl.where(tl.abs(diff) < beta, diff / beta, sign)
    return grad * grad_output.to(tl.float32)


@pointwise_dynamic(
    is_tensor=[True, True, False, False],
    promotion_methods=[(0, 1, "DEFAULT")],
    config=config_,
)
@triton.jit
def _smooth_backward_scalar(input, target, grad_output, beta):
    diff = input.to(tl.float32) - target.to(tl.float32)
    sign = tl.where(diff > 0.0, 1.0, tl.where(diff < 0.0, -1.0, 0.0))
    grad = tl.where(tl.abs(diff) < beta, diff / beta, sign)
    return grad * grad_output


@libentry()
@triton.jit(do_not_specialize=["grad_scale", "rcp_beta"])
def _smooth_backward_scalar_clamp_kernel(
    in0, in1, out, M, grad_scale, rcp_beta,
    BLOCK: tl.constexpr, NEED_MASK: tl.constexpr,
):
    # Smooth-L1 derivative with beta > 0, scalar-grad path:
    #     grad = clamp((input - target) / beta, -1, 1) * grad_scale
    # ``min(max(x,-1),1)`` is exactly the piecewise smooth-L1 gradient and
    # lowers to single-cycle min/max instructions on this XPU backend, whereas
    # the tl.where/tl.abs chain (previous pwd form) is ALU-bound: ~4x slower
    # at 16.7M elements (0.91ms -> measured memory-bound 0.25ms, ~800 GB/s).
    pid = ext.program_id(0)
    off = pid * BLOCK + tl.arange(0, BLOCK)
    if NEED_MASK:
        mask = off < M
        a = tl.load(in0 + off, mask=mask).to(tl.float32)
        b = tl.load(in1 + off, mask=mask).to(tl.float32)
        diff = a - b
        v = tl.minimum(tl.maximum(diff * rcp_beta, -1.0), 1.0)
        tl.store(out + off, (v * grad_scale).to(out.dtype.element_ty), mask=mask)
    else:
        a = tl.load(in0 + off).to(tl.float32)
        b = tl.load(in1 + off).to(tl.float32)
        diff = a - b
        v = tl.minimum(tl.maximum(diff * rcp_beta, -1.0), 1.0)
        tl.store(out + off, (v * grad_scale).to(out.dtype.element_ty))


# 8192 lanes/CTA: measured sweet spot for the 2-load+1-store stream on this
# backend (2048/4096 lanes drop to 54-60% of the 8192 rate; 16384+ flat).
_SMOOTH_BWD_BLOCK = 8192


def _normalize_reduction(reduction):
    if isinstance(reduction, str):
        return {"none": 0, "mean": 1, "sum": 2}[reduction]
    return reduction


def _broadcast_inputs(input, target):
    shape = torch.broadcast_shapes(input.shape, target.shape)
    if input.numel() == 0 or target.numel() == 0:
        return shape, None, None
    return shape, input, target


def _loss_values(input, target, beta):
    if beta == 0.0:
        return _l1_loss(input, target)
    return _smooth_loss(input, target, beta)


@libentry()
@triton.jit
def _smooth_l1_loss_partial_sum_kernel(
    inp, target, mid, M, beta: tl.constexpr, reduction: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Masked stage-1 (legacy path): one program sums BLOCK_SIZE elements of
    # the smooth-l1 loss into mid[pid] (fp32 accumulation). Tail blocks are
    # handled with mask + other=0 (needs TRITONXPU_OTHER_SIM=1 at launch).
    pid = ext.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < M
    inp_val = tl.load(inp + offset, mask=mask, other=0.0).to(tl.float32)
    target_val = tl.load(target + offset, mask=mask, other=0.0).to(tl.float32)
    diff = tl.abs(inp_val - target_val)
    if beta == 0.0:
        loss = diff
    else:
        loss = tl.where(diff < beta, 0.5 * diff * diff / beta, diff - 0.5 * beta)
    if reduction == 1:
        sum_val = tl.sum(loss) / M
    else:
        sum_val = tl.sum(loss)
    tl.store(mid + pid, sum_val)


@libentry()
@triton.jit
def _smooth_l1_loss_final_sum_kernel(mid, out, mid_size, BLOCK_MID: tl.constexpr):
    offset = tl.arange(0, BLOCK_MID)
    mid_ptrs = mid + offset
    mask = offset < mid_size
    mid_val = tl.load(mid_ptrs, mask=mask, other=0.0).to(tl.float32)
    sum_val = tl.sum(mid_val)
    tl.store(out, sum_val)


# Stage-2 must stay inside the 32768-lane tl.sum ceiling; grow stage-1 blocks
# if the grid would exceed it (legacy MAX_MID rule).
_MAX_MID = 32768


def _smooth_l1_loss_reduce_fused(input, target, beta, reduction):
    input = input.contiguous()
    target = target.contiguous()
    M = input.numel()
    dtype = input.dtype

    block_size = get_block_size_1d(M, input.element_size() * 2)

    # Masked two-stage reduce (single source of truth): stage-1 masks the
    # tail with other=0.0 (TRITONXPU_OTHER_SIM=1 at launch), sums in fp32,
    # stage-2 combines the partials in fp32 and stores the final scalar.
    # NOTE: an earlier "unmasked" 32768-lane stage-1 variant was removed --
    # its maskless tl.sum is incomplete on this backend (fp16 returns NaN,
    # fp32 picks up ~1e-7 lane-drop noise), i.e. the >8192-lane tl.sum mask
    # defect. The masked path is exact for every M including M % 32768 == 0.
    mid_size = triton.cdiv(M, block_size)
    if mid_size > _MAX_MID:
        block_size = triton.next_power_of_2(triton.cdiv(M, _MAX_MID))
        mid_size = triton.cdiv(M, block_size)
    block_mid = triton.next_power_of_2(mid_size)

    mid = torch.empty((mid_size,), dtype=torch.float32, device=input.device)
    out = torch.empty([], dtype=dtype, device=input.device)

    os.environ["TRITONXPU_OTHER_SIM"] = "1"
    with torch_device_fn.device(input.device):
        _smooth_l1_loss_partial_sum_kernel[(mid_size, 1, 1)](
            input, target, mid, M, beta, reduction, block_size,
            buffer_size_limit=2048,
        )
        if mid_size == 1:
            if "TRITONXPU_OTHER_SIM" in os.environ:
                del os.environ["TRITONXPU_OTHER_SIM"]
            return mid.reshape([]).to(dtype)
        _smooth_l1_loss_final_sum_kernel[(1, 1, 1)](
            mid, out, mid_size, block_mid, buffer_size_limit=2048
        )
    if "TRITONXPU_OTHER_SIM" in os.environ:
        del os.environ["TRITONXPU_OTHER_SIM"]

    return out


def smooth_l1_loss(input, target, reduction=1, beta: float = 1.0):
    logger.debug("GEMS KUNLUNXIN SMOOTH_L1_LOSS")
    reduction = _normalize_reduction(reduction)
    beta = float(beta)
    if beta < 0:
        raise RuntimeError("smooth_l1_loss does not support negative values for beta.")

    shape, input_expanded, target_expanded = _broadcast_inputs(input, target)
    if input_expanded is None:
        if reduction == 0:
            return torch.empty(shape, device=input.device, dtype=input.dtype)
        if reduction == 1:
            return torch.full((), float("nan"), device=input.device, dtype=input.dtype)
        return torch.zeros((), device=input.device, dtype=input.dtype)

    if reduction == 0:
        return _loss_values(input_expanded, target_expanded, beta)

    # Fused two-stage reduction (fp32 accumulation, 2 launches): ~2x faster
    # than materializing the loss + vendor sum, and fixes the fp16/bf16
    # large-shape overflow (sum in input dtype -> inf) of the old path.
    if input_expanded.shape == target_expanded.shape:
        return _smooth_l1_loss_reduce_fused(
            input_expanded, target_expanded, beta, reduction
        )

    # Broadcast inputs keep the generic pointwise path (the fused stage-1
    # kernel indexes both operands with the same linear offset).
    loss = _loss_values(input_expanded, target_expanded, beta)
    result = torch.sum(loss)
    if reduction == 1:
        result = result / loss.numel()
    return result


def smooth_l1_loss_out(input, target, reduction=1, beta: float = 1.0, *, out):
    logger.debug("GEMS KUNLUNXIN SMOOTH_L1_LOSS OUT")
    result = smooth_l1_loss(input, target, reduction, beta)
    out.resize_(result.shape)
    out.copy_(result)
    return out


def smooth_l1_loss_backward(grad_output, input, target, reduction, beta: float):
    logger.debug("GEMS KUNLUNXIN SMOOTH_L1_LOSS BACKWARD")
    reduction = _normalize_reduction(reduction)
    beta = float(beta)
    if beta < 0:
        raise RuntimeError("smooth_l1_loss does not support negative values for beta.")

    shape = torch.broadcast_shapes(input.shape, target.shape)
    if input.numel() == 0 or target.numel() == 0:
        return torch.empty(shape, device=input.device, dtype=input.dtype)

    if grad_output.numel() == 1:
        grad_scale = grad_output.item()
        if reduction == 1:
            grad_scale /= input.numel()
        if beta == 0.0:
            return _l1_backward_scalar(input, target, grad_scale)
        # Fast path (beta > 0, same-shape contiguous tensors): single
        # elementwise clamp kernel.  The pwd `_smooth_backward_scalar` (nested
        # tl.where + tl.abs) is ALU-bound on this backend (~4x slower); the
        # min/max form is memory-bound at ~800 GB/s.  Broadcast / non-contiguous
        # shapes keep the general pwd path.
        if input.shape == target.shape and input.is_contiguous() and target.is_contiguous():
            M = input.numel()
            out = torch.empty_like(input)
            with torch_device_fn.device(input.device):
                _smooth_backward_scalar_clamp_kernel[(triton.cdiv(M, _SMOOTH_BWD_BLOCK),)](
                    input,
                    target,
                    out,
                    M,
                    grad_scale,
                    1.0 / beta,
                    BLOCK=_SMOOTH_BWD_BLOCK,
                    NEED_MASK=(M % _SMOOTH_BWD_BLOCK != 0),
                    num_warps=4,
                )
            return out
        return _smooth_backward_scalar(input, target, grad_scale, beta)

    if beta == 0.0:
        return _l1_backward(input, target, grad_output)
    return _smooth_backward(input, target, grad_output, beta)
