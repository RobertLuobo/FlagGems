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

from ..utils.pointwise_dynamic import pointwise_dynamic

logger = logging.getLogger(__name__)

# Mirror silu (closest exp-based unary activation) for the general fallback path.
config_ = CodeGenConfig(
    512,
    (65536, 65536, 65536),
    32,
    True,
    prefer_1d_tile=True,
    buffer_size_limit=4096,
    isCloseVectorization=True,
    unroll_num=8,
)


@pointwise_dynamic(
    is_tensor=[True, False, False, False],
    promotion_methods=[(0, "DEFAULT")],
    config=config_,
)
@triton.jit
def elu_forward_kernel(x, alpha, scale, input_scale):
    x_fp32 = x.to(tl.float32)
    return tl.where(
        x_fp32 > 0,
        scale * input_scale * x_fp32,
        scale * alpha * (tl.exp(x_fp32 * input_scale) - 1),
    )


@pointwise_dynamic(
    is_tensor=[True, True, False, False, False, False],
    promotion_methods=[(0, 1, "DEFAULT")],
)
@triton.jit
def elu_backward_kernel(grad_output, x, alpha, scale, input_scale, is_result):
    x_fp32 = x.to(tl.float32)
    grad_pos = grad_output * scale * input_scale
    if is_result:
        grad_neg = grad_output * input_scale * (x_fp32 + scale * alpha)
    else:
        grad_neg = (
            grad_output * scale * alpha * input_scale * tl.exp(x_fp32 * input_scale)
        )

    return tl.where(x_fp32 > 0, grad_pos, grad_neg)


# Flat contiguous fast path (mirrors leaky_relu): a hand-written flat kernel with buffer_size_limit>=2048 overlaps gm2lm DMAs, ~2-3x faster than the pointwise_dynamic memory path on XPU3.
_ELU_FLAT_DTYPES = (torch.float16, torch.float32, torch.bfloat16)
_ELU_FLAT_TIERS = (
    (8192, 1024, 4),
    (65536, 2048, 4),
    (524288, 4096, 8),
    (1 << 20, 16384, 8),
    (None, 16384, 8),
)
_ELU_BSL = 8192
_ELU_FAT_BLOCK = 131072
_ELU_FAT_MIN_NUMEL = 1 << 22


@triton.jit
def elu_forward_flat_kernel(
    x_ptr,
    out_ptr,
    n,
    alpha,
    scale,
    input_scale,
    BLOCK: tl.constexpr,
    NEED_MASK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    if NEED_MASK:
        mask = offs < n
        x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    else:
        x = tl.load(x_ptr + offs)
    x32 = x.to(tl.float32)
    pos = scale * input_scale * x32
    neg = scale * alpha * (tl.exp(x32 * input_scale) - 1.0)
    o = tl.where(x32 > 0, pos, neg)
    if NEED_MASK:
        tl.store(out_ptr + offs, o.to(x.dtype), mask=mask)
    else:
        tl.store(out_ptr + offs, o.to(x.dtype))


@triton.jit
def elu_backward_flat_kernel(
    g_ptr,
    x_ptr,
    out_ptr,
    n,
    alpha,
    scale,
    input_scale,
    IS_RESULT: tl.constexpr,
    BLOCK: tl.constexpr,
    NEED_MASK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    if NEED_MASK:
        mask = offs < n
        g = tl.load(g_ptr + offs, mask=mask, other=0.0)
        x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    else:
        g = tl.load(g_ptr + offs)
        x = tl.load(x_ptr + offs)
    x32 = x.to(tl.float32)
    g32 = g.to(tl.float32)
    grad_pos = g32 * scale * input_scale
    if IS_RESULT:
        grad_neg = g32 * input_scale * (x32 + scale * alpha)
    else:
        grad_neg = g32 * scale * alpha * input_scale * tl.exp(x32 * input_scale)
    o = tl.where(x32 > 0, grad_pos, grad_neg)
    if NEED_MASK:
        tl.store(out_ptr + offs, o.to(x.dtype), mask=mask)
    else:
        tl.store(out_ptr + offs, o.to(x.dtype))


def _elu_flat_launch_params(n, dtype):
    block, warps = 16384, 8
    for hi, b, w in _ELU_FLAT_TIERS:
        if hi is None or n <= hi:
            block, warps = b, w
            break
    if dtype in (torch.float16, torch.bfloat16) and n >= _ELU_FAT_MIN_NUMEL:
        block = _ELU_FAT_BLOCK
    return block, warps


def _elu_forward_flat(A, alpha, scale, input_scale, out):
    n = A.numel()
    block, warps = _elu_flat_launch_params(n, A.dtype)
    need_mask = n % block != 0
    grid = (triton.cdiv(n, block),)
    elu_forward_flat_kernel[grid](
        A,
        out,
        n,
        alpha,
        scale,
        input_scale,
        BLOCK=block,
        NEED_MASK=need_mask,
        num_warps=warps,
        buffer_size_limit=_ELU_BSL,
        unroll_num=16,
    )
    return out


def _elu_backward_flat(grad_output, self_or_result, alpha, scale, input_scale, is_result):
    n = grad_output.numel()
    out = torch.empty_like(self_or_result)
    block, warps = _elu_flat_launch_params(n, grad_output.dtype)
    need_mask = n % block != 0
    grid = (triton.cdiv(n, block),)
    elu_backward_flat_kernel[grid](
        grad_output,
        self_or_result,
        out,
        n,
        alpha,
        scale,
        input_scale,
        IS_RESULT=is_result,
        BLOCK=block,
        NEED_MASK=need_mask,
        num_warps=warps,
        buffer_size_limit=_ELU_BSL,
        unroll_num=16,
    )
    return out


def _elu_flat_eligible(t):
    return t.dtype in _ELU_FLAT_DTYPES and t.is_contiguous() and t.numel() > 0


def elu(A, alpha=1.0, scale=1.0, input_scale=1.0):
    logger.debug("GEMS_KUNLUNXIN ELU")
    if _elu_flat_eligible(A):
        return _elu_forward_flat(A, alpha, scale, input_scale, torch.empty_like(A))
    return elu_forward_kernel(A, alpha, scale, input_scale)


def elu_(A, alpha=1.0, scale=1.0, input_scale=1.0):
    logger.debug("GEMS_KUNLUNXIN ELU_")
    if _elu_flat_eligible(A):
        return _elu_forward_flat(A, alpha, scale, input_scale, A)
    return elu_forward_kernel(A, alpha, scale, input_scale, out0=A)


def elu_backward(grad_output, alpha, scale, input_scale, is_result, self_or_result):
    logger.debug("GEMS_KUNLUNXIN ELU_BACKWARD")
    if (
        _elu_flat_eligible(grad_output)
        and self_or_result.dtype in _ELU_FLAT_DTYPES
        and self_or_result.is_contiguous()
        and grad_output.shape == self_or_result.shape
    ):
        return _elu_backward_flat(
            grad_output, self_or_result, alpha, scale, input_scale, is_result
        )
    grad_input = elu_backward_kernel(
        grad_output, self_or_result, alpha, scale, input_scale, is_result
    )
    return grad_input
