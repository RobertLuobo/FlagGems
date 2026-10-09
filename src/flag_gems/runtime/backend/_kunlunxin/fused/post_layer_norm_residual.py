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

import math

import torch
import triton
import triton.language as tl

from flag_gems.fused.post_layer_norm_residual import (
    _ONE_PASS_HEURISTICS,
    _normalize_shape,
    _validate_inputs,
    post_layer_norm_residual_loop_kernel,
    post_layer_norm_residual_one_pass_kernel,
)
from flag_gems.runtime import torch_device_fn
from flag_gems.runtime.backend._kunlunxin.ops.layernorm import (
    _ln_bwd_col_size,
    _wb_bm_size,
    layer_norm_backward,
    weight_bias_backward_finish_kernel,
)
from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as ext

@libentry()
@triton.jit
def _wb_backward_1d_masked_kernel(
    dY,
    X,
    Mean,
    Rstd,
    OutW,
    OutB,
    M,
    N,
    BM: tl.constexpr,
    C: tl.constexpr,
    NEED_TAIL: tl.constexpr,
    DIRECT: tl.constexpr,
):
    n0 = ext.program_id(0) * C
    mi = ext.program_id(1)
    m0 = mi * BM
    accW = tl.zeros([C], dtype=tl.float32)
    accB = tl.zeros([C], dtype=tl.float32)
    if not NEED_TAIL:
        for r in range(0, BM):
            m = m0 + r
            base = m * N + n0
            cols = tl.arange(0, C)
            dy = tl.load(dY + base + cols).to(tl.float32)
            x = tl.load(X + base + cols).to(tl.float32)
            mean = tl.load(Mean + m).to(tl.float32)
            rstd = tl.load(Rstd + m).to(tl.float32)
            accW += dy * ((x - mean) * rstd)
            accB += dy
    else:
        for r in range(0, BM):
            m = m0 + r
            base = m * N + n0
            cols = tl.arange(0, C)
            cmask = n0 + cols < N
            dy = tl.load(dY + base + cols, mask=cmask, other=0.0).to(tl.float32)
            x = tl.load(X + base + cols, mask=cmask, other=0.0).to(tl.float32)
            mean = tl.load(Mean + m).to(tl.float32)
            rstd = tl.load(Rstd + m).to(tl.float32)
            x = tl.where(cmask, x - mean, 0.0)
            accW += tl.where(cmask, dy, 0.0) * (x * rstd)
            accB += tl.where(cmask, dy, 0.0)
    cols = tl.arange(0, C)
    cmask = n0 + cols < N
    if DIRECT:
        if OutW is not None:
            tl.store(OutW + n0 + cols, accW, mask=cmask)
        if OutB is not None:
            tl.store(OutB + n0 + cols, accB, mask=cmask)
    else:
        if OutW is not None:
            tl.store(OutW + mi * N + n0 + cols, accW, mask=cmask)
        if OutB is not None:
            tl.store(OutB + mi * N + n0 + cols, accB, mask=cmask)


def _weight_bias_backward(grad_out, input, mean, rstd, weight, bias, output_mask):
    grad_out = grad_out.contiguous()
    input = input.contiguous()
    mean = mean.contiguous()
    rstd = rstd.contiguous()
    weight = None if weight is None else weight.contiguous()
    bias = None if bias is None else bias.contiguous()

    M = input.shape[0]
    N = input.numel() // M
    bc = _ln_bwd_col_size(N)
    need_tail = N % bc != 0

    if output_mask[1]:
        weight_grad = torch.empty_strided(
            weight.size(), weight.stride(), dtype=weight.dtype, device=weight.device
        )
    else:
        weight_grad = None
    if output_mask[2]:
        bias_grad = torch.empty_strided(
            bias.size(), bias.stride(), dtype=bias.dtype, device=bias.device
        )
    else:
        bias_grad = None

    bm = _wb_bm_size(M)
    if bm >= M:
        with torch_device_fn.device(input.device):
            _wb_backward_1d_masked_kernel[(triton.cdiv(N, bc), 1, 1)](
                grad_out,
                input,
                mean,
                rstd,
                weight_grad,
                bias_grad,
                M,
                N,
                BM=bm,
                C=bc,
                NEED_TAIL=need_tail,
                DIRECT=True,
                isCloseUnrollControl=True,
            )
    else:
        P = M // bm
        pw = (
            torch.empty_strided(
                (P, N), (N, 1), dtype=torch.float32, device=input.device
            )
            if weight_grad is not None
            else None
        )
        pb = (
            torch.empty_strided(
                (P, N), (N, 1), dtype=torch.float32, device=input.device
            )
            if bias_grad is not None
            else None
        )
        with torch_device_fn.device(input.device):
            _wb_backward_1d_masked_kernel[(triton.cdiv(N, bc), P, 1)](
                grad_out,
                input,
                mean,
                rstd,
                pw,
                pb,
                M,
                N,
                BM=bm,
                C=bc,
                NEED_TAIL=need_tail,
                DIRECT=False,
                isCloseUnrollControl=True,
            )
            weight_bias_backward_finish_kernel[(triton.cdiv(N, bc), 1, 1)](
                pw,
                pb,
                weight_grad,
                bias_grad,
                P,
                N,
                C=bc,
                NEED_TAIL=need_tail,
                isCloseUnrollControl=True,
            )
    return weight_grad, bias_grad


def _post_layer_norm_residual_forward(
    input, residual, normalized_shape, weight, bias, eps, save_stats
):
    N = math.prod(normalized_shape)
    M = input.numel() // N
    output = torch.empty_like(input)

    if save_stats:
        mean = torch.empty(M, dtype=torch.float32, device=input.device)
        rstd = torch.empty_like(mean)
    else:
        mean = output
        rstd = output

    with torch_device_fn.device(input.device):
        if N <= 4096 and N % 8 == 0:
            heuristic_args = {"M": M, "N": N}
            tile_m = _ONE_PASS_HEURISTICS["TILE_M"](heuristic_args)
            tile_n = _ONE_PASS_HEURISTICS["TILE_N"](heuristic_args)
            grid = (triton.cdiv(M, tile_m), 1, 1)
            post_layer_norm_residual_one_pass_kernel[grid](
                input,
                residual,
                output,
                weight,
                bias,
                mean,
                rstd,
                M,
                N,
                eps,
                TILE_M=tile_m,
                TILE_N=tile_n,
                SAVE_STATS=save_stats,
            )
        else:
            post_layer_norm_residual_loop_kernel[(M, 1, 1)](
                input,
                residual,
                output,
                weight,
                bias,
                mean,
                rstd,
                M,
                N,
                eps,
                SAVE_STATS=save_stats,
            )

    return output, mean, rstd, M, N


class PostLayerNormResidual(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, residual, normalized_shape, weight, bias, eps):
        needs_layer_norm_grad = (
            input.requires_grad
            or (weight is not None and weight.requires_grad)
            or (bias is not None and bias.requires_grad)
        )
        output, mean, rstd, M, N = _post_layer_norm_residual_forward(
            input,
            residual,
            normalized_shape,
            weight,
            bias,
            eps,
            save_stats=needs_layer_norm_grad,
        )

        if needs_layer_norm_grad:
            weight_saved = input.new_empty(0) if weight is None else weight
            bias_saved = input.new_empty(0) if bias is None else bias
            ctx.save_for_backward(input, weight_saved, bias_saved, mean, rstd)
            ctx.has_weight = weight is not None
            ctx.has_bias = bias is not None
            ctx.input_shape = input.shape
            ctx.normalized_shape = normalized_shape
            ctx.M = M
            ctx.N = N
        return output

    @staticmethod
    def backward(ctx, grad_output):
        need_input, need_residual, _, need_weight, need_bias, _ = ctx.needs_input_grad
        output_mask = [need_input, need_weight, need_bias]
        if not any(output_mask):
            return None, grad_output if need_residual else None, None, None, None, None

        input, weight_saved, bias_saved, mean, rstd = ctx.saved_tensors
        weight = weight_saved if ctx.has_weight else None
        bias = bias_saved if ctx.has_bias else None

        input_2d = input.reshape(ctx.M, ctx.N)
        grad_2d = grad_output.contiguous().reshape(ctx.M, ctx.N)
        weight_1d = None if weight is None else weight.reshape(ctx.N)
        bias_1d = None if bias is None else bias.reshape(ctx.N)

        grad_input = None
        if need_input:
            grad_input, _, _ = layer_norm_backward(
                grad_2d,
                input_2d,
                (ctx.N,),
                mean,
                rstd,
                weight_1d,
                bias_1d,
                [True, False, False],
            )

        grad_weight = None
        grad_bias = None
        if need_weight or need_bias:
            grad_weight, grad_bias = _weight_bias_backward(
                grad_2d,
                input_2d,
                mean,
                rstd,
                weight_1d,
                bias_1d,
                [False, need_weight, need_bias],
            )

        if grad_input is not None:
            grad_input = grad_input.reshape(ctx.input_shape)
        if grad_weight is not None:
            grad_weight = grad_weight.reshape(ctx.normalized_shape)
        if grad_bias is not None:
            grad_bias = grad_bias.reshape(ctx.normalized_shape)
        grad_residual = grad_output if need_residual else None
        return grad_input, grad_residual, None, grad_weight, grad_bias, None


def post_layer_norm_residual(
    input, residual, normalized_shape, weight=None, bias=None, eps=1e-5
):
    normalized_shape = _normalize_shape(normalized_shape)
    _validate_inputs(input, residual, normalized_shape, weight, bias)

    if (
        input.numel() == 0
        or weight is None
        or bias is None
        or not input.is_contiguous()
        or not residual.is_contiguous()
    ):
        output = torch.layer_norm(input, normalized_shape, weight, None, eps)
        if bias is not None:
            output = output + bias
        return output + residual

    weight = weight.contiguous()
    bias = bias.contiguous()

    needs_autograd = torch.is_grad_enabled() and any(
        tensor.requires_grad for tensor in (input, residual, weight, bias)
    )
    if not needs_autograd:
        output, _, _, _, _ = _post_layer_norm_residual_forward(
            input,
            residual,
            normalized_shape,
            weight,
            bias,
            eps,
            save_stats=False,
        )
        return output

    return PostLayerNormResidual.apply(
        input, residual, normalized_shape, weight, bias, eps
    )

