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

import importlib
import math

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import triton_lang_extension as ext

_gen = importlib.import_module("flag_gems.fused.post_layer_norm_residual")
_ln = importlib.import_module("flag_gems.runtime.backend._kunlunxin.ops.layernorm")

_ONE_PASS_HEURISTICS = _gen._ONE_PASS_HEURISTICS
_normalize_shape = _gen._normalize_shape
_validate_inputs = _gen._validate_inputs


# The stock kunlunxin weight_bias_backward_1d_kernel stores a full C-wide vector
# without a column mask. When N is not a multiple of the column tile (need_tail),
# the tail block overruns the destination buffer; on the two-pass path that
# overrun corrupts the low columns of the next partial row and yields wrong
# grad_weight / grad_bias (observed for M=1024, N=513). This copy adds the
# missing store masks; the reduction body is otherwise identical.
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
    cols = tl.arange(0, C)
    cmask = n0 + cols < N
    if not NEED_TAIL:
        for r in range(0, BM):
            m = m0 + r
            base = m * N + n0
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
            dy = tl.load(dY + base + cols, mask=cmask, other=0.0).to(tl.float32)
            x = tl.load(X + base + cols, mask=cmask, other=0.0).to(tl.float32)
            mean = tl.load(Mean + m).to(tl.float32)
            rstd = tl.load(Rstd + m).to(tl.float32)
            x = tl.where(cmask, x - mean, 0.0)
            accW += tl.where(cmask, dy, 0.0) * (x * rstd)
            accB += tl.where(cmask, dy, 0.0)
    smask = cmask if NEED_TAIL else None
    if DIRECT:
        if OutW is not None:
            tl.store(OutW + n0 + cols, accW, mask=smask)
        if OutB is not None:
            tl.store(OutB + n0 + cols, accB, mask=smask)
    else:
        if OutW is not None:
            tl.store(OutW + mi * N + n0 + cols, accW, mask=smask)
        if OutB is not None:
            tl.store(OutB + mi * N + n0 + cols, accB, mask=smask)


def _weight_bias_backward(grad_2d, input_2d, mean, rstd, need_weight, need_bias, N):
    M = input_2d.shape[0]
    bc = _ln._ln_bwd_col_size(N)
    need_tail = N % bc != 0

    weight_grad = (
        torch.empty(N, dtype=torch.float32, device=input_2d.device)
        if need_weight
        else None
    )
    bias_grad = (
        torch.empty(N, dtype=torch.float32, device=input_2d.device)
        if need_bias
        else None
    )

    bm = _ln._wb_bm_size(M)
    with torch_device_fn.device(input_2d.device):
        if bm >= M:
            _wb_backward_1d_masked_kernel[(triton.cdiv(N, bc), 1, 1)](
                grad_2d,
                input_2d,
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
                torch.empty((P, N), dtype=torch.float32, device=input_2d.device)
                if weight_grad is not None
                else None
            )
            pb = (
                torch.empty((P, N), dtype=torch.float32, device=input_2d.device)
                if bias_grad is not None
                else None
            )
            _wb_backward_1d_masked_kernel[(triton.cdiv(N, bc), P, 1)](
                grad_2d,
                input_2d,
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
            _ln.weight_bias_backward_finish_kernel[(triton.cdiv(N, bc), 1, 1)](
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
            _gen.post_layer_norm_residual_one_pass_kernel[grid](
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
            _gen.post_layer_norm_residual_loop_kernel[(M, 1, 1)](
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

        input_2d = input.reshape(ctx.M, ctx.N)
        grad_2d = grad_output.contiguous().reshape(ctx.M, ctx.N)
        weight_1d = None if weight is None else weight.reshape(ctx.N)

        grad_input = None
        if need_input:
            # grad_input path in layer_norm_backward masks its stores correctly,
            # so it is safe to reuse; the weight/bias reduction below is handled
            # by the masked kernel in this overlay to avoid the tail overrun.
            from flag_gems import layer_norm_backward

            grad_input, _, _ = layer_norm_backward(
                grad_2d,
                input_2d,
                (ctx.N,),
                mean,
                rstd,
                weight_1d,
                None,
                [True, False, False],
            )

        grad_weight = None
        grad_bias = None
        if need_weight or need_bias:
            grad_weight, grad_bias = _weight_bias_backward(
                grad_2d, input_2d, mean, rstd, need_weight, need_bias, ctx.N
            )
            in_dtype = input.dtype
            if grad_weight is not None:
                grad_weight = grad_weight.to(in_dtype)
            if grad_bias is not None:
                grad_bias = grad_bias.to(in_dtype)

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
