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

import flag_gems.ops.thnn_differentiable_lstm_cell_backward as _l
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)

_lstm_cell_backward_kernel = _l._lstm_cell_backward_kernel
_validate_inputs = _l._validate_inputs
_differentiable_lstm_cell_backward = _l._differentiable_lstm_cell_backward


@libentry()
@triton.jit(do_not_specialize=["batch_size"])
def _lstm_bias_backward_kernel(
    grad_gates,
    grad_bias,
    batch_size,
    gate_size,
    BLOCK_GATE: tl.constexpr,
    IS_FP64: tl.constexpr,
):
    pid = tl.program_id(0)
    offs_g = pid * BLOCK_GATE + tl.arange(0, BLOCK_GATE)
    g_mask = offs_g < gate_size
    if IS_FP64:
        acc = tl.zeros((BLOCK_GATE,), dtype=tl.float64)
    else:
        acc = tl.zeros((BLOCK_GATE,), dtype=tl.float32)
    for row in range(0, batch_size):
        values = tl.load(grad_gates + row * gate_size + offs_g, mask=g_mask, other=0.0)
        if IS_FP64:
            acc += values.to(tl.float64)
        else:
            acc += values.to(tl.float32)
    tl.store(grad_bias + offs_g, acc.to(grad_bias.dtype.element_ty), mask=g_mask)


def _thnn_differentiable_lstm_cell_backward(
    grad_hy, grad_cy, input_gates, hidden_gates, input_bias, hidden_bias, cx, cy
):
    logger.debug("GEMS_KUNLUNXIN _THNN_DIFFERENTIABLE_LSTM_CELL_BACKWARD")

    if grad_hy is None and grad_cy is None:
        return None, None, None, None, None

    _validate_inputs(
        grad_hy,
        grad_cy,
        input_gates,
        hidden_gates,
        input_bias,
        hidden_bias,
        cx,
        cy,
    )
    differentiable_inputs = (
        grad_hy,
        grad_cy,
        input_gates,
        hidden_gates,
        input_bias,
        hidden_bias,
        cx,
        cy,
    )
    if torch.is_grad_enabled() and any(
        value is not None and value.requires_grad for value in differentiable_inputs
    ):
        return _differentiable_lstm_cell_backward(*differentiable_inputs)

    batch_size, hidden_size = cx.shape
    gate_size = 4 * hidden_size
    grad_gates = torch.empty(
        (batch_size, gate_size),
        dtype=input_gates.dtype,
        device=input_gates.device,
    )
    grad_cx = torch.empty_like(cx, memory_format=torch.contiguous_format)

    n_elements = batch_size * hidden_size
    if n_elements != 0:
        block_size = 256
        input_bias_arg = input_bias if input_bias is not None else input_gates
        hidden_bias_arg = hidden_bias if hidden_bias is not None else hidden_gates
        grad_hy_arg = grad_hy if grad_hy is not None else cx
        grad_cy_arg = grad_cy if grad_cy is not None else cx
        with torch_device_fn.device(input_gates.device):
            _lstm_cell_backward_kernel[(triton.cdiv(n_elements, block_size),)](
                grad_hy_arg,
                grad_cy_arg,
                input_gates,
                hidden_gates,
                input_bias_arg,
                hidden_bias_arg,
                cx,
                cy,
                grad_gates,
                grad_cx,
                batch_size,
                hidden_size,
                grad_hy_arg.stride(0),
                grad_hy_arg.stride(1),
                grad_cy_arg.stride(0),
                grad_cy_arg.stride(1),
                input_gates.stride(0),
                input_gates.stride(1),
                hidden_gates.stride(0),
                hidden_gates.stride(1),
                input_bias_arg.stride(0),
                hidden_bias_arg.stride(0),
                cx.stride(0),
                cx.stride(1),
                cy.stride(0),
                cy.stride(1),
                BLOCK_SIZE=block_size,
                HAS_GRAD_HY=grad_hy is not None,
                HAS_GRAD_CY=grad_cy is not None,
                HAS_INPUT_BIAS=input_bias is not None,
                HAS_HIDDEN_BIAS=hidden_bias is not None,
                IS_FP64=input_gates.dtype == torch.float64,
                IS_FP16=input_gates.dtype == torch.float16,
                IS_BF16=input_gates.dtype == torch.bfloat16,
            )

    if input_bias is None:
        return grad_gates, grad_gates, grad_cx, None, None

    grad_bias = torch.empty(
        (gate_size,), dtype=input_gates.dtype, device=input_gates.device
    )
    if gate_size != 0:
        block_gate = 64
        with torch_device_fn.device(input_gates.device):
            _lstm_bias_backward_kernel[(triton.cdiv(gate_size, block_gate),)](
                grad_gates,
                grad_bias,
                batch_size,
                gate_size,
                BLOCK_GATE=block_gate,
                IS_FP64=input_gates.dtype == torch.float64,
            )
    return grad_gates, grad_gates, grad_cx, grad_bias, grad_bias
