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
import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)

_g = importlib.import_module("flag_gems.ops._thnn_differentiable_gru_cell_backward")
_gru_cell_backward_kernel = _g._gru_cell_backward_kernel
_validate_inputs = _g._validate_inputs


@libentry()
@triton.jit(do_not_specialize=["batch_size"])
def _gru_bias_backward_kernel(
    grad_input_gates,
    grad_hidden_gates,
    grad_input_bias,
    grad_hidden_bias,
    batch_size,
    gate_size,
    BLOCK_GATE: tl.constexpr,
    IS_FP64: tl.constexpr,
):
    pid = tl.program_id(0)
    offs_g = pid * BLOCK_GATE + tl.arange(0, BLOCK_GATE)
    g_mask = offs_g < gate_size
    if IS_FP64:
        input_acc = tl.zeros((BLOCK_GATE,), dtype=tl.float64)
        hidden_acc = tl.zeros((BLOCK_GATE,), dtype=tl.float64)
    else:
        input_acc = tl.zeros((BLOCK_GATE,), dtype=tl.float32)
        hidden_acc = tl.zeros((BLOCK_GATE,), dtype=tl.float32)
    for row in range(0, batch_size):
        offsets = row * gate_size + offs_g
        input_values = tl.load(grad_input_gates + offsets, mask=g_mask, other=0.0)
        hidden_values = tl.load(grad_hidden_gates + offsets, mask=g_mask, other=0.0)
        if IS_FP64:
            input_acc += input_values.to(tl.float64)
            hidden_acc += hidden_values.to(tl.float64)
        else:
            input_acc += input_values.to(tl.float32)
            hidden_acc += hidden_values.to(tl.float32)
    tl.store(
        grad_input_bias + offs_g,
        input_acc.to(grad_input_bias.dtype.element_ty),
        mask=g_mask,
    )
    tl.store(
        grad_hidden_bias + offs_g,
        hidden_acc.to(grad_hidden_bias.dtype.element_ty),
        mask=g_mask,
    )


def _thnn_differentiable_gru_cell_backward(
    grad_hy: torch.Tensor,
    input_gates: torch.Tensor,
    hidden_gates: torch.Tensor,
    hx: torch.Tensor,
    input_bias: torch.Tensor = None,
    hidden_bias: torch.Tensor = None,
):
    logger.debug("GEMS_KUNLUNXIN _THNN_DIFFERENTIABLE_GRU_CELL_BACKWARD")
    _validate_inputs(grad_hy, input_gates, hidden_gates, hx)

    batch_size, hidden_size = hx.shape
    gate_size = 3 * hidden_size
    grad_input_gates = torch.empty(
        (batch_size, gate_size), dtype=input_gates.dtype, device=input_gates.device
    )
    grad_hidden_gates = torch.empty(
        (batch_size, gate_size), dtype=hidden_gates.dtype, device=hidden_gates.device
    )
    grad_hx = torch.empty((batch_size, hidden_size), dtype=hx.dtype, device=hx.device)

    n_elements = batch_size * hidden_size
    if n_elements != 0:
        block_size = 256
        grid = (triton.cdiv(n_elements, block_size),)
        input_bias_arg = input_bias if input_bias is not None else input_gates
        hidden_bias_arg = hidden_bias if hidden_bias is not None else hidden_gates
        with torch_device_fn.device(grad_hy.device):
            _gru_cell_backward_kernel[grid](
                grad_hy,
                input_gates,
                hidden_gates,
                hx,
                input_bias_arg,
                hidden_bias_arg,
                grad_input_gates,
                grad_hidden_gates,
                grad_hx,
                batch_size,
                hidden_size,
                grad_hy.stride(0),
                grad_hy.stride(1),
                input_gates.stride(0),
                input_gates.stride(1),
                hidden_gates.stride(0),
                hidden_gates.stride(1),
                hx.stride(0),
                hx.stride(1),
                input_bias_arg.stride(0),
                hidden_bias_arg.stride(0),
                BLOCK_SIZE=block_size,
                HAS_INPUT_BIAS=input_bias is not None,
                HAS_HIDDEN_BIAS=hidden_bias is not None,
                IS_FP64=input_gates.dtype == torch.float64,
                IS_FP16=input_gates.dtype == torch.float16,
                IS_BF16=input_gates.dtype == torch.bfloat16,
            )

    if input_bias is None:
        return grad_input_gates, grad_hidden_gates, grad_hx, None, None

    grad_input_bias = torch.empty(
        (gate_size,), dtype=input_gates.dtype, device=input_gates.device
    )
    grad_hidden_bias = torch.empty(
        (gate_size,), dtype=hidden_gates.dtype, device=hidden_gates.device
    )
    if gate_size != 0:
        block_gate = min(triton.next_power_of_2(gate_size), 64)
        grid = (triton.cdiv(gate_size, block_gate),)
        with torch_device_fn.device(grad_hy.device):
            _gru_bias_backward_kernel[grid](
                grad_input_gates,
                grad_hidden_gates,
                grad_input_bias,
                grad_hidden_bias,
                batch_size,
                gate_size,
                BLOCK_GATE=block_gate,
                IS_FP64=input_gates.dtype == torch.float64,
            )
    return (
        grad_input_gates,
        grad_hidden_gates,
        grad_hx,
        grad_input_bias,
        grad_hidden_bias,
    )
