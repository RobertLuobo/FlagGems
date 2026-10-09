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

import flag_gems.ops.thnn_fused_gru_cell_backward as _g
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)

_gru_cell_backward_kernel = _g._gru_cell_backward_kernel
_validate_inputs = _g._validate_inputs
_prepare_out = _g._prepare_out


@libentry()
@triton.jit(do_not_specialize=["batch_size"])
def _gru_bias_grad_kernel(
    grad_input_gates,
    grad_hidden_gates,
    grad_input_bias,
    grad_hidden_bias,
    batch_size,
    gate_size,
    grad_input_stride_0,
    grad_input_stride_1,
    grad_hidden_stride_0,
    grad_hidden_stride_1,
    grad_input_bias_stride,
    grad_hidden_bias_stride,
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
        input_vals = tl.load(
            grad_input_gates + row * grad_input_stride_0 + offs_g * grad_input_stride_1,
            mask=g_mask,
            other=0.0,
        )
        hidden_vals = tl.load(
            grad_hidden_gates
            + row * grad_hidden_stride_0
            + offs_g * grad_hidden_stride_1,
            mask=g_mask,
            other=0.0,
        )
        if IS_FP64:
            input_acc += input_vals.to(tl.float64)
            hidden_acc += hidden_vals.to(tl.float64)
        else:
            input_acc += input_vals.to(tl.float32)
            hidden_acc += hidden_vals.to(tl.float32)
    tl.store(
        grad_input_bias + offs_g * grad_input_bias_stride,
        input_acc.to(grad_input_bias.dtype.element_ty),
        mask=g_mask,
    )
    tl.store(
        grad_hidden_bias + offs_g * grad_hidden_bias_stride,
        hidden_acc.to(grad_hidden_bias.dtype.element_ty),
        mask=g_mask,
    )


def _launch(
    grad_hy,
    workspace,
    has_bias,
    grad_input_gates,
    grad_hidden_gates,
    grad_hx,
    grad_input_bias,
    grad_hidden_bias,
):
    batch_size, hidden_size = grad_hy.shape
    n_elements = batch_size * hidden_size
    gate_size = 3 * hidden_size
    with torch_device_fn.device(grad_hy.device):
        if n_elements != 0:
            block_size = 256
            _gru_cell_backward_kernel[(triton.cdiv(n_elements, block_size),)](
                grad_hy,
                workspace,
                grad_input_gates,
                grad_hidden_gates,
                grad_hx,
                batch_size,
                hidden_size,
                grad_hy.stride(0),
                grad_hy.stride(1),
                workspace.stride(0),
                workspace.stride(1),
                grad_input_gates.stride(0),
                grad_input_gates.stride(1),
                grad_hidden_gates.stride(0),
                grad_hidden_gates.stride(1),
                grad_hx.stride(0),
                grad_hx.stride(1),
                BLOCK_SIZE=block_size,
                IS_FP64=grad_hy.dtype == torch.float64,
            )
        if has_bias and gate_size != 0:
            block_gate = min(triton.next_power_of_2(gate_size), 64)
            grid = (triton.cdiv(gate_size, block_gate),)
            _gru_bias_grad_kernel[grid](
                grad_input_gates,
                grad_hidden_gates,
                grad_input_bias,
                grad_hidden_bias,
                batch_size,
                gate_size,
                grad_input_gates.stride(0),
                grad_input_gates.stride(1),
                grad_hidden_gates.stride(0),
                grad_hidden_gates.stride(1),
                grad_input_bias.stride(0),
                grad_hidden_bias.stride(0),
                BLOCK_GATE=block_gate,
                IS_FP64=grad_hy.dtype == torch.float64,
            )


def _thnn_fused_gru_cell_backward(
    grad_hy: torch.Tensor, workspace: torch.Tensor, has_bias: bool
):
    logger.debug("GEMS_KUNLUNXIN _THNN_FUSED_GRU_CELL_BACKWARD")
    _validate_inputs(grad_hy, workspace)
    batch_size, hidden_size = grad_hy.shape
    gate_shape = (batch_size, 3 * hidden_size)
    options = {"dtype": grad_hy.dtype, "device": grad_hy.device}
    grad_input_gates = torch.empty(gate_shape, **options)
    grad_hidden_gates = torch.empty(gate_shape, **options)
    grad_hx = torch.empty((batch_size, hidden_size), **options)
    if has_bias:
        bias_shape = (3 * hidden_size,)
        grad_input_bias = torch.empty(bias_shape, **options)
        grad_hidden_bias = torch.empty(bias_shape, **options)
    else:
        grad_input_bias = torch.empty(0, **options)
        grad_hidden_bias = torch.empty(0, **options)

    _launch(
        grad_hy,
        workspace,
        has_bias,
        grad_input_gates,
        grad_hidden_gates,
        grad_hx,
        grad_input_bias,
        grad_hidden_bias,
    )
    return (
        grad_input_gates,
        grad_hidden_gates,
        grad_hx,
        grad_input_bias,
        grad_hidden_bias,
    )


def _thnn_fused_gru_cell_backward_out(
    grad_hy: torch.Tensor,
    workspace: torch.Tensor,
    has_bias: bool,
    *,
    out0: torch.Tensor,
    out1: torch.Tensor,
    out2: torch.Tensor,
    out3: torch.Tensor,
    out4: torch.Tensor,
):
    logger.debug("GEMS_KUNLUNXIN _THNN_FUSED_GRU_CELL_BACKWARD_OUT")
    _validate_inputs(grad_hy, workspace)
    if not has_bias:
        raise RuntimeError("the out overload requires has_bias=True")
    batch_size, hidden_size = grad_hy.shape
    gate_shape = (batch_size, 3 * hidden_size)
    bias_shape = (3 * hidden_size,)
    for out, shape, name in (
        (out0, gate_shape, "out0"),
        (out1, gate_shape, "out1"),
        (out2, (batch_size, hidden_size), "out2"),
        (out3, bias_shape, "out3"),
        (out4, bias_shape, "out4"),
    ):
        _prepare_out(out, shape, grad_hy.dtype, grad_hy.device, name)

    _launch(
        grad_hy,
        workspace,
        True,
        out0,
        out1,
        out2,
        out3,
        out4,
    )
    return out0, out1, out2, out3, out4
