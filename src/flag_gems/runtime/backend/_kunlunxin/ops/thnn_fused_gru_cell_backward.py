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

from flag_gems.ops.thnn_fused_gru_cell_backward import (
    _gru_cell_backward_kernel,
    _prepare_out,
    _validate_inputs,
)
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)


@libentry()
@triton.jit
def _gru_bias_backward_scalar_kernel(
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
    gate_block = tl.program_id(0)
    gate_idx = gate_block * BLOCK_GATE + tl.arange(0, BLOCK_GATE)
    gate_mask = gate_idx < gate_size
    if IS_FP64:
        input_accumulator = tl.zeros((BLOCK_GATE,), dtype=tl.float64)
        hidden_accumulator = tl.zeros((BLOCK_GATE,), dtype=tl.float64)
    else:
        input_accumulator = tl.zeros((BLOCK_GATE,), dtype=tl.float32)
        hidden_accumulator = tl.zeros((BLOCK_GATE,), dtype=tl.float32)

    for row in range(batch_size):
        input_value = tl.load(
            grad_input_gates
            + row * grad_input_stride_0
            + gate_idx * grad_input_stride_1,
            mask=gate_mask,
            other=0.0,
        )
        hidden_value = tl.load(
            grad_hidden_gates
            + row * grad_hidden_stride_0
            + gate_idx * grad_hidden_stride_1,
            mask=gate_mask,
            other=0.0,
        )
        if IS_FP64:
            input_accumulator += input_value.to(tl.float64)
            hidden_accumulator += hidden_value.to(tl.float64)
        else:
            input_accumulator += input_value.to(tl.float32)
            hidden_accumulator += hidden_value.to(tl.float32)

    tl.store(
        grad_input_bias + gate_idx * grad_input_bias_stride,
        input_accumulator,
        mask=gate_mask,
    )
    tl.store(
        grad_hidden_bias + gate_idx * grad_hidden_bias_stride,
        hidden_accumulator,
        mask=gate_mask,
    )


def _launch_gru_cell_backward(
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
        gate_size = 3 * hidden_size
        if has_bias and gate_size != 0:
            block_gate = 64
            _gru_bias_backward_scalar_kernel[(triton.cdiv(gate_size, block_gate),)](
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
                num_stages=1,
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
        grad_input_bias = None
        grad_hidden_bias = None

    _launch_gru_cell_backward(
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

    _launch_gru_cell_backward(
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
