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

from ..utils.pointwise_dynamic import pointwise_dynamic

logger = logging.getLogger(__name__)

# Small shapes route through a single-tile kernel that bypasses the
# pointwise_dynamic host layer (dynamic shape/stride/broadcast analysis + task
# build + arg packing), which dominates small numel on XPU3. Measured device
# collapse point for this two-input exp/div op is numel=4096 (fp32 regresses);
# 2048 is the largest safe power-of-2 cap with an all-dtype win.
_FAST_CAP = 2048
_FAST_DTYPES = (torch.float16, torch.float32, torch.bfloat16)


@triton.jit
def _log_sigmoid_backward_single_tile_kernel(
    grad_ptr, self_ptr, out_ptr, numel, TILE: tl.constexpr
):
    tid = tl.arange(0, TILE)
    mask = tid < numel
    g = tl.load(grad_ptr + tid, mask=mask)
    x = tl.load(self_ptr + tid, mask=mask).to(tl.float32)
    z = tl.exp(-tl.abs(x))
    r = 1.0 / (1.0 + z)
    d = tl.where(x < 0.0, 1.0, z) * r
    out = g.to(tl.float32) * d
    tl.store(out_ptr + tid, out.to(out_ptr.dtype.element_ty), mask=mask)


def _fast_eligible(grad_output, self):
    numel = grad_output.numel()
    return (
        0 < numel <= _FAST_CAP
        and grad_output.dtype == self.dtype
        and grad_output.dtype in _FAST_DTYPES
        and grad_output.shape == self.shape
        and grad_output.is_contiguous()
        and self.is_contiguous()
    )


def _fast_launch(grad_output, self, out):
    numel = grad_output.numel()
    TILE = triton.next_power_of_2(numel)
    _log_sigmoid_backward_single_tile_kernel[(1,)](
        grad_output, self, out, numel, TILE=TILE, num_warps=4
    )
    return out


@pointwise_dynamic(is_tensor=[True, True], promotion_methods=[(0, 1, "DEFAULT")])
@triton.jit
def log_sigmoid_backward_kernel(grad_output, self):
    # Recompute the derivative from `self` instead of consuming the buffer.
    # ATen's forward contract defines buffer == exp(-|self|), so both formulas
    # are mathematically identical, but the buffer produced by the vendor
    # log_sigmoid_forward on XPU cannot be trusted (the autograd test exercises
    # a full-size garbage buffer returned by the native forward).
    #
    # The two-branch form `where(x < 0, 1 / (1 + z), z / (1 + z))` evaluates
    # BOTH divisions (SIMT), and XPU division is expensive (~150us per 16M
    # fp32 division over the mul floor). Hoisting the shared denominator into
    # a single reciprocal is mathematically identical:
    #   sigmoid(-x) = where(x < 0, 1, z) * (1 / (1 + z)),  z = exp(-|x|)
    self_fp32 = self.to(tl.float32)
    z = tl.exp(-tl.abs(self_fp32))
    r = 1.0 / (1.0 + z)
    derivative = tl.where(self_fp32 < 0.0, 1.0, z) * r
    return grad_output * derivative


def log_sigmoid_backward(grad_output, self, buffer):
    logger.debug("GEMS_KUNLUNXIN LOG_SIGMOID_BACKWARD")

    del buffer
    if _fast_eligible(grad_output, self):
        return _fast_launch(grad_output, self, torch.empty_like(grad_output))
    return log_sigmoid_backward_kernel(grad_output, self)


def log_sigmoid_backward_out(grad_output, self, buffer, *, grad_input):
    logger.debug("GEMS_KUNLUNXIN LOG_SIGMOID_BACKWARD_OUT")

    del buffer
    if _fast_eligible(grad_output, self) and grad_input.is_contiguous():
        return _fast_launch(grad_output, self, grad_input)
    return log_sigmoid_backward_kernel(grad_output, self, out0=grad_input)
