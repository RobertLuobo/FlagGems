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

import _kunlunxin.utils.pointwise_dynamic as _xpu_pd

from _kunlunxin.utils.codegen_config_utils import CodeGenConfig

from flag_gems.utils import pointwise_dynamic as generic_pointwise_dynamic
from flag_gems.utils import triton_lang_extension as ext

# `import ... as` (an ast.Import) on purpose: the generic
# `flag_gems.utils.pointwise_dynamic` re-emits *ImportFrom* nodes that appear
# in this module into its generated wrapper, and a relative `from ..utils...`
# would be re-emitted as a broken `from utils...` (level dropped, measured
# ModuleNotFoundError). ast.Import nodes are not collected, hence the alias.
xpu_pointwise_dynamic = _xpu_pd.pointwise_dynamic

logger = logging.getLogger(__name__)

# Kunlunxin/XPU override for aten::log_sigmoid_backward{,.grad_input}.
#
# 1) CORRECTNESS. The vendor's native aten::log_sigmoid_forward returns an
#    *uninitialized* `buffer` (probed on XPU: bf16 buffer came back with inf /
#    4.8125 while exp(-|x|) was 0.617). The generic implementation trusts that
#    buffer whenever it matches shape/dtype/contiguity and the dtype is not
#    fp32, so autograd through log_sigmoid produced garbage fp16/bf16 gradients
#    (tests/test_log_sigmoid_backward.py::test_log_sigmoid_backward_via_autograd
#    [dtype0]/[dtype2] failed). This override never reads `buffer` and always
#    recomputes the derivative from `self`, which is exactly what the CUDA ATen
#    kernel does (there `buffer` is empty by construction).
#
# 2) PERFORMANCE. d log_sigmoid(x)/dx = 1 - sigmoid(x) = sigmoid(-x). On
#    TritonXPU `tl.sigmoid(-x)` lowers to a much cheaper sequence than
#    `1 / (1 + exp(x))`: at 16.7M fp16 `1/(1+exp)` is 0.458 ms while
#    `sigmoid(-x)` is 0.244 ms (probes /tmp/lsb_probe2..4.py, XPU 4), and it
#    is bit-identical in accuracy (same fp32 math, maxabsdiff vs CPU reference
#    unchanged: 1.95e-3 fp16 / 4.77e-7 fp32 / 3.91e-3 bf16).
#
#    Two kernel shapes are kept (measured same machine, 2026-09-04):
#      * the flat single-tile-per-CTA kernels below (BLOCK=32768/16384/2048,
#        num_warps 8/4, unroll 2, buffer_size_limit 8192) win up to ~4M
#        elements (1M: 0.021 ms vs 0.075 ms; 4.2M: 0.066 ms vs 0.075 ms);
#      * the 1D-tile codegen with kunlunAutoGrid (12 CTAs, one wide tile per
#        CTA - the proven mse_loss_backward / lt_ / hardsigmoid_backward
#        config) wins from ~16M elements (16.7M bf16: 0.223 ms vs 0.325 ms).
#      The split point is FLAT_MAX_NUMEL; large fp16/fp32 tie within noise so
#      the 12-CTA path is used there for bf16's benefit.
#
#    Derivative algebra: sigmoid(-x) = 1/(1+exp(x)) is unconditionally stable
#    (no overflow): x=-inf -> 0 -> d=1, x=+inf -> d=0, x=-300 -> d=1, x=0 ->
#    0.5, and NaN propagates exactly as the CPU reference (verified elementwise
#    on the edge probe: -inf/inf/-300/0/+-20/nan).
#
# 3) BACKEND TRAP (from the original closure). A masked load written as
#    `tl.load(p + off, mask=mask, other=0.0)` silently corrupts a few percent
#    of the *valid* lanes on TritonXPU when the vendor guards
#    TRITONXPU_OTHER_SIM / TRITONXPU_STORE_MASK_SIM are not exported - and it
#    does so even when the mask is entirely true. The masked kernel below
#    relies on the masked store to discard the tail lanes and never passes
#    `other=`.

UNROLL_NUM = 2
BUFFER_SIZE_LIMIT = 8192
IS_CLOSE_MEMORY_ASYNC = False

# Above this many elements the 12-CTA 1D-tile codegen measures faster than the
# flat per-CTA kernels (see the module docstring, section 2).
FLAT_MAX_NUMEL = 4 * 1024 * 1024

# 12-CTA auto-grid 1D-tile codegen: same proven config as hardsigmoid_backward.
config_ = CodeGenConfig(
    512,
    (65536, 65536, 65536),
    32,
    True,
    prefer_1d_tile=True,
    buffer_size_limit=4096,
    kunlunAutoGrid=True,
    unroll_num=8,
)


@xpu_pointwise_dynamic(promotion_methods=[(0, 1, "DEFAULT")], config=config_)
@triton.jit
def log_sigmoid_backward_func(grad_output, self):
    # 1 - sigmoid(self) == sigmoid(-self) == 1 / (1 + exp(self))
    go = grad_output.to(tl.float32)
    x = self.to(tl.float32)
    return (go * tl.sigmoid(0.0 - x)).to(grad_output.dtype)


def _pick_block(n_elements):
    # Keep the number of compiled variants small: two unmasked tiles for the
    # divisible (benchmark / large tensor) cases and two masked fallbacks.
    if n_elements >= 32768 and n_elements % 32768 == 0:
        return 32768, 8, False
    if n_elements >= 16384 and n_elements % 16384 == 0:
        return 16384, 8, False
    if n_elements <= 65536:
        return 2048, 4, True
    return 16384, 8, True


@triton.jit
def log_sigmoid_backward_flat_kernel(
    grad_output_ptr,
    self_ptr,
    grad_input_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = ext.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    # NEVER pass `other=` here: on TritonXPU a masked load with an explicit
    # `other` value silently corrupts a few percent of the *valid* lanes (see
    # the module docstring), and the masked store already discards the
    # out-of-range lanes.
    g = tl.load(grad_output_ptr + offsets, mask=mask)
    x = tl.load(self_ptr + offsets, mask=mask)
    derivative = tl.sigmoid(0.0 - x.to(tl.float32))
    res = g.to(tl.float32) * derivative
    tl.store(
        grad_input_ptr + offsets,
        res.to(grad_input_ptr.dtype.element_ty),
        mask=mask,
    )


@triton.jit
def log_sigmoid_backward_flat_kernel_unmasked(
    grad_output_ptr,
    self_ptr,
    grad_input_ptr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = ext.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    g = tl.load(grad_output_ptr + offsets)
    x = tl.load(self_ptr + offsets)
    derivative = tl.sigmoid(0.0 - x.to(tl.float32))
    res = g.to(tl.float32) * derivative
    tl.store(grad_input_ptr + offsets, res.to(grad_input_ptr.dtype.element_ty))


@generic_pointwise_dynamic(
    is_tensor=[True, True], promotion_methods=[(0, 1, "DEFAULT")]
)
@triton.jit
def log_sigmoid_backward_pointwise_kernel(grad_output, self):
    # Strided / broadcast / mixed-dtype fallback with the same algebra as the
    # flat kernel above (still a Triton kernel, no ATen redispatch).
    derivative = 1.0 / (1.0 + tl.exp(self.to(tl.float32)))
    return grad_output * derivative


def _can_use_flat_kernel(grad_output, self, grad_input=None):
    return (
        grad_output.shape == self.shape
        and grad_output.dtype == self.dtype
        and grad_output.is_contiguous()
        and self.is_contiguous()
        and (
            grad_input is None
            or (
                grad_input.shape == self.shape
                and grad_input.dtype == self.dtype
                and grad_input.is_contiguous()
            )
        )
    )


def _launch_flat_kernel(grad_output, self, grad_input):
    n_elements = self.numel()
    if n_elements == 0:
        return grad_input
    block_size, num_warps, masked = _pick_block(n_elements)
    if masked:
        grid = (triton.cdiv(n_elements, block_size),)
        log_sigmoid_backward_flat_kernel[grid](
            grad_output,
            self,
            grad_input,
            n_elements,
            BLOCK_SIZE=block_size,
            num_warps=num_warps,
            unroll_num=UNROLL_NUM,
            buffer_size_limit=BUFFER_SIZE_LIMIT,
            isCloseMemoryAsync=IS_CLOSE_MEMORY_ASYNC,
        )
    else:
        grid = (n_elements // block_size,)
        log_sigmoid_backward_flat_kernel_unmasked[grid](
            grad_output,
            self,
            grad_input,
            BLOCK_SIZE=block_size,
            num_warps=num_warps,
            unroll_num=UNROLL_NUM,
            buffer_size_limit=BUFFER_SIZE_LIMIT,
            isCloseMemoryAsync=IS_CLOSE_MEMORY_ASYNC,
        )
    return grad_input


def log_sigmoid_backward(grad_output, self, buffer):
    logger.debug("GEMS_KUNLUNXIN LOG_SIGMOID_BACKWARD")
    # `buffer` is intentionally unused: the vendor forward leaves it
    # uninitialized (see the module docstring above).
    if _can_use_flat_kernel(grad_output, self):
        if self.numel() > FLAT_MAX_NUMEL:
            return log_sigmoid_backward_func(grad_output, self)
        return _launch_flat_kernel(grad_output, self, torch.empty_like(self))
    return log_sigmoid_backward_pointwise_kernel(grad_output, self)


def log_sigmoid_backward_out(grad_output, self, buffer, *, grad_input):
    logger.debug("GEMS_KUNLUNXIN LOG_SIGMOID_BACKWARD OUT")
    if _can_use_flat_kernel(grad_output, self, grad_input):
        if self.numel() > FLAT_MAX_NUMEL:
            return log_sigmoid_backward_func(grad_output, self, out0=grad_input)
        return _launch_flat_kernel(grad_output, self, grad_input)
    return log_sigmoid_backward_pointwise_kernel(grad_output, self, out0=grad_input)