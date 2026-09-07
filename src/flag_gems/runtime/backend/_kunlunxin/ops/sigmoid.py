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

from flag_gems.utils import tl_extra_shim
from flag_gems.utils import triton_lang_extension as ext

from ..utils.pointwise_dynamic import pointwise_dynamic

logger = logging.getLogger(__name__)
exp2 = tl_extra_shim.exp2


config_ = CodeGenConfig(
    512,
    (65536, 65536, 65536),
    32,
    True,
    prefer_1d_tile=True,
    buffer_size_limit=4096,
    isCloseVectorization=True,
    kunlunAutoGrid=True,
    unroll_num=8,
)


@pointwise_dynamic(promotion_methods=[(0, "INT_TO_FLOAT")], config=config_)
@triton.jit
def sigmoid_forward(x):
    # log2e: tl.constexpr = math.log2(math.e)
    # triton 3.0.0 disallow calling non-jitted function inside jitted function, even if it is in
    # the rhs of an assignment to a constexpr, so we use numeric literal instead to work around this.
    # log2e: tl.constexpr = 1.4426950408889634
    return 1 / (1 + tl.exp(-x.to(tl.float32)))


@pointwise_dynamic(promotion_methods=[(0, "INT_TO_FLOAT")])
@triton.jit
def sigmoid_backward_kernel(dy, y):
    y_f32 = y.to(tl.float32)
    dy_f32 = dy.to(tl.float32)
    return dy_f32 * (1.0 - y_f32) * y_f32


# sigmoid_backward fast path (contiguous fp16/fp32/bf16): a flat 1D kernel
# that skips the pointwise_dynamic wrapper machinery.  The wrapper's
# Python-side per-call dispatch (type promotion, StridedBuffer wrapping,
# generated launcher) shows up directly in the measured autograd.grad latency
# and grows to tens of microseconds on this platform under host contention;
# the flat launcher is a plain JIT launch with ~1-3us of overhead.
#
# The XRE emulator's triton fork only takes the wide-load/vectorized codegen
# path when the launch is given buffer_size_limit=2048; without it the same
# kernel is 1.5-2x slower.  With bsl the best flat block size grows with the
# tensor (measured, median of 3x do_bench(rep=100), W=8):
#     <=8K elements     single CTA (next pow2, unmasked when divisible)
#     8K-64K            BLOCK=8192
#     128K-2M           BLOCK=32768      (beats 65536 by 1.1-1.3x at 256K/1M)
#     >=4M              BLOCK=131072     (beats 65536 by 1.03-1.1x at 4M+;
#                                         262144 is worse than 131072)
# vs the generated grid=12-cluster pointwise kernel.  The generated kernel
# wins for bf16 at >=512K (its cluster12 + constexpr num_tasks shape handles
# the bf16->fp32 conversion far better than the flat kernel: 0.85 vs 0.52 at
# 16.7M), so those sizes route through _flat_block -> 0 and keep the
# generated kernel.  For bf16 <= 256K the flat kernel wins or ties, and it
# additionally avoids a second output allocation (3.5us) that the routed
# pointwise path would pay, so the flat path is used end-to-end there.
#
# Math: t = dy * y; out = t - t * y  (fused-multiply-add form, identical to
# dy * (1 - y) * y up to fp32 rounding), computed in fp32, downcast at store.
_FAST_WARPS = 8
_FAST_BSL = 2048  # enables XRE wide-load codegen; see comment above
_TINY_BLOCK = 2048
_TINY_WARPS = 4


def _flat_block(dtype, numel):
    """Flat-kernel BLOCK for (dtype, numel); 0 means use generated pointwise."""
    if dtype == torch.bfloat16:
        if numel <= 8192:
            return triton.next_power_of_2(numel)
        if numel <= 131072:
            return 8192  # flat B=8192 ties the generated pointwise here, and
            # only the flat path is measurable through the op without a
            # second output allocation (routed-PW is ~5us slower)
        if numel <= 262144:
            return 8192  # flat B=8192 beats pointwise at 256K (0.72 vs 0.69)
        return 0  # >=512K: generated pointwise wins (0.82-0.87 spd)
    # float16 / float32: flat beats the generated pointwise on every band.
    if numel <= 8192:
        return triton.next_power_of_2(numel)
    if numel < 16384:
        # odd sizes in (8K, 16K): single masked CTA, no bsl (see launcher)
        return 16384
    if numel <= 65536:
        return 8192
    if numel <= 2097152:
        return 32768
    return 131072


@triton.jit
def sigmoid_backward_fast_kernel(dy_ptr, y_ptr, out_ptr, BLOCK: tl.constexpr):
    pid = ext.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    y = tl.load(y_ptr + offs).to(tl.float32)
    dy = tl.load(dy_ptr + offs).to(tl.float32)
    t = dy * y
    tl.store(out_ptr + offs, (t - t * y).to(out_ptr.dtype.element_ty))


@triton.jit
def sigmoid_backward_masked_kernel(
    dy_ptr, y_ptr, out_ptr, start, numel, BLOCK: tl.constexpr
):
    pid = ext.program_id(0)
    offs = pid * BLOCK + start + tl.arange(0, BLOCK)
    mask = offs < numel
    y = tl.load(y_ptr + offs, mask=mask).to(tl.float32)
    dy = tl.load(dy_ptr + offs, mask=mask).to(tl.float32)
    t = dy * y
    tl.store(
        out_ptr + offs,
        (t - t * y).to(out_ptr.dtype.element_ty),
        mask=mask,
    )


def _sigmoid_backward_fast(grad_output, output):
    numel = output.numel()
    block = _flat_block(output.dtype, numel)
    if block == 0:
        # bf16 sizes where the generated cluster12 kernel wins outright.
        # Allocate nothing here: the wrapper allocates its own output, and a
        # second empty_strided costs ~3.5us on this device.
        return sigmoid_backward_kernel(grad_output, output)
    # Allocate via empty_strided (unregistered by gems) to dodge the
    # registered-empty dispatch tax inside use_gems contexts.
    out = torch.empty_strided(
        output.shape, output.stride(), dtype=output.dtype, device=output.device
    )
    if numel == 0:
        return out
    if numel <= block:
        # Single-CTA region: no buffer_size_limit here -- the XRE codegen
        # path it enables is only beneficial (and only verified) for
        # multi-CTA launches; single-CTA launches keep the plain config.
        if numel < _TINY_BLOCK:
            b = _TINY_BLOCK
            sigmoid_backward_masked_kernel[(1,)](
                grad_output,
                output,
                out,
                0,
                numel,
                BLOCK=b,
                num_warps=_TINY_WARPS,
            )
            return out
        b = triton.next_power_of_2(numel)
        if numel % b == 0:
            sigmoid_backward_fast_kernel[(numel // b,)](
                grad_output, output, out, BLOCK=b, num_warps=_FAST_WARPS
            )
        else:
            sigmoid_backward_masked_kernel[(1,)](
                grad_output,
                output,
                out,
                0,
                numel,
                BLOCK=b,
                num_warps=_FAST_WARPS,
            )
        return out
    full = numel // block
    if full:
        sigmoid_backward_fast_kernel[(full,)](
            grad_output,
            output,
            out,
            BLOCK=block,
            num_warps=_FAST_WARPS,
            buffer_size_limit=_FAST_BSL,
        )
    tail = numel - full * block
    if tail:
        # residual elements; one masked CTA anchored with a global start
        # offset.  Note: pass the base tensors (no sliced views) -- offset
        # views of an output tensor are not reliably handled by the XPU
        # kernel launcher and raise an illegal-memory-access error.
        start = full * block
        sigmoid_backward_masked_kernel[(1,)](
            grad_output,
            output,
            out,
            start,
            numel,
            BLOCK=block,
            num_warps=_FAST_WARPS,
            buffer_size_limit=_FAST_BSL,
        )
    return out


def sigmoid(self):
    logger.debug("GEMS_KUNLUNXIN SIGMOID")
    output = sigmoid_forward(self)
    return output


def sigmoid_backward(grad_output, output):
    logger.debug("GEMS_KUNLUNXIN SIGMOID_BACKWARD")
    if (
        output.dtype in (torch.float16, torch.float32, torch.bfloat16)
        and output.is_contiguous()
        and grad_output.is_contiguous()
        and output.dim() > 0
        and output.device.type == "cuda"
    ):
        return _sigmoid_backward_fast(grad_output, output)
    grad_input = sigmoid_backward_kernel(grad_output, output)
    return grad_input


def sigmoid_(A):
    logger.debug("GEMS_KUNLUNXIN SIGMOID_")
    out = sigmoid_forward(A, out0=A)
    return out
