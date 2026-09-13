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

from flag_gems.utils import libentry

from ..utils.pointwise_dynamic import pointwise_dynamic

logger = logging.getLogger(__name__)

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


@pointwise_dynamic(promotion_methods=[(0, "DEFAULT")], config=config_)
@triton.jit
def silu_forward(x):
    x_fp32 = x.to(tl.float32)
    y = tl.fdiv(x_fp32, (1.0 + tl.exp(-x_fp32)))
    return y


# silu_backward uses a dedicated bounded-tile kernel on XPU. The previous
# pointwise_dynamic implementation (tile = next_pow2(numel/12), up to 2M-wide
# per CTA) combined with `div_rn` (IEEE round-to-nearest division, ~2.9x slower
# than plain `/` on XPU) and `isCloseVectorization/unroll_num` left large shapes
# at ~0.51 gems speedup. The first custom kernel pinned BLOCK=min(next_pow2(n),
# 65536), which left the mid-size band (16K..4M elements) at 0.3..0.8 speedup:
# 65536 lanes per CTA serializes and masks everything.
#
# Block-size policy (probed on XPU card 6, 2026-09-10, do_bench event timing on
# the exact benchmark shapes; /tmp/silu_bw_probe). The optimum keeps a small
# fixed CTA count: ~8 CTAs for n <= 131072, ~32 CTAs for n <= 2M, ~128 CTAs
# above, with BLOCK capped at 65536 (no unmasked variant, `other=` never used):
#   n=16384  (1024,16):   17.2us ->  5.9us   (fp16, was m65536=8.09/17.5us)
#   n=65536  (64,64,16):  17.1us ->  7.3us
#   n=262144 (1024,256):  17.2us -> 11.0us
#   n=1048576(64,64,256): 29.9us -> 23.8us
#   n=4194304(1024,4096): 74.9us -> 71.5us
# `unroll_num`/`buffer_size_limit` sweeps were flat (<0.3%), so the existing
# launch knobs (num_warps=16, buffer_size_limit=4096) are kept. Masked-vs-
# unmasked differs by <1%, but the unmasked variant is used when the tensor is
# exactly divisible by BLOCK (no mask registers, matches log_sigmoid_backward).
_SILU_BW_MAX_BLOCK = 65536


@libentry()
@triton.jit(do_not_specialize=["n_elements"])
def silu_backward_kernel_xpu(
    x_ptr, dy_ptr, out_ptr, n_elements, BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    tid = pid * BLOCK + tl.arange(0, BLOCK)
    mask = tid < n_elements
    x = tl.load(x_ptr + tid, mask=mask).to(tl.float32)
    dy = tl.load(dy_ptr + tid, mask=mask).to(tl.float32)
    sigma = 1.0 / (1.0 + tl.exp(-x))
    dx = dy * sigma * (1.0 + x * (1.0 - sigma))
    tl.store(out_ptr + tid, dx.to(x_ptr.type.element_ty), mask=mask)


@libentry()
@triton.jit
def silu_backward_kernel_xpu_unmasked(
    x_ptr, dy_ptr, out_ptr, BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    tid = pid * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(x_ptr + tid).to(tl.float32)
    dy = tl.load(dy_ptr + tid).to(tl.float32)
    sigma = 1.0 / (1.0 + tl.exp(-x))
    dx = dy * sigma * (1.0 + x * (1.0 - sigma))
    tl.store(out_ptr + tid, dx.to(x_ptr.type.element_ty))


def _silu_backward_pick_block(n_elements):
    # 1 CTA (BLOCK = next_pow2(n)) for n <= 4096: probe on XPU card 6 shows a
    # single wide CTA is ~2x faster than the 8-CTA tier below 4K elements
    # (n=1024: 5.6us vs 14.4us; n=2048: 5.8us vs 10.7us) and still faster at
    # n=4096 (fp16 6.0us vs 6.6us). Tiered CTA count only pays off at n >= 8192.
    if n_elements <= 4096:
        return min(triton.next_power_of_2(n_elements), _SILU_BW_MAX_BLOCK)
    # ~8 CTAs for small/mid, ~32 for large, ~128+ for huge (capped at 65536).
    if n_elements <= 131072:
        ctas = 8
    elif n_elements <= 2097152:
        ctas = 32
    else:
        ctas = 128
    block = (n_elements + ctas - 1) // ctas
    return min(triton.next_power_of_2(block), _SILU_BW_MAX_BLOCK)


# Note: the earlier "tiny fast path" (flat kernel with BLOCK=2048, num_warps=4)
# was removed: A/B on XPU card 6 (2026-09-10) shows it is 0.7..1.4us slower
# than the plain 1-CTA masked main kernel at every n <= 2048 and every dtype
# (e.g. n=1024 fp16 6.50us vs 5.37us; n=2048 bf16 6.12us vs 5.72us), so the
# 1-CTA tier above subsumes it with identical math (fp32 staging, plain `/`,
# downcast at store).


def silu(self):
    logger.debug("GEMS_KUNLUNXIN SILU")
    output = silu_forward(self)
    return output


def silu_backward(grad_output, self):
    logger.debug("GEMS_KUNLUNXIN SILU_BACKWARD")
    x = self if self.is_contiguous() else self.contiguous()
    dy = grad_output if grad_output.is_contiguous() else grad_output.contiguous()
    n_elements = x.numel()
    if n_elements == 0:
        return torch.empty_like(x)
    grad_input = torch.empty_like(x)
    block = _silu_backward_pick_block(n_elements)
    if n_elements % block == 0:
        grid = (n_elements // block, 1, 1)
        silu_backward_kernel_xpu_unmasked[grid](
            x,
            dy,
            grad_input,
            BLOCK=block,
            num_warps=16,
            buffer_size_limit=4096,
        )
    else:
        grid = (triton.cdiv(n_elements, block), 1, 1)
        silu_backward_kernel_xpu[grid](
            x,
            dy,
            grad_input,
            n_elements,
            BLOCK=block,
            num_warps=16,
            buffer_size_limit=4096,
        )
    if grad_input.shape != self.shape or grad_input.stride() != self.stride():
        grad_input = grad_input.reshape(self.shape).as_strided(
            self.size(), self.stride()
        )
    return grad_input


def silu_(A):
    logger.debug("GEMS_KUNLUNXIN SILU_")
    out = silu_forward(A, out0=A)
    return out
