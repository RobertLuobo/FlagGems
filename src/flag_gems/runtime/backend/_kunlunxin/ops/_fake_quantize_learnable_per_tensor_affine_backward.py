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
import triton.language.extra.libdevice as libdevice

logger = logging.getLogger(__name__)


@triton.jit
def _fq_lpt_backward_kernel(
    grad_ptr,
    self_ptr,
    scale_ptr,
    zero_point_ptr,
    grad_self_ptr,
    partial_scale_ptr,
    partial_zp_ptr,
    n_elements,
    quant_min,
    quant_max,
    grad_factor,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    grad = tl.load(grad_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    x = tl.load(self_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    scale = tl.load(scale_ptr).to(tl.float32)
    zero_point = tl.load(zero_point_ptr).to(tl.float32)

    qmnf = quant_min.to(tl.float32)
    qmxf = quant_max.to(tl.float32)

    # Use the correctly-rounded fp32 reciprocal (1.0f/scale) then a plain
    # multiply instead of a true fp32 division: a true div flips
    # round-half-to-even at exact half-way quotients and diverges from the
    # native output by one quant level on XPU3.
    inv_s = libdevice.div_rn(1.0, scale)
    xi = x * inv_s
    q = libdevice.rint(xi + zero_point)
    in_range = (q >= qmnf) & (q <= qmxf)
    qc = tl.minimum(tl.maximum(q, qmnf), qmxf)
    in_range_f = in_range.to(tl.float32)

    grad_self = grad * in_range_f
    tl.store(grad_self_ptr + offsets, grad_self, mask=mask)

    grad_scale_contrib = grad * ((qc - zero_point) - xi * in_range_f) * grad_factor
    grad_zero_point_contrib = grad * scale * (in_range_f - 1.0) * grad_factor

    tl.store(partial_scale_ptr + pid, tl.sum(grad_scale_contrib, axis=0))
    tl.store(partial_zp_ptr + pid, tl.sum(grad_zero_point_contrib, axis=0))


@triton.jit
def _reduce_partials_kernel(partial_ptr, out_ptr, n_partials, BLOCK: tl.constexpr):
    acc = 0.0
    for start in range(0, n_partials, BLOCK):
        offsets = start + tl.arange(0, BLOCK)
        v = tl.load(partial_ptr + offsets)
        acc += tl.sum(v, axis=0)
    tl.store(out_ptr, acc)


def _fake_quantize_learnable_per_tensor_affine_backward(
    grad, self, scale, zero_point, quant_min, quant_max, grad_factor=1.0
):
    logger.debug(
        "GEMS_KUNLUNXIN _FAKE_QUANTIZE_LEARNABLE_PER_TENSOR_AFFINE_BACKWARD"
    )

    # The ATen reference always returns float32 gradients (it rejects float16
    # grads and promotes bfloat16 inputs to float32), so match that contract.
    grad_self = torch.empty(self.shape, dtype=torch.float32, device=self.device)
    grad_scale = torch.zeros(scale.shape, dtype=torch.float32, device=scale.device)
    grad_zero_point = torch.zeros(
        zero_point.shape, dtype=torch.float32, device=zero_point.device
    )

    n_elements = self.numel()
    if n_elements == 0:
        return grad_self, grad_scale, grad_zero_point

    BLOCK_SIZE = 1024
    n_blocks = triton.cdiv(n_elements, BLOCK_SIZE)
    n_padded = n_blocks * BLOCK_SIZE

    # Pad the flattened inputs to a whole multiple of BLOCK_SIZE with real
    # zeros so every program sees a FULL block. A partial tail block makes
    # tl.sum mis-reduce even the valid lanes on XPU3; physically-zeroed pad
    # lanes contribute 0 and keep every block full.
    grad_pad = torch.zeros(n_padded, dtype=grad.dtype, device=grad.device)
    grad_pad[:n_elements] = grad.reshape(-1)
    self_pad = torch.zeros(n_padded, dtype=self.dtype, device=self.device)
    self_pad[:n_elements] = self.reshape(-1)
    grad_self_pad = torch.empty(n_padded, dtype=torch.float32, device=self.device)

    partial_scale = torch.zeros(n_blocks, dtype=torch.float32, device=self.device)
    partial_zp = torch.zeros(n_blocks, dtype=torch.float32, device=self.device)

    _fq_lpt_backward_kernel[(n_blocks,)](
        grad_pad,
        self_pad,
        scale.contiguous(),
        zero_point.contiguous(),
        grad_self_pad,
        partial_scale,
        partial_zp,
        n_padded,
        int(quant_min),
        int(quant_max),
        float(grad_factor),
        BLOCK_SIZE=BLOCK_SIZE,
    )
    grad_self = grad_self_pad[:n_elements].reshape(self.shape)

    # Reduce per-block partials in a second triton pass (avoids the cross-block
    # tl.atomic_add that fails the XPU3 ConvertTritonXPUToLLVM pass). Pad the
    # partial buffers so the reduction also runs on whole blocks only.
    RBLOCK = 1024
    n_part_padded = triton.cdiv(n_blocks, RBLOCK) * RBLOCK
    if n_part_padded != n_blocks:
        ps = torch.zeros(n_part_padded, dtype=torch.float32, device=self.device)
        pz = torch.zeros(n_part_padded, dtype=torch.float32, device=self.device)
        ps[:n_blocks] = partial_scale
        pz[:n_blocks] = partial_zp
        partial_scale, partial_zp = ps, pz

    _reduce_partials_kernel[(1,)](partial_scale, grad_scale, n_part_padded, BLOCK=RBLOCK)
    _reduce_partials_kernel[(1,)](partial_zp, grad_zero_point, n_part_padded, BLOCK=RBLOCK)

    return grad_self, grad_scale, grad_zero_point
