# Copyright 2026 FlagOS Contributors.
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

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import tl_extra_shim

logger = logging.getLogger(__name__)

_TILE = 8192


@triton.jit
def _fq_lpt_elementwise_kernel(
    grad_ptr,
    self_ptr,
    scale_ptr,
    zero_point_ptr,
    grad_self_ptr,
    buf_scale_ptr,
    buf_zp_ptr,
    n_elements,
    quant_min,
    quant_max,
    grad_factor,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements

    grad = tl.load(grad_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    x = tl.load(self_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    scale = tl.load(scale_ptr).to(tl.float32)
    zero_point = tl.load(zero_point_ptr).to(tl.float32)

    q = tl_extra_shim.nearbyint(x / scale + zero_point)
    in_range = (q >= quant_min) & (q <= quant_max)
    q = tl.minimum(tl.maximum(q, quant_min), quant_max)
    in_range_f = in_range.to(tl.float32)

    grad_self = grad * in_range_f
    scale_c = grad * ((q - zero_point) - (x / scale) * in_range_f) * grad_factor
    zp_c = grad * scale * (in_range_f - 1.0) * grad_factor

    tl.store(grad_self_ptr + offsets, grad_self, mask=mask)
    tl.store(buf_scale_ptr + offsets, scale_c, mask=mask)
    tl.store(buf_zp_ptr + offsets, zp_c, mask=mask)


@triton.jit
def _fq_lpt_reduce_kernel(
    buf_scale_ptr,
    buf_zp_ptr,
    part_scale_ptr,
    part_zp_ptr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    tl.store(part_scale_ptr + pid, tl.sum(tl.load(buf_scale_ptr + offsets)))
    tl.store(part_zp_ptr + pid, tl.sum(tl.load(buf_zp_ptr + offsets)))


@triton.jit
def _fq_lpt_fold_kernel(
    part_scale_ptr,
    part_zp_ptr,
    out_scale_ptr,
    out_zp_ptr,
    BLOCK: tl.constexpr,
):
    idx = tl.arange(0, BLOCK)
    tl.store(out_scale_ptr, tl.sum(tl.load(part_scale_ptr + idx)))
    tl.store(out_zp_ptr, tl.sum(tl.load(part_zp_ptr + idx)))


def _fake_quantize_learnable_per_tensor_affine_backward(
    grad, self, scale, zero_point, quant_min, quant_max, grad_factor=1.0
):
    logger.debug(
        "GEMS_KUNLUNXIN _FAKE_QUANTIZE_LEARNABLE_PER_TENSOR_AFFINE_BACKWARD"
    )

    if grad.device.type != "cuda" or self.device.type != "cuda":
        raise ValueError("Inputs must be on a CUDA device.")

    if not (
        grad.is_floating_point()
        and self.is_floating_point()
        and scale.is_floating_point()
        and zero_point.is_floating_point()
    ):
        raise ValueError("All inputs must be floating-point tensors.")

    if self.dtype not in (torch.float32, torch.bfloat16):
        raise ValueError(
            f"Unsupported dtype {self.dtype}; expected float32 or bfloat16."
        )

    grad = grad.contiguous()
    self = self.contiguous()
    scale = scale.contiguous()
    zero_point = zero_point.contiguous()

    grad_self = torch.empty(self.shape, dtype=torch.float32, device=self.device)
    grad_scale = torch.zeros(scale.shape, dtype=torch.float32, device=scale.device)
    grad_zero_point = torch.zeros(
        zero_point.shape, dtype=torch.float32, device=zero_point.device
    )

    n_elements = self.numel()
    if n_elements == 0:
        return grad_self, grad_scale, grad_zero_point

    grad_flat = grad.reshape(-1)
    self_flat = self.reshape(-1)
    grad_self_flat = grad_self.reshape(-1)

    n_tiles = triton.cdiv(n_elements, _TILE)
    padded = n_tiles * _TILE
    buf_scale = torch.zeros(padded, dtype=torch.float32, device=self.device)
    buf_zp = torch.zeros(padded, dtype=torch.float32, device=self.device)

    block_part = triton.next_power_of_2(n_tiles)
    part_scale = torch.zeros(block_part, dtype=torch.float32, device=self.device)
    part_zp = torch.zeros(block_part, dtype=torch.float32, device=self.device)

    grad_factor = float(grad_factor)

    with torch_device_fn.device(self.device):
        _fq_lpt_elementwise_kernel[(n_tiles,)](
            grad_flat,
            self_flat,
            scale,
            zero_point,
            grad_self_flat,
            buf_scale,
            buf_zp,
            n_elements,
            quant_min,
            quant_max,
            grad_factor,
            BLOCK=_TILE,
        )
        _fq_lpt_reduce_kernel[(n_tiles,)](
            buf_scale,
            buf_zp,
            part_scale,
            part_zp,
            BLOCK=_TILE,
        )
        _fq_lpt_fold_kernel[(1,)](
            part_scale,
            part_zp,
            grad_scale,
            grad_zero_point,
            BLOCK=block_part,
        )

    return grad_self, grad_scale, grad_zero_point
