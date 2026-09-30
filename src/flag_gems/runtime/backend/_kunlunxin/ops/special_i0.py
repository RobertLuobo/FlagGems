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

from flag_gems.ops.special_i0 import _special_i0_kernel
from flag_gems.runtime import torch_device_fn

logger = logging.getLogger(__name__)

BLOCK_SIZE = 1024


def _launch_special_i0(out: torch.Tensor, x: torch.Tensor):
    assert x.is_cuda and out.is_cuda, "Input and output must be CUDA tensors"
    assert (
        out.numel() == x.numel()
    ), "Input and output must have the same number of elements"
    assert out.device == x.device, "Input and output must be on the same device"

    x_in = x
    out_in = out

    if not x_in.is_floating_point():
        x_in = x_in.to(torch.get_default_dtype())

    if x_in.dtype != out_in.dtype:
        x_in = x_in.to(out_in.dtype)

    x_contig = x_in.contiguous()
    out_was_noncontig = not out_in.is_contiguous()
    out_contig = out_in.contiguous() if out_was_noncontig else out_in

    n_elements = out_contig.numel()
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)

    with torch_device_fn.device(x.device):
        _special_i0_kernel[grid](
            x_contig, out_contig, n_elements, BLOCK_SIZE=BLOCK_SIZE
        )

    if out_was_noncontig:
        out_in.copy_(out_contig)
    return out_in


def special_i0(x: torch.Tensor):
    logger.debug("GEMS_KUNLUNXIN SPECIAL_I0")
    if not x.is_cuda:
        raise ValueError("special_i0: input tensor must be on CUDA device")
    out_dtype = x.dtype if x.is_floating_point() else torch.get_default_dtype()
    out = torch.empty_like(x.to(dtype=out_dtype), dtype=out_dtype, device=x.device)
    _launch_special_i0(out, x)
    return out


def special_i0_out(x: torch.Tensor, out: torch.Tensor):
    logger.debug("GEMS_KUNLUNXIN SPECIAL_I0_OUT")
    if not (x.is_cuda and out.is_cuda):
        raise ValueError(
            "special_i0_out: input and output tensors must be on CUDA device"
        )
    if not out.is_floating_point():
        raise TypeError("special_i0_out: output tensor must be a floating point type")
    if x.numel() != out.numel():
        raise ValueError(
            "special_i0_out: input and output must have the same number of elements"
        )
    _launch_special_i0(out, x)
    return out
