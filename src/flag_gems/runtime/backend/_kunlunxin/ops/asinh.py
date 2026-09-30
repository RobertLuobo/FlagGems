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

from flag_gems.runtime import torch_device_fn

logger = logging.getLogger(__name__)

BLOCK_SIZE = 1024

_INT_DTYPES = (
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
    torch.uint8,
    torch.bool,
)


@triton.jit
def asinh_kernel(x_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask, other=0)
    x_fp32 = x.to(tl.float32)

    abs_x = tl.abs(x_fp32)
    y = tl.log(abs_x + tl.sqrt(abs_x * abs_x + 1.0))
    result = tl.where(x_fp32 < 0.0, -y, y)

    tl.store(out_ptr + offsets, result, mask=mask)


def _out_dtype(x: torch.Tensor):
    if x.dtype in _INT_DTYPES:
        return torch.float32
    return x.dtype


def _launch_asinh(x: torch.Tensor, out: torch.Tensor):
    assert x.is_cuda and out.is_cuda, "Input and output must be CUDA tensors"

    x_contig = x.contiguous()
    out_contig = out if out.is_contiguous() else torch.empty_like(out)

    n_elements = x_contig.numel()
    if n_elements == 0:
        if out_contig is not out:
            out.copy_(out_contig)
        return out

    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    with torch_device_fn.device(x.device):
        asinh_kernel[grid](x_contig, out_contig, n_elements, BLOCK_SIZE=BLOCK_SIZE)

    if out_contig is not out:
        out.copy_(out_contig)
    return out


def asinh(A):
    logger.debug("GEMS_KUNLUNXIN ASINH")
    out = torch.empty(A.shape, dtype=_out_dtype(A), device=A.device)
    _launch_asinh(A, out)
    return out


def asinh_out(A, out):
    logger.debug("GEMS_KUNLUNXIN ASINH_OUT")
    _launch_asinh(A, out)
    return out
