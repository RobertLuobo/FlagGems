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

from flag_gems.ops.arctanh import arctanh_kernel
from flag_gems.ops.arctanh_ import arctanh_kernel_
from flag_gems.runtime import torch_device_fn

logger = logging.getLogger(__name__)

BLOCK_SIZE = 1024


def _launch_arctanh(x: torch.Tensor, out: torch.Tensor):
    assert x.is_cuda and out.is_cuda, "Input and output must be CUDA tensors"
    assert x.shape == out.shape, "Input and output shapes must match"
    assert out.dtype == x.dtype, "Output dtype must match input dtype"
    assert x.dtype in (
        torch.float16,
        torch.bfloat16,
        torch.float32,
    ), "Supported dtypes: float16, bfloat16, float32"

    x_contig = x.contiguous()
    out_contig = out if out.is_contiguous() else torch.empty_like(out)

    n_elements = x_contig.numel()
    if n_elements == 0:
        if out_contig is not out:
            out.copy_(out_contig)
        return out

    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    with torch_device_fn.device(x.device):
        arctanh_kernel[grid](x_contig, out_contig, n_elements, BLOCK_SIZE=BLOCK_SIZE)

    if out_contig is not out:
        out.copy_(out_contig)
    return out


def arctanh(x: torch.Tensor):
    logger.debug("GEMS_KUNLUNXIN ARCTANH")
    out = torch.empty_like(x)
    _launch_arctanh(x, out)
    return out


def arctanh_out(x: torch.Tensor, out: torch.Tensor):
    logger.debug("GEMS_KUNLUNXIN ARCTANH_OUT")
    _launch_arctanh(x, out)
    return out


def arctanh_(*args, **kwargs):
    logger.debug("GEMS_KUNLUNXIN ARCTANH_")
    x = None
    if len(args) >= 1 and isinstance(args[0], torch.Tensor):
        x = args[0]
    else:
        x = kwargs.get("input", kwargs.get("self", None))
    if not isinstance(x, torch.Tensor):
        raise TypeError("arctanh_ expects a single Tensor argument")

    if not x.is_contiguous():
        raise ValueError("Input tensor must be contiguous")
    if not x.is_floating_point():
        raise TypeError("arctanh_ only supports floating point tensors")

    n_elements = x.numel()
    if n_elements == 0:
        return x

    use_fp32 = x.dtype in (torch.float16, torch.bfloat16)

    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    with torch_device_fn.device(x.device):
        arctanh_kernel_[grid](
            x, n_elements, BLOCK_SIZE=BLOCK_SIZE, COMPUTE_IN_FP32=use_fp32
        )
    return x
