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

import flag_gems
from flag_gems.ops.special_i1e import _special_i1e_kernel

logger = logging.getLogger("flag_gems.ops.special_i1e")


def _run_special_i1e_kernel(x: torch.Tensor, out: torch.Tensor):
    if x.device.type != flag_gems.device or out.device.type != flag_gems.device:
        raise ValueError(f"Tensors must be {flag_gems.device} tensors")
    assert x.dtype in (
        torch.float16,
        torch.bfloat16,
        torch.float32,
        torch.float64,
    ), "Unsupported dtype"
    assert out.dtype == x.dtype, "Output dtype must match input dtype"

    x_c = x.contiguous()
    out_c = out.contiguous()

    n_elements = out_c.numel()
    if n_elements == 0:
        return out

    grid = (triton.cdiv(n_elements, 1024),)
    _special_i1e_kernel[grid](x_c, out_c, n_elements, BLOCK_SIZE=1024)

    if out_c.data_ptr() != out.data_ptr():
        out.copy_(out_c)
    return out


def special_i1e(x: torch.Tensor):
    """
    ATen wrapper: special_i1e(Tensor self) -> Tensor
    """ 
    logger.debug("GEMS SPECIAL_I1E")
    logger.debug("GEMS_KUNLUNXIN SPECIAL_I1E")
    out = torch.empty_like(x)
    return _run_special_i1e_kernel(x, out)


def special_i1e_out(self: torch.Tensor, out: torch.Tensor):
    """
    ATen wrapper: special_i1e.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
    """
    logger.debug("GEMS SPECIAL_I1E_OUT")
    logger.debug("GEMS_KUNLUNXIN SPECIAL_I1E_OUT")
    x = self
    return _run_special_i1e_kernel(x, out)
