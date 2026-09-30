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


@triton.jit
def _rshift_tensor_kernel(v_ptr, s_ptr, out_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    v = tl.load(v_ptr + offs, mask=mask)
    s = tl.load(s_ptr + offs, mask=mask)
    tl.store(out_ptr + offs, v >> s, mask=mask)


@triton.jit
def _rshift_scalar_kernel(v_ptr, scalar, out_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    v = tl.load(v_ptr + offs, mask=mask)
    tl.store(out_ptr + offs, v >> scalar, mask=mask)


def _launch_tensor(value, shift, out):
    result_dtype = torch.result_type(value, shift)
    shape = torch.broadcast_shapes(value.shape, shift.shape)
    v = value.to(result_dtype).broadcast_to(shape).contiguous()
    s = shift.to(result_dtype).broadcast_to(shape).contiguous()

    if out is None:
        result = torch.empty(shape, dtype=result_dtype, device=value.device)
        buf = result
    else:
        result = out
        buf = out if (out.is_contiguous() and out.dtype == result_dtype) else \
            torch.empty(shape, dtype=result_dtype, device=value.device)

    n = v.numel()
    if n == 0:
        if buf is not result:
            result.copy_(buf)
        return result

    grid = (triton.cdiv(n, BLOCK_SIZE),)
    with torch_device_fn.device(value.device):
        _rshift_tensor_kernel[grid](v, s, buf, n, BLOCK=BLOCK_SIZE)

    if buf is not result:
        result.copy_(buf)
    return result


def _launch_scalar(value, scalar, out):
    result_dtype = torch.result_type(value, scalar)
    v = value.to(result_dtype).contiguous()
    shape = value.shape

    if out is None:
        result = torch.empty(shape, dtype=result_dtype, device=value.device)
        buf = result
    else:
        result = out
        buf = out if (out.is_contiguous() and out.dtype == result_dtype) else \
            torch.empty(shape, dtype=result_dtype, device=value.device)

    n = v.numel()
    if n == 0:
        if buf is not result:
            result.copy_(buf)
        return result

    grid = (triton.cdiv(n, BLOCK_SIZE),)
    with torch_device_fn.device(value.device):
        _rshift_scalar_kernel[grid](v, scalar, buf, n, BLOCK=BLOCK_SIZE)

    if buf is not result:
        result.copy_(buf)
    return result


def __rshift__(self: torch.Tensor, other, *, out=None) -> torch.Tensor:
    logger.debug("GEMS_KUNLUNXIN RSHIFT")
    if isinstance(other, torch.Tensor):
        return _launch_tensor(self, other, out)
    return _launch_scalar(self, other, out)


def __rshift___out(self: torch.Tensor, other, *, out) -> torch.Tensor:
    logger.debug("GEMS_KUNLUNXIN RSHIFT_OUT")
    if isinstance(other, torch.Tensor):
        return _launch_tensor(self, other, out)
    return _launch_scalar(self, other, out)
