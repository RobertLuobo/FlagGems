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

from flag_gems.ops.take_along_dim import _prepare, take_along_dim_kernel
from flag_gems.runtime import torch_device_fn

logger = logging.getLogger("flag_gems.ops.take_along_dim")

BLOCK_SIZE = 1024


def _launch(inp2d, idx2d, out2d):
    R, D_in = inp2d.shape
    _, D_out = idx2d.shape
    N = out2d.numel()
    if N == 0:
        return
    grid = (triton.cdiv(N, BLOCK_SIZE),)
    with torch_device_fn.device(inp2d.device):
        take_along_dim_kernel[grid](
            inp2d, idx2d, out2d, R, D_in, D_out, N, BLOCK_SIZE=BLOCK_SIZE
        )


def take_along_dim(input, indices, dim=None):
    logger.debug("GEMS_KUNLUNXIN TAKE_ALONG_DIM")
    if input.device != indices.device:
        raise RuntimeError("input and indices must be on the same device")
    inp2d, idx2d, out_shape, restore = _prepare(input, indices, dim)
    out2d = torch.empty(idx2d.shape, device=input.device, dtype=input.dtype)
    _launch(inp2d, idx2d, out2d)
    return restore(out2d)


def take_along_dim_out(input, indices, dim=None, *, out):
    logger.debug("GEMS_KUNLUNXIN TAKE_ALONG_DIM_OUT")
    if not (input.device == indices.device == out.device):
        raise RuntimeError("input, indices and out must be on the same device")
    if out.dtype != input.dtype:
        raise RuntimeError(f"out must have dtype {input.dtype}, but got {out.dtype}")
    inp2d, idx2d, out_shape, restore = _prepare(input, indices, dim)
    out2d = torch.empty(idx2d.shape, device=input.device, dtype=input.dtype)
    _launch(inp2d, idx2d, out2d)
    result = restore(out2d)
    if tuple(out.shape) != tuple(result.shape):
        out.resize_(result.shape)
    out.copy_(result)
    return out
