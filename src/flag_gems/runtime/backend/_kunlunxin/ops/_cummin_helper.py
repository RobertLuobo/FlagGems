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

from flag_gems.ops.cummin import (
    scan_then_fan,
    scan_then_fan_col,
    scan_then_fan_loop,
)

logger = logging.getLogger(__name__)

_BLOCK = 1024


def _cummin_helper(input, values, indices, dim):
    logger.debug("GEMS_KUNLUNXIN CUMMIN_HELPER")
    assert dim >= -input.ndim and dim < input.ndim, "Invalid dim"
    if input.numel() == 0:
        return None

    dim = dim % input.ndim

    # bool is computed in int32 (never int64): the shared associative-scan
    # kernels miscompile with an int64 value payload on XPU3 (wrong carried
    # value/index), while int32 is exact. 0/1 fits int32 and copies back to the
    # bool output buffer losslessly.
    is_bool = input.dtype is torch.bool
    work = input.to(torch.int32) if is_bool else input

    # Move the scanned axis to the trailing (contiguous) position and flatten to
    # (rows, N). This reduces every case to the contiguous C==1 scan, avoiding
    # the strided middle-axis path.
    moved = work.movedim(dim, -1).contiguous()
    perm_shape = moved.shape
    n = perm_shape[-1]
    rows = moved.numel() // n
    flat = moved.reshape(rows, n)

    # The multi-block carry in the loop/abc kernels is only correct on XPU3 when
    # the final block is full (B a multiple of the 1024 block). For a large
    # non-multiple N, pad the trailing axis with the identity value (+inf / int
    # max, which never wins a min and sits after the real data) so the carry is
    # exact, then slice the padding off.
    if n > 4096 and (n % _BLOCK) != 0:
        n_pad = ((n + _BLOCK - 1) // _BLOCK) * _BLOCK
        if flat.is_floating_point():
            pad_val = float("inf")
        else:
            pad_val = torch.iinfo(flat.dtype).max
        padding = torch.full(
            (rows, n_pad - n), pad_val, dtype=flat.dtype, device=flat.device
        )
        scan_in = torch.cat([flat, padding], dim=1)
    else:
        n_pad = n
        scan_in = flat

    out = torch.empty_like(scan_in)
    out_indices = torch.empty_like(scan_in, dtype=torch.int64)

    compute_dtype = out.dtype
    if scan_in.dtype in (torch.float16, torch.bfloat16):
        compute_dtype = torch.float32

    if rows == 1:
        scan_then_fan_col(scan_in, out, out_indices, n_pad, compute_dtype)
    elif rows <= 16:
        scan_then_fan(scan_in, out, out_indices, rows, n_pad, 1, compute_dtype)
    else:
        scan_then_fan_loop(scan_in, out, out_indices, rows, n_pad, 1, compute_dtype)

    out = out[:, :n].reshape(perm_shape).movedim(-1, dim)
    out_indices = out_indices[:, :n].reshape(perm_shape).movedim(-1, dim)

    values.copy_(out)
    indices.copy_(out_indices)
    return None
