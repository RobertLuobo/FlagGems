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

from flag_gems.ops.cummin import scan_then_fan, scan_then_fan_col
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils.limits import get_dtype_max

logger = logging.getLogger(__name__)

ROW_BLOCK = 128


@libentry()
@triton.jit(do_not_specialize=["n_elements"])
def _copy_to_out_kernel(src, dst, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < n_elements
    vals = tl.load(src + offset, mask=mask)
    tl.store(dst + offset, vals, mask=mask)


def _write_back(src, dst):
    if dst.is_contiguous():
        n_elements = src.numel()
        if n_elements == 0:
            return
        BLOCK_SIZE = 1024
        grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
        _copy_to_out_kernel[grid](
            src.reshape(-1), dst.reshape(-1), n_elements, BLOCK_SIZE
        )
    else:
        dst.copy_(src)


@libentry()
@triton.jit(do_not_specialize=["B", "n_rows"])
def rowvec_cummin_kernel(
    inp,
    out,
    out_indices,
    B,
    C,
    n_rows,
    IS_FLOAT: tl.constexpr,
    IS_LOW_PREC: tl.constexpr,
    ROW_BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    r = pid * ROW_BLOCK + tl.arange(0, ROW_BLOCK)
    rmask = r < n_rows
    a = r // C
    c = r % C
    base = a * B * C + c
    max_value = get_dtype_max(inp.type.element_ty)
    if IS_LOW_PREC:
        run_val = tl.full((ROW_BLOCK,), max_value, dtype=tl.float32)
    else:
        run_val = tl.full((ROW_BLOCK,), max_value, dtype=inp.type.element_ty)
    run_idx = tl.zeros((ROW_BLOCK,), dtype=tl.int64)
    for i in tl.range(B, num_stages=1):
        off = base + i * C
        v = tl.load(inp + off, mask=rmask, other=max_value)
        if IS_LOW_PREC:
            v = v.to(tl.float32)
        if IS_FLOAT:
            take = (v != v) | ((run_val == run_val) & (v <= run_val))
        else:
            take = v <= run_val
        run_val = tl.where(take, v, run_val)
        run_idx = tl.where(take, i.to(tl.int64), run_idx)
        tl.store(out + off, run_val.to(out.type.element_ty), mask=rmask)
        tl.store(out_indices + off, run_idx, mask=rmask)


def scan_then_fan_loop_rowvec(inp, out, out_indices, A, B, C, dtype):
    n_rows = A * C
    is_float = inp.dtype in (torch.float16, torch.float32, torch.bfloat16)
    is_low_prec = inp.dtype in (torch.float16, torch.bfloat16)
    grid = (triton.cdiv(n_rows, ROW_BLOCK),)
    with torch_device_fn.device(inp.device):
        rowvec_cummin_kernel[grid](
            inp, out, out_indices, B, C, n_rows, is_float, is_low_prec, ROW_BLOCK
        )


def _cummin_helper(
    input: torch.Tensor,
    values: torch.Tensor,
    indices: torch.Tensor,
    dim: int,
) -> None:
    logger.debug("GEMS_KUNLUNXIN CUMMIN_HELPER")
    assert dim >= -input.ndim and dim < input.ndim, "Invalid dim"
    shape = input.shape
    dim = dim % input.ndim
    M = 1
    N = shape[dim]
    for i in range(dim):
        M *= shape[i]
    input = input.contiguous()
    K = input.numel() // M // N

    compute_dtype = values.dtype
    if input.dtype == torch.float16 or input.dtype == torch.bfloat16:
        compute_dtype = torch.float32

    if (
        values.is_contiguous()
        and indices.is_contiguous()
        and values.dtype == input.dtype
    ):
        out = values
        out_indices = indices
        if M == 1 and K == 1:
            scan_then_fan_col(input, out, out_indices, N, compute_dtype)
        elif M * K <= 16:
            scan_then_fan(input, out, out_indices, M, N, K, compute_dtype)
        else:
            scan_then_fan_loop_rowvec(input, out, out_indices, M, N, K, compute_dtype)
        return None

    out = torch.empty_like(input, dtype=values.dtype)
    out_indices = torch.empty_like(input, dtype=torch.int64)

    if M == 1 and K == 1:
        scan_then_fan_col(input, out, out_indices, N, compute_dtype)
    elif M * K <= 16:
        scan_then_fan(input, out, out_indices, M, N, K, compute_dtype)
    else:
        scan_then_fan_loop_rowvec(input, out, out_indices, M, N, K, compute_dtype)

    _write_back(out, values)
    _write_back(out_indices, indices)

    return None
