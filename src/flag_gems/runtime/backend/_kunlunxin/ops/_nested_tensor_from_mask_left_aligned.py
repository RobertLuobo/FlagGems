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
from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as ext

logger = logging.getLogger(__name__)


@triton.jit
def reduce_all(a, b):
    return a and b


@libentry()
@triton.jit
def row_left_aligned_kernel(
    mask,
    mid,
    L,
    BLOCK_N: tl.constexpr,
):
    pid = ext.program_id(0)
    row_base = pid * L

    num_true = tl.zeros([], dtype=tl.int32)
    first_false = tl.full([], L, dtype=tl.int32)

    for off in range(0, L, BLOCK_N):
        cols = off + tl.arange(0, BLOCK_N)
        in_range = cols < L
        ptrs = mask + row_base + cols
        vals = tl.load(ptrs, mask=in_range, other=0).to(tl.int32)

        num_true += tl.sum(vals, axis=0)

        # Index of the first False element (L for all-True / out-of-range cols).
        candidate = tl.where((vals == 0) & in_range, cols, L).to(tl.int32)
        first_false = tl.minimum(first_false, tl.min(candidate, axis=0))

    row_aligned = (num_true == first_false).to(tl.int1)
    tl.store(mid + pid, row_aligned)


@libentry()
@triton.jit
def reduce_all_kernel(mid, out, MID_SIZE, BLOCK_MID: tl.constexpr):
    offset = tl.arange(0, BLOCK_MID)
    ptrs = mid + offset
    mask_cond = offset < MID_SIZE
    vals = tl.load(ptrs, mask=mask_cond, other=1).to(tl.int1)
    all_val = tl.reduce(vals, axis=0, combine_fn=reduce_all)
    tl.store(out, all_val)


def _nested_tensor_from_mask_left_aligned(t, mask):
    logger.debug("GEMS_KUNLUNXIN _NESTED_TENSOR_FROM_MASK_LEFT_ALIGNED")

    if mask.dtype != torch.bool:
        raise RuntimeError(
            f"Expected mask to be of ScalarType Bool, but got {mask.dtype} instead."
        )
    if mask.dim() != 2:
        raise RuntimeError("Padding mask should be 2D")
    if t.dim() != 3:
        raise RuntimeError("Input should be a 3D tensor, N * L * D")

    N, L = t.size(0), t.size(1)
    NN, LL = mask.size(0), mask.size(1)
    if N != NN or L != LL:
        raise RuntimeError("Mask size should match input size")

    if N == 0 or L == 0:
        return True

    mask = mask.contiguous()
    BLOCK_N = min(1024, triton.next_power_of_2(L))
    block_mid = triton.next_power_of_2(N)

    mid = torch.empty((N,), dtype=torch.bool, device=mask.device)
    out = torch.empty([], dtype=torch.bool, device=mask.device)

    with torch_device_fn.device(mask.device):
        row_left_aligned_kernel[(N, 1)](mask, mid, L, BLOCK_N=BLOCK_N)
        reduce_all_kernel[(1, 1)](mid, out, N, BLOCK_MID=block_mid)

    return bool(out.item())
