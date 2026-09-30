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

import sys
from typing import Optional, Tuple

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn


@triton.jit
def _compute_global_topk_indices_and_lens_kernel(
    global_indices_ptr,
    global_stride,
    lens_ptr,
    local_indices_ptr,
    local_stride,
    topk,
    token_to_req_indices_ptr,
    block_table_ptr,
    block_table_stride,
    block_size,
    is_valid_token_ptr,
    num_tokens,
    BLOCK: tl.constexpr,
):
    token = tl.program_id(0)

    req_idx = tl.load(token_to_req_indices_ptr + token)
    is_valid = tl.load(is_valid_token_ptr + token)

    local_row = local_indices_ptr + token * local_stride
    global_row = global_indices_ptr + token * global_stride
    block_table_row = block_table_ptr + req_idx * block_table_stride

    count = tl.zeros((BLOCK,), dtype=tl.int32)
    for start in range(0, topk, BLOCK):
        offs = start + tl.arange(0, BLOCK)
        topk_mask = offs < topk

        local_idx = tl.load(local_row + offs, mask=topk_mask, other=-1)
        valid = local_idx >= 0

        block_idx = local_idx // block_size
        block_off = local_idx - block_idx * block_size

        block_no = tl.load(
            block_table_row + block_idx,
            mask=topk_mask & valid,
            other=0,
        )
        slot = block_no * block_size + block_off
        slot = tl.where(valid, slot, -1)

        tl.store(global_row + offs, slot, mask=topk_mask)
        count += tl.where(valid, 1, 0).to(tl.int32)

    total = tl.sum(count, axis=0)
    lens = tl.where(is_valid != 0, total, 0)
    tl.store(lens_ptr + token, lens)


def compute_global_topk_indices_and_lens(
    topk_indices: torch.Tensor,
    token_to_req_indices: torch.Tensor,
    block_table: torch.Tensor,
    block_size: int,
    is_valid_token: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    assert topk_indices.ndim == 2
    if is_valid_token is None:
        is_valid_token = torch.ones(
            (topk_indices.shape[0],), device=topk_indices.device, dtype=torch.int32
        )
    num_tokens, topk = topk_indices.shape
    global_indices = torch.empty_like(topk_indices, dtype=torch.int32)
    lens = torch.empty((num_tokens,), device=topk_indices.device, dtype=torch.int32)

    if num_tokens == 0:
        return global_indices, lens

    BLOCK = min(1024, max(1, triton.next_power_of_2(topk)))
    with torch_device_fn.device(topk_indices.device):
        _compute_global_topk_indices_and_lens_kernel[(num_tokens,)](
            global_indices,
            global_indices.stride(0),
            lens,
            topk_indices,
            topk_indices.stride(0),
            topk,
            token_to_req_indices,
            block_table,
            block_table.stride(0),
            block_size,
            is_valid_token,
            num_tokens,
            BLOCK=BLOCK,
        )
    return global_indices, lens


def _install():
    from flag_gems.fused.deepseek_v4_attention_compute_global_topk_indices_and_lens import (
        compute_global_topk_indices_and_lens as _generic_fn,
    )

    fused_pkg = sys.modules.get("flag_gems.fused")
    if fused_pkg is not None:
        cur = getattr(fused_pkg, "compute_global_topk_indices_and_lens", None)
        if cur is _generic_fn:
            fused_pkg.compute_global_topk_indices_and_lens = (
                compute_global_topk_indices_and_lens
            )

    sub = sys.modules.get(
        "flag_gems.fused.deepseek_v4_attention_compute_global_topk_indices_and_lens"
    )
    if sub is not None:
        cur = getattr(sub, "compute_global_topk_indices_and_lens", None)
        if cur is _generic_fn:
            sub.compute_global_topk_indices_and_lens = (
                compute_global_topk_indices_and_lens
            )

    top = sys.modules.get("flag_gems")
    if top is not None:
        cur = getattr(top, "compute_global_topk_indices_and_lens", None)
        if cur is _generic_fn:
            top.compute_global_topk_indices_and_lens = (
                compute_global_topk_indices_and_lens
            )


_install()


__all__ = [
    "compute_global_topk_indices_and_lens",
]
