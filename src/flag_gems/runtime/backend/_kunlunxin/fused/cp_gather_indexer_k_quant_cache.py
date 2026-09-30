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

import torch
import triton
import triton.language as tl


@triton.jit
def _cp_gather_indexer_quant_cache_kernel(
    kv_cache_ptr,
    kv_cache_scale_ptr,
    k_fp8_ptr,
    k_scale_ptr,
    block_table_ptr,
    batch_id_ptr,
    batch_start_ptr,
    block_size,
    block_table_stride,
    kv_cache_stride,
    kv_cache_scale_stride,
    k_fp8_stride,
    num_quant_blocks,
    num_tokens,
    HEAD_DIM: tl.constexpr,
    QUANT_BLOCK_SIZE: tl.constexpr,
    TOKEN_BLOCK: tl.constexpr,
):
    tid = tl.program_id(0) * TOKEN_BLOCK + tl.arange(0, TOKEN_BLOCK)
    quant_block_id = tl.program_id(1)
    in_bounds = tid < num_tokens

    # The per-token owning batch is precomputed on host (searchsorted over
    # cu_seqlen); every in-kernel search variant explodes on XPU3 (static
    # unroll -> buffer-size tune fail at large batch, runtime loop -> uni_sram
    # OOR, 2D axis reduction -> miscompiled values). Only the byte gather runs
    # in the kernel.
    batch_id = tl.load(batch_id_ptr + tid, mask=in_bounds, other=-1)
    batch_start = tl.load(batch_start_ptr + tid, mask=in_bounds, other=0)

    valid_tokens = (batch_id >= 0) & in_bounds
    safe_batch_id = tl.maximum(batch_id, 0)
    batch_offset = tid - batch_start
    block_table_id = batch_offset // block_size
    block_offset = batch_offset % block_size
    block_table_offset = safe_batch_id * block_table_stride + block_table_id
    block_id = tl.load(block_table_ptr + block_table_offset, mask=valid_tokens, other=0)

    offsets = quant_block_id * QUANT_BLOCK_SIZE + tl.arange(0, QUANT_BLOCK_SIZE)
    mask = valid_tokens[:, None]
    src_cache_offset = (
        block_id[:, None].to(tl.int64) * kv_cache_stride
        + block_offset[:, None].to(tl.int64) * HEAD_DIM
    )
    src_scale_offset = (
        block_id * kv_cache_scale_stride
        + block_offset * num_quant_blocks
        + quant_block_id
    )
    dst_offset = tid[:, None].to(tl.int64) * k_fp8_stride

    src_scale_ptr = kv_cache_scale_ptr + src_scale_offset
    src_cache_ptr = kv_cache_ptr + src_cache_offset
    dst_k_ptr = k_fp8_ptr + dst_offset

    # Read-modify-write: masked stores may be widened on XPU3, so preserve the
    # existing bytes of non-gathered (padding) tokens explicitly via tl.where,
    # and gate the physical store only on the in-bounds mask.
    scale_dst = k_scale_ptr + tid * num_quant_blocks + quant_block_id
    cur_scale = tl.load(scale_dst, mask=in_bounds, other=0.0)
    new_scale = tl.load(src_scale_ptr, mask=valid_tokens, other=0.0)
    out_scale = tl.where(valid_tokens, new_scale, cur_scale)
    tl.store(scale_dst, out_scale, mask=in_bounds)

    cur_val = tl.load(dst_k_ptr + offsets[None, :], mask=in_bounds[:, None], other=0)
    new_val = tl.load(src_cache_ptr + offsets[None, :], mask=mask, other=0)
    out_val = tl.where(mask, new_val, cur_val)
    tl.store(dst_k_ptr + offsets[None, :], out_val, mask=in_bounds[:, None])


def cp_gather_indexer_k_quant_cache(
    k_cache: torch.Tensor,
    k_fp8: torch.Tensor,
    k_fp8_scale: torch.Tensor,
    block_table: torch.Tensor,
    cu_seqlen: torch.Tensor,
):
    num_tokens = k_fp8.size(0)
    block_size = k_cache.size(1)
    block_table_stride = block_table.stride(0)
    head_dim = k_fp8.shape[-1]
    num_blocks = k_cache.shape[0]
    quant_block_size = head_dim * 4 // k_fp8_scale.size(1)
    if head_dim % quant_block_size != 0:
        raise ValueError("head_dim must be divisible by quant_block_size")
    num_quant_blocks = head_dim // quant_block_size

    k_cache_flat = k_cache.view(num_blocks, -1)
    k_cache_value = k_cache_flat[:, : block_size * head_dim]
    k_cache_scale = k_cache_flat[:, block_size * head_dim :].view(torch.float32)
    k_fp8 = k_fp8.view(torch.uint8)
    k_fp8_scale = k_fp8_scale.view(torch.float32)
    batch_size = block_table.shape[0]

    # Host-side token -> batch mapping (pure indexing metadata; no numeric work
    # of the gather itself). Tokens beyond cu_seqlen[-1] are padding: batch_id
    # is forced to -1 so the kernel leaves their bytes untouched.
    cu_seqlen = cu_seqlen.to(torch.int32)
    tids = torch.arange(num_tokens, device=k_fp8.device, dtype=torch.int32)
    total_valid = cu_seqlen[batch_size]
    token_batch = torch.searchsorted(cu_seqlen, tids, right=True) - 1
    valid = (tids < total_valid) & (token_batch >= 0) & (token_batch < batch_size)
    safe_batch = token_batch.clamp(0, batch_size - 1)
    batch_id = torch.where(valid, safe_batch, torch.full_like(safe_batch, -1)).to(
        torch.int32
    )
    batch_start = cu_seqlen[safe_batch].to(torch.int32)

    if num_tokens < 32:
        token_block = 1
    elif num_tokens < 64:
        token_block = 2
    elif num_tokens < 128:
        token_block = 4
    elif num_tokens < 256:
        token_block = 8
    elif num_tokens < 512:
        token_block = 16
    else:
        token_block = 32

    grid = (triton.cdiv(num_tokens, token_block), num_quant_blocks)
    _cp_gather_indexer_quant_cache_kernel[grid](
        k_cache_value,
        k_cache_scale,
        k_fp8,
        k_fp8_scale,
        block_table,
        batch_id,
        batch_start,
        block_size,
        block_table_stride,
        k_cache_value.stride(0),
        k_cache_scale.stride(0),
        k_fp8.stride(0),
        num_quant_blocks,
        num_tokens,
        head_dim,
        quant_block_size,
        token_block,
    )


def _install():
    from flag_gems.fused.cp_gather_indexer_k_quant_cache import (
        cp_gather_indexer_k_quant_cache as _generic,
    )

    for mod_name in (
        "flag_gems.fused",
        "flag_gems.fused.cp_gather_indexer_k_quant_cache",
        "flag_gems",
    ):
        mod = sys.modules.get(mod_name)
        if mod is not None:
            cur = getattr(mod, "cp_gather_indexer_k_quant_cache", None)
            if cur is _generic:
                mod.cp_gather_indexer_k_quant_cache = cp_gather_indexer_k_quant_cache


_install()
