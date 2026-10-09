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

import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime.backend._kunlunxin.ops._scaled_dot_product_efficient_attention import (
    _scaled_dot_product_efficient_attention,
)

logger = logging.getLogger(__name__)


@triton.jit
def _sdpa_init_metadata_kernel(
    cum_seq_q,
    cum_seq_k,
    philox_seed,
    philox_offset,
    seq_q,
    seq_k,
    BATCH: tl.constexpr,
):
    pid = tl.program_id(0)
    if pid < BATCH:
        tl.store(cum_seq_q + pid * 2, 0)
        tl.store(cum_seq_q + pid * 2 + 1, seq_q)
        tl.store(cum_seq_k + pid * 2, 0)
        tl.store(cum_seq_k + pid * 2 + 1, seq_k)
    if pid == 0:
        tl.store(philox_seed, 0)
        tl.store(philox_offset, 0)


def _expand_kv_heads(tensor, num_query_heads, num_kv_heads):
    if num_query_heads == num_kv_heads:
        return tensor
    batch, _, seq, head_dim = tensor.shape
    group = num_query_heads // num_kv_heads
    expanded = tensor.reshape(batch, num_kv_heads, 1, seq, head_dim).expand(
        batch, num_kv_heads, group, seq, head_dim
    )
    return expanded.reshape(batch, num_query_heads, seq, head_dim).contiguous()


def _normalize_bias(attn_bias, batch, num_query_heads, seq_q, seq_k, dtype, device):
    if attn_bias is None:
        return None
    if attn_bias.dtype == torch.bool:
        mask = attn_bias.expand(batch, num_query_heads, seq_q, seq_k)
        additive = torch.zeros(
            (batch, num_query_heads, seq_q, seq_k), dtype=torch.float32, device=device
        )
        additive = additive.masked_fill(~mask, float("-inf"))
        return additive
    return attn_bias.to(torch.float32)


def _scaled_dot_product_fused_attention_overrideable(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_bias: torch.Tensor = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    return_debug_mask: bool = False,
    scale: float = None,
):
    logger.debug("GEMS_KUNLUNXIN SCALED_DOT_PRODUCT_FUSED_ATTENTION_OVERRIDEABLE")
    assert dropout_p == 0.0, "Only dropout_p=0.0 is supported"
    assert (
        query.ndim == key.ndim == value.ndim == 4
    ), "Only dense 4D attention is supported"
    assert (
        query.shape[0] == key.shape[0] == value.shape[0]
    ), "Batch dimensions must match"
    assert key.shape == value.shape, "key and value must have matching shapes"
    assert (
        query.shape[1] % key.shape[1] == 0
    ), "Query heads must be a multiple of key heads"

    batch_size, num_query_heads, seq_q, head_dim = query.shape
    _, num_key_heads, seq_k, _ = key.shape

    key_expanded = _expand_kv_heads(key, num_query_heads, num_key_heads)
    value_expanded = _expand_kv_heads(value, num_query_heads, num_key_heads)
    bias = _normalize_bias(
        attn_bias, batch_size, num_query_heads, seq_q, seq_k, query.dtype, query.device
    )

    output, logsumexp, _, _ = _scaled_dot_product_efficient_attention(
        query,
        key_expanded,
        value_expanded,
        attn_bias=bias,
        compute_log_sumexp=True,
        dropout_p=0.0,
        is_causal=is_causal,
        scale=scale,
    )

    cum_seq_q = torch.empty((batch_size, 2), dtype=torch.int32, device=query.device)
    cum_seq_k = torch.empty((batch_size, 2), dtype=torch.int32, device=query.device)
    philox_seed = torch.empty(1, dtype=torch.int64, device=query.device)
    philox_offset = torch.empty(1, dtype=torch.int64, device=query.device)
    debug_attn_mask = (
        torch.empty(
            (batch_size, num_query_heads, seq_q, seq_k),
            dtype=query.dtype,
            device=query.device,
        )
        if return_debug_mask
        else torch.empty(0, dtype=query.dtype, device=query.device)
    )

    _sdpa_init_metadata_kernel[(max(batch_size, 1),)](
        cum_seq_q,
        cum_seq_k,
        philox_seed,
        philox_offset,
        seq_q,
        seq_k,
        BATCH=batch_size,
    )

    return (
        output,
        logsumexp,
        cum_seq_q,
        cum_seq_k,
        seq_q,
        seq_k,
        philox_seed,
        philox_offset,
        debug_attn_mask,
    )
