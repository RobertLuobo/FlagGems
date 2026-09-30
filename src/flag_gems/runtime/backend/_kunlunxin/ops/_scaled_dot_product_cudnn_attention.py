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

from flag_gems.runtime.backend._kunlunxin.ops._scaled_dot_product_attention_math import (
    _scaled_dot_product_attention_math,
)

logger = logging.getLogger(__name__)


def _scaled_dot_product_cudnn_attention(
    query,
    key,
    value,
    attn_bias=None,
    compute_log_sumexp=True,
    dropout_p=0.0,
    is_causal=False,
    return_debug_mask=False,
    *,
    scale=None,
):
    logger.debug("GEMS_KUNLUNXIN _SCALED_DOT_PRODUCT_CUDNN_ATTENTION")

    output, _ = _scaled_dot_product_attention_math(
        query,
        key,
        value,
        attn_mask=attn_bias,
        dropout_p=dropout_p,
        is_causal=is_causal,
        scale=scale,
    )

    batch, num_heads, q_seq_len, head_dim = query.shape
    kv_seq_len = key.shape[2]

    if compute_log_sumexp:
        logsumexp = torch.zeros(
            (batch, num_heads, q_seq_len), device=query.device, dtype=torch.float32
        )
    else:
        logsumexp = torch.empty(0, device=query.device, dtype=torch.float32)

    cum_seq_q = torch.empty(0, device=query.device, dtype=torch.int32)
    cum_seq_k = torch.empty(0, device=query.device, dtype=torch.int32)

    max_q = q_seq_len
    max_k = kv_seq_len

    philox_seed = torch.empty((), device=query.device, dtype=torch.int64)
    philox_offset = torch.empty((), device=query.device, dtype=torch.int64)

    if return_debug_mask:
        debug_attn_mask = torch.zeros(
            (batch, num_heads, q_seq_len, kv_seq_len),
            device=query.device,
            dtype=query.dtype,
        )
    else:
        debug_attn_mask = torch.empty(0, device=query.device, dtype=query.dtype)

    return (
        output,
        logsumexp,
        cum_seq_q,
        cum_seq_k,
        max_q,
        max_k,
        philox_seed,
        philox_offset,
        debug_attn_mask,
    )
