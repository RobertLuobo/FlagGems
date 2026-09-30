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

from flag_gems.runtime.backend._kunlunxin.ops._scaled_dot_product_efficient_attention import (
    _scaled_dot_product_efficient_attention,
)

logger = logging.getLogger(__name__)


def _scaled_dot_product_flash_attention(
    query,
    key,
    value,
    dropout_p=0.0,
    is_causal=False,
    return_debug_mask=False,
    *,
    scale=None,
):
    logger.debug("GEMS_KUNLUNXIN _SCALED_DOT_PRODUCT_FLASH_ATTENTION")

    output, logsumexp, philox_seed, philox_offset = (
        _scaled_dot_product_efficient_attention(
            query,
            key,
            value,
            attn_bias=None,
            compute_log_sumexp=True,
            dropout_p=dropout_p,
            is_causal=is_causal,
            scale=scale,
        )
    )

    max_q = query.shape[2]
    max_k = key.shape[2]

    if return_debug_mask:
        batch, num_heads, q_seq_len, _ = query.shape
        kv_seq_len = key.shape[2]
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
        None,
        None,
        max_q,
        max_k,
        philox_seed,
        philox_offset,
        debug_attn_mask,
    )
