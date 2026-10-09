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

"""Forward-only dense cross attention for the Kunlunxin XPU3 backend.

Same public contract and numerics as ``flag_gems.ops.cross_attention`` (BNSD
layout, MHA/GQA/MQA, non-zero mask entries blocked, exact zeros for fully
masked rows). The generic implementation gates the whole op behind
``triton.experimental.tle`` (``HAS_TLE``), which is disabled on Kunlunxin, so
it raises ``RuntimeError`` for every call. A standalone flash-attention kernel
cannot be compiled correctly on XPU3 (multi-iteration loops exceed the arch-3
64 pipeline-event budget; the single-tile form miscompiles), so this overlay
delegates to the vendor's existing, XPU3-native two-pass SDPA
(``scaled_dot_product_attention_forward``), which supports GQA/MQA, a smaller
value dim, and masking.

Two conversions bridge the two contracts:
  * mask polarity -- cross_attention blocks on non-zero entries, SDPA's bool
    mask participates on True, so the mask is passed as ``attn_mask == 0``.
  * key padding -- the vendor kernel corrupts roughly half the query rows when
    the key length is not a multiple of its tile width (32); the partial last
    tile is mishandled. K/V are padded up to a multiple of 32 and the padded
    keys are masked out, which keeps the result identical while routing every
    shape through the vendor's clean full-tile path.
"""

import logging
import math

import torch

from flag_gems.ops.cross_attention import _validate_inputs

from .attention import scaled_dot_product_attention_forward

logger = logging.getLogger(__name__)

_TILE = 32


def cross_attention(query, key, value, attn_mask=None, scale=None):
    """Compute forward cross attention for BNSD inputs (Kunlunxin XPU3).

    Q and K have the same head dimension; V may have a smaller dimension. MHA,
    GQA, and MQA are supported. Non-zero bool/uint8 mask entries are blocked.
    Fully masked rows produce exact zeros. This API currently has no backward.
    """
    logger.debug("GEMS_KUNLUNXIN CROSS ATTENTION")
    _validate_inputs(query, key, value, attn_mask, scale)
    batch, query_heads, query_len, qk_dim = query.shape
    kv_heads, key_len = key.shape[1], key.shape[2]
    softmax_scale = 1.0 / math.sqrt(qk_dim) if scale is None else float(scale)

    # cross blocks on non-zero; SDPA participates on True (also contiguifies).
    bool_mask = None if attn_mask is None else attn_mask == 0

    pad = (-key_len) % _TILE
    if pad:
        key = torch.cat(
            [key, key.new_zeros(batch, kv_heads, pad, qk_dim)], dim=2
        )
        value = torch.cat(
            [value, value.new_zeros(batch, kv_heads, pad, value.shape[3])], dim=2
        )
        if bool_mask is None:
            cols = torch.arange(key_len + pad, device=query.device) < key_len
            bool_mask = cols[None, :].expand(query_len, key_len + pad)
        else:
            pad_cols = torch.zeros(
                (*bool_mask.shape[:-1], pad),
                dtype=torch.bool,
                device=bool_mask.device,
            )
            bool_mask = torch.cat([bool_mask, pad_cols], dim=-1)

    return scaled_dot_product_attention_forward(
        query,
        key,
        value,
        attn_mask=bool_mask,
        scale=softmax_scale,
        enable_gqa=query_heads != kv_heads,
    )


__all__ = ["cross_attention"]
