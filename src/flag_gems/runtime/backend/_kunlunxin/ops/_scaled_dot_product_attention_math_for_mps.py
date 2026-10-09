# Copyright 2026, The FlagOS Contributors.
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

from .scaled_dot_product_attention_math import _scaled_dot_product_attention_math

logger = logging.getLogger(__name__)


def _scaled_dot_product_attention_math_for_mps(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: torch.Tensor = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    dropout_mask: torch.Tensor = None,
    *,
    scale: float = None,
):
    """XPU overlay for aten::_scaled_dot_product_attention_math_for_mps.

    The MPS math fallback is numerically identical to the SDPA math reference:
    it returns both the attention output and the full softmax weight matrix.
    Reuse the XPU3-safe math overlay (vendor ``bmm`` for QK^T / P@V plus a
    trailing-axis 1-D softmax kernel) instead of the generic ``tl.dot`` /
    ``tl.trans`` kernel, which fails ``TritonSDNNOptimizeRcLayout`` on XPU3.
    """
    logger.debug("GEMS_KUNLUNXIN _SCALED_DOT_PRODUCT_ATTENTION_MATH_FOR_MPS")

    assert dropout_mask is None, "dropout_mask is not supported"

    return _scaled_dot_product_attention_math(
        query,
        key,
        value,
        attn_mask=attn_mask,
        dropout_p=dropout_p,
        is_causal=is_causal,
        scale=scale,
    )
