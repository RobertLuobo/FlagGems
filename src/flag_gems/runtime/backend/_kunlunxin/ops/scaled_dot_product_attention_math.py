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
import math

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn

from .bmm import bmm

logger = logging.getLogger(__name__)


@triton.jit
def _sdpa_math_softmax_kernel(
    Scores,
    Mask,
    Weights,
    sm_scale,
    Sq,
    Skv,
    stride_sz,
    stride_sm,
    stride_sn,
    stride_mz,
    stride_mm,
    stride_mn,
    stride_wz,
    stride_wm,
    stride_wn,
    BLOCK_N: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    HAS_MASK: tl.constexpr,
):
    pid = tl.program_id(0)
    row = pid % Sq
    z = pid // Sq

    offs_n = tl.arange(0, BLOCK_N)
    n_mask = offs_n < Skv

    s_ptrs = Scores + z * stride_sz + row * stride_sm + offs_n * stride_sn
    s = tl.load(s_ptrs, mask=n_mask, other=0.0).to(tl.float32) * sm_scale

    if HAS_MASK:
        m_ptrs = Mask + z * stride_mz + row * stride_mm + offs_n * stride_mn
        s += tl.load(m_ptrs, mask=n_mask, other=0.0).to(tl.float32)

    keep = n_mask
    if IS_CAUSAL:
        keep = keep & (offs_n <= row)

    s = tl.where(keep, s, -float("inf"))
    m = tl.max(s, 0)
    m_safe = tl.where(m == -float("inf"), 0.0, m)
    p = tl.exp(s - m_safe)
    p = tl.where(keep, p, 0.0)
    l = tl.sum(p, 0)
    l_safe = tl.where(l == 0.0, 1.0, l)
    w = p / l_safe

    w_ptrs = Weights + z * stride_wz + row * stride_wm + offs_n * stride_wn
    tl.store(w_ptrs, w.to(Weights.dtype.element_ty), mask=n_mask)


def _scaled_dot_product_attention_math(
    query,
    key,
    value,
    attn_mask=None,
    dropout_p=0.0,
    is_causal=False,
    scale=None,
):
    """XPU overlay for the SDPA math reference.

    Composes QK^T and P@V from the vendor ``bmm`` kernel and performs the
    scaled, masked softmax in a single XPU3-safe Triton kernel (``tl.exp``
    instead of the miscompiled ``tl.exp2``; no ``tl.dot``/``tl.trans``).
    Returns both the attention output and the full attention weight matrix.
    """
    logger.debug("GEMS_KUNLUNXIN _SCALED_DOT_PRODUCT_ATTENTION_MATH")

    assert dropout_p == 0.0, "dropout_p != 0.0 is not supported"
    assert query.dim() == 4, "expected 4D (batch, heads, seq_len, head_dim) tensors"

    batch, heads, q_seq_len, head_dim = query.shape
    kv_seq_len = key.shape[2]

    if scale is None:
        scale = 1.0 / math.sqrt(head_dim)

    bh = batch * heads
    q3 = query.reshape(bh, q_seq_len, head_dim).contiguous()
    k3 = key.reshape(bh, kv_seq_len, head_dim).contiguous()
    v3 = value.reshape(bh, kv_seq_len, head_dim).contiguous()

    # QK^T via vendor bmm: (bh, q, d) x (bh, d, kv) -> (bh, q, kv).
    # Accumulate in fp32 (matching the generic kernel's fp32 qk path) so the
    # scaled softmax stays accurate for large custom scales in low precision.
    kt = k3.transpose(1, 2).contiguous()
    scores = bmm(q3.float(), kt.float())

    weights3 = torch.empty(
        (bh, q_seq_len, kv_seq_len), dtype=query.dtype, device=query.device
    )

    has_attn_mask = attn_mask is not None
    if has_attn_mask:
        assert attn_mask.dtype != torch.bool, "boolean attn_mask is not supported"
        am = attn_mask.expand(batch, heads, q_seq_len, kv_seq_len).contiguous()
        am3 = am.reshape(bh, q_seq_len, kv_seq_len)
        m_strides = (am3.stride(0), am3.stride(1), am3.stride(2))
    else:
        am3 = scores  # unused placeholder; HAS_MASK guards all accesses
        m_strides = (0, 0, 0)

    BLOCK_N = triton.next_power_of_2(kv_seq_len)
    grid = (bh * q_seq_len,)

    with torch_device_fn.device(query.device):
        _sdpa_math_softmax_kernel[grid](
            scores,
            am3,
            weights3,
            scale,
            q_seq_len,
            kv_seq_len,
            scores.stride(0),
            scores.stride(1),
            scores.stride(2),
            m_strides[0],
            m_strides[1],
            m_strides[2],
            weights3.stride(0),
            weights3.stride(1),
            weights3.stride(2),
            BLOCK_N=BLOCK_N,
            IS_CAUSAL=is_causal,
            HAS_MASK=has_attn_mask,
        )

    # P@V via vendor bmm: (bh, q, kv) x (bh, kv, d) -> (bh, q, d)
    out3 = bmm(weights3, v3)

    out = out3.reshape(batch, heads, q_seq_len, head_dim)
    weights = weights3.reshape(batch, heads, q_seq_len, kv_seq_len)
    return out, weights
