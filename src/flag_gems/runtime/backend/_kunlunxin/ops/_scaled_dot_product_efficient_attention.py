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
import math

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.runtime.backend._kunlunxin.ops.bmm import bmm
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)

_PHILOX_SEED = torch.tensor(0, dtype=torch.long)
_PHILOX_OFFSET = torch.tensor(0, dtype=torch.long)


@libentry()
@triton.jit
def _sdpea_softmax_kernel(
    Scores,
    ATTN_MASK,
    Weights,
    Lse,
    sm_scale,
    stride_s_row,
    stride_m_row,
    stride_w_row,
    Q_CTX,
    KV_CTX,
    BLOCK_N: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    HAS_ATTN_MASK: tl.constexpr,
):
    row = tl.program_id(0)
    qi = row % Q_CTX

    offs_n = tl.arange(0, BLOCK_N)
    kv_mask = offs_n < KV_CTX

    s_ptrs = Scores + row * stride_s_row + offs_n
    x = tl.load(s_ptrs, mask=kv_mask, other=0.0).to(tl.float32) * sm_scale

    if HAS_ATTN_MASK:
        m_ptrs = ATTN_MASK + row * stride_m_row + offs_n
        add_mask = tl.load(m_ptrs, mask=kv_mask, other=0.0).to(tl.float32)
        x += add_mask

    keep = kv_mask
    if IS_CAUSAL:
        keep = keep & (offs_n <= qi)

    x = tl.where(keep, x, -float("inf"))

    m = tl.max(x, axis=0)
    m_safe = tl.where(m == -float("inf"), 0.0, m)
    p = tl.exp(x - m_safe)
    p = tl.where(keep, p, 0.0)
    s = tl.sum(p, axis=0)
    s_safe = tl.where(s == 0.0, 1.0, s)

    # Natural-log log-sum-exp of the (scaled, masked) scores for this row.
    lse = m_safe + tl.log(s_safe)
    tl.store(Lse + row, lse)

    p = p / s_safe

    w_ptrs = Weights + row * stride_w_row + offs_n
    tl.store(w_ptrs, p.to(Weights.dtype.element_ty), mask=kv_mask)


def _scaled_dot_product_efficient_attention(
    query,
    key,
    value,
    attn_bias=None,
    compute_log_sumexp=False,
    dropout_p=0.0,
    is_causal=False,
    scale=None,
):
    logger.debug("GEMS_KUNLUNXIN _SCALED_DOT_PRODUCT_EFFICIENT_ATTENTION")

    assert dropout_p == 0.0, "dropout_p != 0.0 is not supported"
    assert query.dim() == 4, "expected 4D (batch, heads, seq_len, head_dim) tensors"

    batch, heads, q_seq_len, head_dim = query.shape
    kv_seq_len = key.shape[2]

    if scale is None:
        scale = 1.0 / math.sqrt(head_dim)

    query = query.contiguous()
    key = key.contiguous()
    value = value.contiguous()

    q2 = query.reshape(batch * heads, q_seq_len, head_dim)
    k2 = key.reshape(batch * heads, kv_seq_len, head_dim)
    v2 = value.reshape(batch * heads, kv_seq_len, head_dim)

    # QK^T via the backend GEMM kernel, run in fp32 for score precision.
    kt = k2.transpose(1, 2).contiguous()
    scores = bmm(q2.float(), kt.float())  # (batch*heads, q, kv), fp32
    scores2d = scores.reshape(batch * heads * q_seq_len, kv_seq_len)

    weights = torch.empty(
        (batch, heads, q_seq_len, kv_seq_len),
        dtype=torch.float32,
        device=query.device,
    )
    weights2d = weights.reshape(batch * heads * q_seq_len, kv_seq_len)

    # log_sumexp is always float32, shape (batch, heads, q_seq_len).
    log_sumexp = torch.empty(
        (batch, heads, q_seq_len),
        dtype=torch.float32,
        device=query.device,
    )
    lse1d = log_sumexp.reshape(batch * heads * q_seq_len)

    has_attn_mask = attn_bias is not None
    if has_attn_mask:
        assert attn_bias.dtype != torch.bool, "boolean attn_bias is not supported"
        attn_bias = attn_bias.expand(batch, heads, q_seq_len, kv_seq_len).contiguous()
        am2d = attn_bias.reshape(batch * heads * q_seq_len, kv_seq_len)
        m_stride = am2d.stride(0)
    else:
        am2d = scores2d  # unused placeholder; HAS_ATTN_MASK guards all accesses
        m_stride = 0

    BLOCK_N = triton.next_power_of_2(kv_seq_len)
    grid = (batch * heads * q_seq_len,)

    with torch_device_fn.device(query.device):
        _sdpea_softmax_kernel[grid](
            scores2d,
            am2d,
            weights2d,
            lse1d,
            scale,
            scores2d.stride(0),
            m_stride,
            weights2d.stride(0),
            q_seq_len,
            kv_seq_len,
            BLOCK_N=BLOCK_N,
            IS_CAUSAL=is_causal,
            HAS_ATTN_MASK=has_attn_mask,
        )

    # Attention output: weights @ V via the backend GEMM kernel, fp32 accum.
    out2 = bmm(
        weights2d.reshape(batch * heads, q_seq_len, kv_seq_len), v2.float()
    )
    out = out2.reshape(batch, heads, q_seq_len, head_dim).to(query.dtype)

    return out, log_sumexp, _PHILOX_SEED, _PHILOX_OFFSET
