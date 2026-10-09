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

from flag_gems.runtime import torch_device_fn
from flag_gems.runtime.backend._kunlunxin.ops._scaled_dot_product_efficient_attention_backward import (
    _sdpea_bwd_delta_kernel,
    _sdpea_bwd_ds_kernel,
    _sdpea_bwd_softmax_kernel,
)
from flag_gems.runtime.backend._kunlunxin.ops.bmm import bmm

logger = logging.getLogger(__name__)


def _normalize_bias(bias, batch, heads, q_seq_len, kv_seq_len):
    b = bias
    if b.dim() == 2:
        b = b.reshape(1, 1, q_seq_len, kv_seq_len)
    elif b.dim() == 3:
        b = b.reshape(batch, 1, q_seq_len, kv_seq_len)
    b = b.expand(batch, heads, q_seq_len, kv_seq_len).contiguous()
    return b.reshape(batch * heads * q_seq_len, kv_seq_len)


def cudnn_attention_backward(
    grad_out,
    query,
    key,
    value,
    out,
    logsumexp,
    philox_seed,
    philox_offset,
    attn_bias,
    cum_seq_q,
    cum_seq_k,
    max_q,
    max_k,
    dropout_p,
    is_causal,
    *,
    scale=None,
):
    """BHSD cuDNN-attention backward via vendor bmm + trailing-axis kernels.

    Reuses the efficient-attention-backward overlay math (fp32/no-tf32 bmm
    for dV/dP/dQ/dK plus 1-D softmax-recompute / delta / dS kernels),
    wrapped for the cuDNN backward signature and generalized for GQA,
    unequal q/k vs value head dims, and 2D/3D/4D broadcast bias.
    """
    logger.debug("GEMS_KUNLUNXIN CUDNN_ATTENTION_BACKWARD")

    if dropout_p is not None and dropout_p > 0.0:
        raise NotImplementedError(
            "cudnn_attention_backward: dropout > 0 is not supported"
        )
    if (cum_seq_q is not None) or (cum_seq_k is not None):
        raise NotImplementedError(
            "cudnn_attention_backward: varlen is not supported"
        )

    assert query.dim() == 4, "expected 4D (batch, heads, seq_len, head_dim) tensors"

    batch, heads, q_seq_len, head_dim = query.shape
    kv_heads = key.shape[1]
    kv_seq_len = key.shape[2]
    value_head_dim = value.shape[3]

    if scale is None:
        scale = 1.0 / math.sqrt(head_dim)

    group = heads // kv_heads
    q = query.contiguous()
    if group != 1:
        k = key.repeat_interleave(group, dim=1).contiguous()
        v = value.repeat_interleave(group, dim=1).contiguous()
    else:
        k = key.contiguous()
        v = value.contiguous()
    o = out.contiguous()
    do = grad_out.contiguous()

    bh = batch * heads
    q2 = q.reshape(bh, q_seq_len, head_dim).float()
    k2 = k.reshape(bh, kv_seq_len, head_dim).float()
    v2 = v.reshape(bh, kv_seq_len, value_head_dim).float()
    o2 = o.reshape(bh, q_seq_len, value_head_dim).float()
    do2 = do.reshape(bh, q_seq_len, value_head_dim).float()

    # scores = Q @ K^T (fp32 accum, non-tf32); sm_scale applied in kernel.
    kt = k2.transpose(1, 2).contiguous()
    scores = bmm(q2, kt)
    scores2d = scores.reshape(bh * q_seq_len, kv_seq_len)

    weights = torch.empty(
        (bh, q_seq_len, kv_seq_len), dtype=torch.float32, device=query.device
    )
    weights2d = weights.reshape(bh * q_seq_len, kv_seq_len)

    has_attn_mask = attn_bias is not None
    if has_attn_mask:
        assert attn_bias.dtype != torch.bool, "boolean attn_bias is not supported"
        am2d = _normalize_bias(attn_bias, batch, heads, q_seq_len, kv_seq_len)
        m_stride = am2d.stride(0)
    else:
        am2d = scores2d
        m_stride = 0

    BLOCK_N = triton.next_power_of_2(kv_seq_len)
    grid_rows = (bh * q_seq_len,)

    with torch_device_fn.device(query.device):
        _sdpea_bwd_softmax_kernel[grid_rows](
            scores2d,
            am2d,
            weights2d,
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

    # dP = dO @ V^T (bh, q, kv).
    vt = v2.transpose(1, 2).contiguous()
    dp = bmm(do2, vt)
    dp2d = dp.reshape(bh * q_seq_len, kv_seq_len)

    # delta_i = sum over value head dim of (dO_i * O_i).
    delta = torch.empty((bh * q_seq_len,), dtype=torch.float32, device=query.device)
    do2d_d = do2.reshape(bh * q_seq_len, value_head_dim)
    o2d_d = o2.reshape(bh * q_seq_len, value_head_dim)
    BLOCK_D = triton.next_power_of_2(value_head_dim)
    with torch_device_fn.device(query.device):
        _sdpea_bwd_delta_kernel[grid_rows](
            do2d_d,
            o2d_d,
            delta,
            do2d_d.stride(0),
            o2d_d.stride(0),
            value_head_dim,
            BLOCK_D=BLOCK_D,
        )

    ds_raw = torch.empty(
        (bh, q_seq_len, kv_seq_len), dtype=torch.float32, device=query.device
    )
    ds_raw2d = ds_raw.reshape(bh * q_seq_len, kv_seq_len)
    with torch_device_fn.device(query.device):
        _sdpea_bwd_ds_kernel[grid_rows](
            weights2d,
            dp2d,
            delta,
            ds_raw2d,
            scale,
            weights2d.stride(0),
            dp2d.stride(0),
            ds_raw2d.stride(0),
            kv_seq_len,
            BLOCK_N=BLOCK_N,
            SCALE_OUT=False,
        )

    ds_scaled = ds_raw * scale

    # dV = P^T @ dO ; dQ = scale * dS @ K ; dK = scale * dS^T @ Q.
    pt = weights.transpose(1, 2).contiguous()
    dv2 = bmm(pt, do2)
    dq2 = bmm(ds_scaled, k2)
    dst = ds_scaled.transpose(1, 2).contiguous()
    dk2 = bmm(dst, q2)

    dq = dq2.reshape(batch, heads, q_seq_len, head_dim).to(query.dtype)
    dk_full = dk2.reshape(batch, heads, kv_seq_len, head_dim)
    dv_full = dv2.reshape(batch, heads, kv_seq_len, value_head_dim)

    if group != 1:
        dk_full = dk_full.reshape(
            batch, kv_heads, group, kv_seq_len, head_dim
        ).sum(dim=2)
        dv_full = dv_full.reshape(
            batch, kv_heads, group, kv_seq_len, value_head_dim
        ).sum(dim=2)

    dk = dk_full.to(key.dtype)
    dv = dv_full.to(value.dtype)
    return dq, dk, dv
