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


@libentry()
@triton.jit
def _sdpea_bwd_softmax_kernel(
    Scores,
    ATTN_MASK,
    Weights,
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
    p = p / s_safe

    w_ptrs = Weights + row * stride_w_row + offs_n
    tl.store(w_ptrs, p.to(Weights.dtype.element_ty), mask=kv_mask)


@libentry()
@triton.jit
def _sdpea_bwd_delta_kernel(
    DO,
    O,
    Delta,
    stride_do_row,
    stride_o_row,
    HEAD_DIM,
    BLOCK_D: tl.constexpr,
):
    row = tl.program_id(0)
    offs_d = tl.arange(0, BLOCK_D)
    d_mask = offs_d < HEAD_DIM

    do = tl.load(DO + row * stride_do_row + offs_d, mask=d_mask, other=0.0).to(
        tl.float32
    )
    o = tl.load(O + row * stride_o_row + offs_d, mask=d_mask, other=0.0).to(tl.float32)
    delta = tl.sum(do * o, axis=0)
    tl.store(Delta + row, delta)


@libentry()
@triton.jit
def _sdpea_bwd_ds_kernel(
    P,
    DP,
    Delta,
    DS,
    sm_scale,
    stride_p_row,
    stride_dp_row,
    stride_ds_row,
    KV_CTX,
    BLOCK_N: tl.constexpr,
    SCALE_OUT: tl.constexpr,
):
    row = tl.program_id(0)
    offs_n = tl.arange(0, BLOCK_N)
    kv_mask = offs_n < KV_CTX

    p = tl.load(P + row * stride_p_row + offs_n, mask=kv_mask, other=0.0).to(tl.float32)
    dp = tl.load(DP + row * stride_dp_row + offs_n, mask=kv_mask, other=0.0).to(
        tl.float32
    )
    delta = tl.load(Delta + row).to(tl.float32)

    ds = p * (dp - delta)
    if SCALE_OUT:
        ds = ds * sm_scale

    tl.store(DS + row * stride_ds_row + offs_n, ds, mask=kv_mask)


def _scaled_dot_product_efficient_attention_backward(
    grad_out,
    query,
    key,
    value,
    attn_bias,
    out,
    logsumexp,
    philox_seed,
    philox_offset,
    dropout_p,
    grad_input_mask,
    is_causal=False,
    *,
    scale=None,
):
    logger.debug(
        "GEMS_KUNLUNXIN _SCALED_DOT_PRODUCT_EFFICIENT_ATTENTION_BACKWARD"
    )

    assert dropout_p == 0.0, "dropout_p != 0.0 is not supported"
    assert query.dim() == 4, "expected 4D (batch, heads, seq_len, head_dim) tensors"

    need_dq, need_dk, need_dv, need_dbias = grad_input_mask

    batch, heads, q_seq_len, head_dim = query.shape
    kv_seq_len = key.shape[2]

    if scale is None:
        scale = 1.0 / math.sqrt(head_dim)

    q = query.contiguous()
    k = key.contiguous()
    v = value.contiguous()
    o = out.contiguous()
    do = grad_out.contiguous()

    bh = batch * heads
    q2 = q.reshape(bh, q_seq_len, head_dim).float()
    k2 = k.reshape(bh, kv_seq_len, head_dim).float()
    v2 = v.reshape(bh, kv_seq_len, head_dim).float()
    o2 = o.reshape(bh, q_seq_len, head_dim).float()
    do2 = do.reshape(bh, q_seq_len, head_dim).float()

    # scores = scale * Q @ K^T  (fp32 accum via vendor bmm, non-tf32).
    kt = k2.transpose(1, 2).contiguous()
    scores = bmm(q2, kt)  # (bh, q, kv)
    scores2d = scores.reshape(bh * q_seq_len, kv_seq_len)

    weights = torch.empty(
        (bh, q_seq_len, kv_seq_len), dtype=torch.float32, device=query.device
    )
    weights2d = weights.reshape(bh * q_seq_len, kv_seq_len)

    has_attn_mask = attn_bias is not None
    if has_attn_mask:
        assert attn_bias.dtype != torch.bool, "boolean attn_bias is not supported"
        am = attn_bias.expand(batch, heads, q_seq_len, kv_seq_len).contiguous()
        am2d = am.reshape(bh * q_seq_len, kv_seq_len)
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

    # dP = dO @ V^T  (bh, q, kv), fp32.
    vt = v2.transpose(1, 2).contiguous()
    dp = bmm(do2, vt)
    dp2d = dp.reshape(bh * q_seq_len, kv_seq_len)

    # delta_i = sum_d(dO_i * O_i), fp32, shape (bh*q,).
    delta = torch.empty((bh * q_seq_len,), dtype=torch.float32, device=query.device)
    do2d_d = do2.reshape(bh * q_seq_len, head_dim)
    o2d_d = o2.reshape(bh * q_seq_len, head_dim)
    BLOCK_D = triton.next_power_of_2(head_dim)
    with torch_device_fn.device(query.device):
        _sdpea_bwd_delta_kernel[grid_rows](
            do2d_d,
            o2d_d,
            delta,
            do2d_d.stride(0),
            o2d_d.stride(0),
            head_dim,
            BLOCK_D=BLOCK_D,
        )

    # dS_raw = P * (dP - delta)  -> gradient w.r.t. the (scaled) logits.
    # dBias (if requested) is this unscaled dS; dQ/dK fold in sm_scale.
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

    ds_scaled = ds_raw * scale  # for dQ/dK

    # dV = P^T @ dO  (bh, kv, d).
    if need_dv:
        pt = weights.transpose(1, 2).contiguous()
        dv2 = bmm(pt, do2)
        dv = dv2.reshape(batch, heads, kv_seq_len, head_dim).to(query.dtype)
    else:
        dv = torch.zeros_like(value)

    # dQ = scale * dS @ K  (bh, q, d).
    if need_dq:
        dq2 = bmm(ds_scaled, k2)
        dq = dq2.reshape(batch, heads, q_seq_len, head_dim).to(query.dtype)
    else:
        dq = torch.zeros_like(query)

    # dK = scale * dS^T @ Q  (bh, kv, d).
    if need_dk:
        dst = ds_scaled.transpose(1, 2).contiguous()
        dk2 = bmm(dst, q2)
        dk = dk2.reshape(batch, heads, kv_seq_len, head_dim).to(query.dtype)
    else:
        dk = torch.zeros_like(key)

    # dBias = dS_raw (gradient w.r.t. the additive bias in the logits).
    if need_dbias and has_attn_mask:
        dbias = ds_raw.reshape(batch, heads, q_seq_len, kv_seq_len).to(attn_bias.dtype)
    else:
        dbias = None

    return dq, dk, dv, dbias
