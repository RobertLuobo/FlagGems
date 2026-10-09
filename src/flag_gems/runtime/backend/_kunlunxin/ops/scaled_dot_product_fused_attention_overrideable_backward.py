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
def _recompute_p_kernel(
    Scores,
    Bias,
    Lse,
    P,
    sm_scale,
    Sq,
    Skv,
    stride_sz,
    stride_sm,
    stride_sn,
    stride_bz,
    stride_bm,
    stride_bn,
    stride_lz,
    stride_lm,
    stride_pz,
    stride_pm,
    stride_pn,
    BLOCK_N: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    HAS_BIAS: tl.constexpr,
):
    pid = tl.program_id(0)
    row = pid % Sq
    z = pid // Sq

    offs_n = tl.arange(0, BLOCK_N)
    n_mask = offs_n < Skv

    s_ptrs = Scores + z * stride_sz + row * stride_sm + offs_n * stride_sn
    s = tl.load(s_ptrs, mask=n_mask, other=0.0).to(tl.float32) * sm_scale

    if HAS_BIAS:
        b_ptrs = Bias + z * stride_bz + row * stride_bm + offs_n * stride_bn
        s += tl.load(b_ptrs, mask=n_mask, other=0.0).to(tl.float32)

    keep = n_mask
    if IS_CAUSAL:
        keep = keep & (offs_n <= row)

    lse = tl.load(Lse + z * stride_lz + row * stride_lm).to(tl.float32)
    p = tl.exp(s - lse)
    p = tl.where(keep, p, 0.0)

    p_ptrs = P + z * stride_pz + row * stride_pm + offs_n * stride_pn
    tl.store(p_ptrs, p, mask=n_mask)


@triton.jit
def _rowdot_kernel(
    dOut,
    Out,
    D,
    Sq,
    HeadDim,
    stride_oz,
    stride_om,
    stride_od,
    stride_dz,
    stride_dm,
    BLOCK_D: tl.constexpr,
):
    pid = tl.program_id(0)
    row = pid % Sq
    z = pid // Sq

    offs_d = tl.arange(0, BLOCK_D)
    d_mask = offs_d < HeadDim

    base = z * stride_oz + row * stride_om + offs_d * stride_od
    do = tl.load(dOut + base, mask=d_mask, other=0.0).to(tl.float32)
    o = tl.load(Out + base, mask=d_mask, other=0.0).to(tl.float32)
    d = tl.sum(do * o, 0)

    tl.store(D + z * stride_dz + row * stride_dm, d)


@triton.jit
def _ds_kernel(
    P,
    dP,
    D,
    dS,
    dS_scaled,
    sm_scale,
    Sq,
    Skv,
    stride_pz,
    stride_pm,
    stride_pn,
    stride_qz,
    stride_qm,
    stride_qn,
    stride_dz,
    stride_dm,
    stride_sz,
    stride_sm,
    stride_sn,
    stride_cz,
    stride_cm,
    stride_cn,
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(0)
    row = pid % Sq
    z = pid // Sq

    offs_n = tl.arange(0, BLOCK_N)
    n_mask = offs_n < Skv

    p = tl.load(
        P + z * stride_pz + row * stride_pm + offs_n * stride_pn,
        mask=n_mask,
        other=0.0,
    ).to(tl.float32)
    dp = tl.load(
        dP + z * stride_qz + row * stride_qm + offs_n * stride_qn,
        mask=n_mask,
        other=0.0,
    ).to(tl.float32)
    d = tl.load(D + z * stride_dz + row * stride_dm).to(tl.float32)

    ds = p * (dp - d)
    ds_scaled = ds * sm_scale

    tl.store(
        dS + z * stride_sz + row * stride_sm + offs_n * stride_sn, ds, mask=n_mask
    )
    tl.store(
        dS_scaled + z * stride_cz + row * stride_cm + offs_n * stride_cn,
        ds_scaled,
        mask=n_mask,
    )


def scaled_dot_product_fused_attention_overrideable_backward(
    grad_out,
    query,
    key,
    value,
    attn_bias,
    grad_input_mask,
    out,
    logsumexp,
    cum_seq_q,
    cum_seq_k,
    max_q,
    max_k,
    dropout_p,
    is_causal,
    philox_seed,
    philox_offset,
    *,
    scale=None,
):
    """XPU overlay for fused/overrideable SDPA backward.

    The generic implementation delegates to a flash-attention backward Triton
    kernel whose ``tl.dot``/``tl.trans`` fail the XPU3 TritonSDNN pipeline.
    This overlay recomputes the softmax weights from the saved ``logsumexp``
    and builds dQ/dK/dV/dBias from the vendor ``bmm`` kernel (fp32 GEMMs) plus
    XPU3-safe elementwise/reduction kernels (``tl.exp`` not ``tl.exp2``,
    trailing-axis reduce, no ``tl.dot``/``tl.trans``).
    """
    logger.debug(
        "GEMS_KUNLUNXIN SCALED_DOT_PRODUCT_FUSED_ATTENTION_OVERRIDEABLE_BACKWARD"
    )

    assert dropout_p == 0.0, "dropout_p != 0.0 is not supported"
    assert (cum_seq_q is None) and (
        cum_seq_k is None
    ), "varlen (cum_seq_q/cum_seq_k) is not supported"
    assert query.dim() == 4, "expected 4D (batch, heads, seq_len, head_dim) tensors"

    need_dq, need_dk, need_dv, need_dbias = grad_input_mask
    bias_req_grad = bool(need_dbias) and (attn_bias is not None)

    batch, heads, q_seq_len, head_dim = query.shape
    kv_seq_len = key.shape[2]
    dtype = query.dtype
    device = query.device

    if scale is None:
        scale = 1.0 / math.sqrt(head_dim)
    scale = float(scale)

    bh = batch * heads
    q3 = query.reshape(bh, q_seq_len, head_dim).contiguous()
    k3 = key.reshape(bh, kv_seq_len, head_dim).contiguous()
    v3 = value.reshape(bh, kv_seq_len, head_dim).contiguous()
    do3 = grad_out.reshape(bh, q_seq_len, head_dim).contiguous()
    out3 = out.reshape(bh, q_seq_len, head_dim).contiguous()
    lse2 = logsumexp.reshape(bh, q_seq_len).float().contiguous()

    # scores = Q @ K^T (fp32), softmax scale applied inside the recompute kernel.
    scores = bmm(q3.float(), k3.float().transpose(1, 2))

    has_bias = attn_bias is not None
    if has_bias:
        bias3 = attn_bias.reshape(bh, q_seq_len, kv_seq_len).contiguous()
        b_strides = (bias3.stride(0), bias3.stride(1), bias3.stride(2))
    else:
        bias3 = scores
        b_strides = (0, 0, 0)

    p3 = torch.empty((bh, q_seq_len, kv_seq_len), dtype=torch.float32, device=device)

    BLOCK_N = triton.next_power_of_2(kv_seq_len)
    grid_rows = (bh * q_seq_len,)
    with torch_device_fn.device(device):
        _recompute_p_kernel[grid_rows](
            scores,
            bias3,
            lse2,
            p3,
            scale,
            q_seq_len,
            kv_seq_len,
            scores.stride(0),
            scores.stride(1),
            scores.stride(2),
            b_strides[0],
            b_strides[1],
            b_strides[2],
            lse2.stride(0),
            lse2.stride(1),
            p3.stride(0),
            p3.stride(1),
            p3.stride(2),
            BLOCK_N=BLOCK_N,
            IS_CAUSAL=is_causal,
            HAS_BIAS=has_bias,
        )

    # dV = P^T @ dOut   (fp32 GEMM)
    dV3 = bmm(p3.transpose(1, 2), do3.float())

    # dP = dOut @ V^T   (fp32 GEMM)
    dP3 = bmm(do3.float(), v3.float().transpose(1, 2))

    # D = rowsum(dOut * Out) over head_dim
    D2 = torch.empty((bh, q_seq_len), dtype=torch.float32, device=device)
    BLOCK_D = triton.next_power_of_2(head_dim)
    with torch_device_fn.device(device):
        _rowdot_kernel[grid_rows](
            do3,
            out3,
            D2,
            q_seq_len,
            head_dim,
            do3.stride(0),
            do3.stride(1),
            do3.stride(2),
            D2.stride(0),
            D2.stride(1),
            BLOCK_D=BLOCK_D,
        )

    # dS = P * (dP - D); dS_scaled = dS * scale
    dS3 = torch.empty((bh, q_seq_len, kv_seq_len), dtype=torch.float32, device=device)
    dS_scaled3 = torch.empty(
        (bh, q_seq_len, kv_seq_len), dtype=torch.float32, device=device
    )
    with torch_device_fn.device(device):
        _ds_kernel[grid_rows](
            p3,
            dP3,
            D2,
            dS3,
            dS_scaled3,
            scale,
            q_seq_len,
            kv_seq_len,
            p3.stride(0),
            p3.stride(1),
            p3.stride(2),
            dP3.stride(0),
            dP3.stride(1),
            dP3.stride(2),
            D2.stride(0),
            D2.stride(1),
            dS3.stride(0),
            dS3.stride(1),
            dS3.stride(2),
            dS_scaled3.stride(0),
            dS_scaled3.stride(1),
            dS_scaled3.stride(2),
            BLOCK_N=BLOCK_N,
        )

    # dQ = (dS * scale) @ K ;  dK = (dS * scale)^T @ Q   (fp32 GEMMs)
    dQ3 = bmm(dS_scaled3, k3.float())
    dK3 = bmm(dS_scaled3.transpose(1, 2), q3.float())

    dQ = dQ3.reshape(batch, heads, q_seq_len, head_dim).to(dtype)
    dK = dK3.reshape(batch, heads, kv_seq_len, head_dim).to(dtype)
    dV = dV3.reshape(batch, heads, kv_seq_len, head_dim).to(dtype)

    if not need_dq:
        dQ = torch.zeros_like(query)
    if not need_dk:
        dK = torch.zeros_like(key)
    if not need_dv:
        dV = torch.zeros_like(value)

    if bias_req_grad:
        dBias = dS3.reshape(batch, heads, q_seq_len, kv_seq_len).to(dtype)
    else:
        dBias = None

    return dQ, dK, dV, dBias
