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

import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Triton kernels: sparse attention with attention-sink (昆仑芯适配版本)
#
# The generic (nvidia) implementation runs one fused kernel that mixes
# ``tl.dot`` with a data-dependent gather (``topk``) and with softmax
# reductions.  On this backend that combination is broken (see
# ``sparse_mla.py`` / ``flashmla_sparse.py`` for the same class of findings):
# * ``tl.dot`` mixed with a data-dependent gather is a hard compile failure.
# * A 2D reduction along the *outer* axis (``tl.sum(t, axis=0)`` of a
#   (BK, D) tile) exhausts the CTA unified-SRAM ("out of resource:
#   uni_sram").  All 2D reductions therefore reduce the inner (contiguous)
#   axis.
# * Mixing a transcendental (``tl.exp``) with a 2D (non-affine) gather in the
#   same kernel also exhausts uni_sram.
# * A transposed (D, BK) *gather* whose inner axis is strided compiles but
#   silently returns wrong/constant values.  The PV pass therefore runs on a
#   dense, pre-gathered ``gkv_td`` buffer with an affine ``tl.dot``.
#
# So the op is split into four kernels, none of which mixes ``tl.dot`` with a
# data-dependent address or a transcendental:
#
#   A ``_sparse_attn_scores_kernel`` : gather + 1D dot       -> logits
#   B ``_sparse_attn_softmax_kernel``: 1D reductions/exp     -> probs
#   C ``_sparse_attn_gather_td_kernel``: gather, no tl.dot  -> gkv_td
#   D ``_sparse_attn_pv_kernel``     : dense tl.dot only     -> output
#
# The attention-sink is incorporated once, in the softmax kernel.  All
# intermediate buffers are over-allocated to whole tile boundaries
# (``TP = cdiv(topk, BLOCK_K) * BLOCK_K``, ``AH`` head blocks) so that every
# store is unmasked (masked stores are known to write past tight allocations
# on this backend).  When a head block overruns ``H`` (possible only for tiny
# ``H < 16`` shapes), the output is written into a padded buffer and the
# wrapper returns a slimmed view of it.
# ---------------------------------------------------------------------------


@triton.jit
def _sparse_attn_scores_kernel(
    Q,  # (b, m, h, d)   bf16
    KV,  # (b, n, d)     bf16
    LOGITS,  # (b*m*AH, TP) fp32, TP = cdiv(topk, BLOCK_K) * BLOCK_K
    topk_idxs,  # (b, m, topk) int32
    stride_qb,
    stride_qm,
    stride_qh,
    stride_qd,
    stride_kvb,
    stride_kvn,
    stride_kvd,
    stride_idxb,
    stride_idxm,
    stride_idxk,
    scale,
    topk,
    TP,
    D: tl.constexpr,
    BLOCK_K: tl.constexpr,
    M: tl.constexpr,
    H: tl.constexpr,
):
    # grid = (b*m*h, cdiv(topk, BLOCK_K)) -- one CTA per (pos, head, k-tile)
    i_bmh = tl.program_id(0)
    i_t = tl.program_id(1)
    i_h = i_bmh % H
    i_bm = i_bmh // H
    i_m = i_bm % M
    i_b = i_bm // M

    offs_d = tl.arange(0, D)
    q = tl.load(
        Q + i_b * stride_qb + i_m * stride_qm + i_h * stride_qh + offs_d * stride_qd
    ).to(tl.float32)  # (D,)

    offs_k = i_t * BLOCK_K + tl.arange(0, BLOCK_K)
    ks = tl.minimum(offs_k, topk - 1)  # clamp address, never OOB
    ids = tl.load(
        topk_idxs + i_b * stride_idxb + i_m * stride_idxm + ks * stride_idxk
    )  # (BLOCK_K,)
    # -1 is the padding sentinel: invalid positions get -inf (zero prob),
    # matching the generic (non-XPU) kernel semantics.
    valid = (offs_k < topk) & (ids >= 0)
    ids = tl.where(valid, ids, 0)

    # -- gather KV block (BLOCK_K, D), masked non-affine gather --
    kv = tl.load(
        KV
        + i_b * stride_kvb
        + ids[:, None] * stride_kvn
        + offs_d[None, :] * stride_kvd,
        mask=valid[:, None],
        other=0.0,
    )  # (BLOCK_K, D) bf16

    # -- scores: 1D dot (sum over contig D axis) --
    sc = tl.sum(q[None, :] * kv.to(tl.float32), axis=1)  # (BLOCK_K,)
    sc = sc * scale
    sc = tl.where(valid, sc, float("-inf"))

    tl.store(LOGITS + i_bmh * TP + offs_k, sc)  # unmasked: offs_k < TP


@triton.jit
def _sparse_attn_softmax_kernel(
    LOGITS,  # (b*m*AH, TP) fp32 (already scaled)
    PROBS,  # (b*m*AH, TP) bf16
    attn_sink,  # (h,) fp32
    TP,
    topk,
    H_ACTUAL,
    H: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # grid = (b*m*AH,) -- one CTA per (pos, head); flat 1D tiles only
    i_bmh = tl.program_id(0)
    i_h = i_bmh % H
    offs_k = tl.arange(0, BLOCK_K)
    lg_base = LOGITS + i_bmh * TP
    p_base = PROBS + i_bmh * TP
    n_blocks = (topk + BLOCK_K - 1) // BLOCK_K

    # pass 1: max over the (already scaled) logits
    run_max = float("-inf")
    for t in range(n_blocks):
        k_offs = t * BLOCK_K + offs_k
        x = tl.load(lg_base + k_offs)
        x = tl.where(k_offs < topk, x, float("-inf"))
        run_max = tl.maximum(run_max, tl.max(x))

    # -- incorporate attn_sink into the normalization --
    if i_h < H_ACTUAL:
        sink_val = tl.load(attn_sink + i_h)
        run_sum = tl.exp(sink_val - run_max)
    else:
        run_sum = 0.0
    total_sum = run_sum
    for t in range(n_blocks):
        k_offs = t * BLOCK_K + offs_k
        x = tl.load(lg_base + k_offs)
        x = tl.where(k_offs < topk, x, float("-inf"))
        total_sum += tl.sum(tl.exp(x - run_max))

    lse = run_max + tl.math.log(total_sum)

    # pass 3: store normalized probs (all p >= 0; padding rows are 0)
    for t in range(n_blocks):
        k_offs = t * BLOCK_K + offs_k
        x = tl.load(lg_base + k_offs)
        x = tl.where(k_offs < topk, x, float("-inf"))
        p = tl.exp(x - lse)
        p = tl.where(k_offs < topk, p, 0.0)
        tl.store(p_base + k_offs, p.to(tl.bfloat16))


@triton.jit
def _sparse_attn_gather_td_kernel(
    KV,  # (b, n, d)     bf16
    GKV_TD,  # (b*m, TP, D) bf16, d-contiguous
    topk_idxs,  # (b, m, topk) int32
    stride_kvb,
    stride_kvn,
    stride_kvd,
    stride_idxb,
    stride_idxm,
    stride_idxk,
    topk,
    TP,
    D: tl.constexpr,
    BT: tl.constexpr,
    BD: tl.constexpr,
    M: tl.constexpr,
):
    # grid = (b*m, (TP//BT) * (D//BD)) -- one CTA per (pos, k-tile, d-tile)
    i_bm = tl.program_id(0).to(tl.int64)
    i_z = tl.program_id(1)
    i_d = i_z % (D // BD)
    i_t = i_z // (D // BD)
    i_b = i_bm // M
    i_m = i_bm % M

    offs_t = i_t * BT + tl.arange(0, BT)
    offs_d = i_d * BD + tl.arange(0, BD)
    in_range = offs_t < topk
    t_off = tl.minimum(offs_t, topk - 1)  # clamp address, never OOB
    ids = tl.load(
        topk_idxs + i_b * stride_idxb + i_m * stride_idxm + t_off * stride_idxk
    )
    ids_safe = tl.where(in_range & (ids >= 0), ids, 0)

    # [BT, BD] tile: rows are gathered kv rows, d contiguous
    v = tl.load(
        KV
        + i_b * stride_kvb
        + ids_safe[:, None] * stride_kvn
        + offs_d[None, :] * stride_kvd
    )  # (BT, BD) bf16

    tl.store(
        GKV_TD + i_bm * (TP * D) + offs_t[:, None] * D + offs_d[None, :], v
    )  # unmasked: offs_t < TP


@triton.jit
def _sparse_attn_pv_kernel(
    PROBS,  # (b*m*AH, TP) bf16 (normalized)
    GKV_TD,  # (b*m, TP, D) bf16
    O,  # (b*m*AH, D) bf16
    TP: tl.constexpr,
    D: tl.constexpr,
    BH: tl.constexpr,
    BT: tl.constexpr,
    BDV: tl.constexpr,
    AH: tl.constexpr,
):
    # grid = (b*m, (AH//BH) * (D//BDV)) -- dense tl.dot only
    i_bm = tl.program_id(0).to(tl.int64)
    i_z = tl.program_id(1)
    i_v = i_z % (D // BDV)
    i_bh = i_z // (D // BDV)

    offs_h = i_bh * BH + tl.arange(0, BH)
    offs_t = tl.arange(0, BT)
    offs_v = i_v * BDV + tl.arange(0, BDV)

    p_base = PROBS + (i_bm * AH + offs_h[:, None]) * TP
    v_base = GKV_TD + i_bm * (TP * D)

    acc = tl.zeros([BH, BDV], dtype=tl.float32)
    for it in range(TP // BT):
        pb = tl.load(p_base + it * BT + offs_t[None, :])  # (BH, BT) bf16
        vb = tl.load(v_base + (it * BT + offs_t)[:, None] * D + offs_v[None, :])  # (BT, BDV) bf16
        acc = tl.dot(pb, vb, acc, out_dtype=tl.float32)

    tl.store(O + (i_bm * AH + offs_h[:, None]) * D + offs_v[None, :], acc.to(tl.bfloat16))


# ---------------------------------------------------------------------------
# Python wrapper
# ---------------------------------------------------------------------------
def sparse_attn_triton(
    q: torch.Tensor,
    kv: torch.Tensor,
    attn_sink: torch.Tensor,
    topk_idxs: torch.Tensor,
    softmax_scale: float,
) -> torch.Tensor:
    b, m, h, d = q.shape
    topk = topk_idxs.shape[-1]

    # Tile sizes. 64 is the smallest value that is safe on this backend
    # (BLOCK_N == 16 does not compile, 32 silently corrupts pointwise tiles,
    # and 2D tiles with a row pitch < 64 silently overwrite following rows).
    BT = 64  # topk tile
    BD = 64  # d tile used by the gather kernel
    BDV = 256  # value-dim tile used by the PV matmul
    BH = max(16, min(64, triton.next_power_of_2(h)))  # head tile

    n_blocks = (topk + BT - 1) // BT
    tp = n_blocks * BT
    AH = ((h + BH - 1) // BH) * BH
    ND = (d + BD - 1) // BD
    NDV = (d + BDV - 1) // BDV
    bm = b * m

    logits = torch.zeros((bm * AH, tp), device=q.device, dtype=torch.float32)
    probs = torch.empty((bm * AH, tp), device=q.device, dtype=torch.bfloat16)
    gkv_td = torch.empty((bm, tp, d), device=q.device, dtype=torch.bfloat16)
    o_pad = torch.empty((bm * AH, d), device=q.device, dtype=torch.bfloat16)

    _sparse_attn_scores_kernel[(b * m * h, n_blocks)](
        q,
        kv,
        logits,
        topk_idxs,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        q.stride(3),
        kv.stride(0),
        kv.stride(1),
        kv.stride(2),
        topk_idxs.stride(0),
        topk_idxs.stride(1),
        topk_idxs.stride(2),
        softmax_scale,
        topk,
        tp,
        D=d,
        BLOCK_K=BT,
        M=m,
        H=h,
        num_warps=8,
    )

    _sparse_attn_softmax_kernel[(bm * AH,)](
        logits,
        probs,
        attn_sink,
        tp,
        topk,
        h,
        H=AH,
        BLOCK_K=BT,
        num_warps=4,
    )

    _sparse_attn_gather_td_kernel[(bm, n_blocks * ND)](
        kv,
        gkv_td,
        topk_idxs,
        kv.stride(0),
        kv.stride(1),
        kv.stride(2),
        topk_idxs.stride(0),
        topk_idxs.stride(1),
        topk_idxs.stride(2),
        topk,
        tp,
        D=d,
        BT=BT,
        BD=BD,
        M=m,
    )

    _sparse_attn_pv_kernel[(bm, (AH // BH) * NDV)](
        probs,
        gkv_td,
        o_pad,
        TP=tp,
        D=d,
        BH=BH,
        BT=BT,
        BDV=BDV,
        AH=AH,
    )

    return o_pad.view(b, m, AH, d)[:, :, :h, :].contiguous()