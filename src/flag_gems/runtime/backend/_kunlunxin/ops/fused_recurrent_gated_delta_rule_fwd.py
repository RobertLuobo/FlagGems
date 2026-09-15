# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

import logging

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)


def _fused_recurrent_gated_delta_rule_fwd_python(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float,
    initial_state: torch.Tensor,
    inplace_final_state: bool = True,
    cu_seqlens: torch.LongTensor | None = None,
    ssm_state_indices: torch.Tensor | None = None,
    num_accepted_tokens: torch.Tensor | None = None,
    use_qk_l2norm_in_kernel: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    logger.debug("GEMS_KUNLUNXIN FUSED RECURRENT GATED DELTA RULE FWD (PYTHON)")
    batch, seq_len, heads, _ = q.shape
    value_heads = v.shape[2]
    output = torch.zeros_like(v)
    source_state = initial_state.clone() if inplace_final_state else initial_state
    if inplace_final_state:
        final_state = initial_state
    else:
        final_state = torch.zeros(
            seq_len,
            value_heads,
            k.shape[-1],
            v.shape[-1],
            dtype=initial_state.dtype,
            device=initial_state.device,
        )

    num_sequences = batch if cu_seqlens is None else len(cu_seqlens) - 1
    for sequence in range(num_sequences):
        if cu_seqlens is None:
            batch_idx, begin, end = sequence, 0, seq_len
        else:
            batch_idx = 0
            begin = cu_seqlens[sequence].item()
            end = cu_seqlens[sequence + 1].item()

        initial_idx = (
            sequence if ssm_state_indices is None else ssm_state_indices[begin].item()
        )
        for value_head in range(value_heads):
            query_head = value_head // (value_heads // heads)
            state = source_state[initial_idx, value_head].float().clone()
            for position in range(begin, end):
                query = q[batch_idx, position, query_head].float()
                key = k[batch_idx, position, query_head].float()
                value = v[batch_idx, position, value_head].float()
                if use_qk_l2norm_in_kernel:
                    query = query / (query.norm() + 1e-6)
                    key = key / (key.norm() + 1e-6)
                query = query * scale
                state = state * torch.exp(g[batch_idx, position, value_head].float())
                value = value - (state * key[:, None]).sum(0)
                value = value * beta[batch_idx, position, value_head].float()
                state = state + key[:, None] * value[None, :]
                output[batch_idx, position, value_head] = (
                    (state * query[:, None]).sum(0).to(output.dtype)
                )

                state_idx = (
                    sequence
                    if ssm_state_indices is None
                    else ssm_state_indices[position].item()
                )
                if inplace_final_state:
                    final_state[state_idx, value_head] = state.to(final_state.dtype)
                else:
                    final_state[position, value_head] = state.to(final_state.dtype)

    return output, final_state


@triton.jit
def _fused_recurrent_gated_delta_rule_fwd_kernel(
    q,
    k,
    v,
    g,
    beta,
    o,
    h0,
    ht,
    cu_seqlens,
    ssm_state_indices,
    scale,
    T,  # sequence length (runtime), used only when cu_seqlens is None
    stride_q_t,
    stride_q_h,
    stride_q_k,
    stride_k_t,
    stride_k_h,
    stride_k_k,
    stride_v_t,
    stride_v_hv,
    stride_v_v,
    stride_o_t,
    stride_o_hv,
    stride_o_v,
    stride_g_t,
    stride_g_hv,
    stride_beta_t,
    stride_beta_hv,
    stride_cu,
    stride_ssm,
    H: tl.constexpr,  # number of q/k heads (Hg)
    HV: tl.constexpr,  # number of v heads
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    H0_STRIDE_S: tl.constexpr,
    H0_STRIDE_HV: tl.constexpr,
    HT_STRIDE_S: tl.constexpr,
    HT_STRIDE_HV: tl.constexpr,
    USE_CU: tl.constexpr,  # cu_seqlens is not None
    USE_SSM: tl.constexpr,  # ssm_state_indices is not None
    LAST_SEQ: tl.constexpr,  # last sequence index (N - 1)
    INPLACE: tl.constexpr,  # inplace_final_state (state is written back in-place)
    USE_L2: tl.constexpr,  # l2-normalize q/k inside the kernel
):
    # One program per (sequence, value head, value column); the K-vector state is
    # kept in registers and the per-token recurrence is unrolled sequentially.
    i_seq, i_hv, i_v = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    if USE_CU:
        t0 = tl.load(cu_seqlens + i_seq * stride_cu).to(tl.int64)
        t1 = tl.load(cu_seqlens + (i_seq + 1) * stride_cu).to(tl.int64)
    else:
        t0 = i_seq.to(tl.int64) * T
        t1 = t0 + T

    # q/k head of this value-head group (grouped heads: H divides HV)
    i_h = i_hv // (HV // H)
    offs = tl.arange(0, BK)

    # initial state slot: ssm_state_indices[sequence start] (or the sequence id)
    if USE_SSM:
        s0 = tl.load(ssm_state_indices + t0 * stride_ssm).to(tl.int64)
        s1 = tl.load(ssm_state_indices + (t1 - 1) * stride_ssm).to(tl.int64)
    else:
        s0 = i_seq.to(tl.int64)
        s1 = s0

    p_q = q + t0 * stride_q_t + i_h * stride_q_h + offs * stride_q_k
    p_k = k + t0 * stride_k_t + i_h * stride_k_h + offs * stride_k_k
    p_v = v + t0 * stride_v_t + i_hv * stride_v_hv + i_v * stride_v_v
    p_g = g + t0 * stride_g_t + i_hv * stride_g_hv
    p_beta = beta + t0 * stride_beta_t + i_hv * stride_beta_hv
    p_o = o + t0 * stride_o_t + i_hv * stride_o_hv + i_v * stride_o_v

    p_h = (
        h0
        + (s0 * H0_STRIDE_S + i_hv * H0_STRIDE_HV).to(tl.int64)
        + offs * V
        + i_v
    )
    h = tl.load(p_h).to(tl.float32)

    # S_t = exp(g_t) * S_{t-1} + beta_t * k_t^T (v_t - k_t @ S_{t-1})
    # o_t = q_t @ S_t * scale
    for it in range(0, t1 - t0):
        bq = tl.load(p_q).to(tl.float32)
        bk = tl.load(p_k).to(tl.float32)
        bv = tl.load(p_v).to(tl.float32)
        bg = tl.load(p_g).to(tl.float32)
        bb = tl.load(p_beta).to(tl.float32)

        if USE_L2:
            bq = bq / tl.sqrt(tl.sum(bq * bq) + 1e-6)
            bk = bk / tl.sqrt(tl.sum(bk * bk) + 1e-6)
        bq = bq * scale

        h = h * tl.exp(bg)
        acc = tl.sum(h * bk)  # (v_t - k_t @ S) component
        bv = (bv - acc) * bb
        h = h + bk * bv
        bo = tl.sum(h * bq)
        tl.store(p_o, bo.to(o.dtype.element_ty))

        p_q += stride_q_t
        p_k += stride_k_t
        p_v += stride_v_t
        p_g += stride_g_t
        p_beta += stride_beta_t
        p_o += stride_o_t

    # Final-state write-back.  With a constant ssm index all sequences race on
    # the same slot, so only the last sequence stores (last-write-wins order of
    # the sequential reference); without ssm each sequence owns its own slot.
    if (i_seq == LAST_SEQ) or (not USE_SSM):
        p_ht = (
            ht
            + (s1 * HT_STRIDE_S + i_hv * HT_STRIDE_HV).to(tl.int64)
            + i_v * K
            + offs
        )
        tl.store(p_ht, h.to(ht.dtype.element_ty))


def fused_recurrent_gated_delta_rule_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float,
    initial_state: torch.Tensor,
    inplace_final_state: bool = True,
    cu_seqlens: torch.LongTensor | None = None,
    ssm_state_indices: torch.Tensor | None = None,
    num_accepted_tokens: torch.Tensor | None = None,
    use_qk_l2norm_in_kernel: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    logger.debug("GEMS_KUNLUNXIN FUSED RECURRENT GATED DELTA RULE FWD")

    # Fast path (column-parallel Triton kernel).  The kernel requires:
    #  - K a power of two (BK = K, no tail masked lanes)
    #  - initial_state / final_state contiguous (state (S, HV, K, V) layout)
    #  - beta headwise-scalar only (shape (B, T, HV))
    #  - no speculative decoding (num_accepted_tokens)
    use_ssm = ssm_state_indices is not None
    use_triton = (
        (K := q.shape[-1]) & (K - 1) == 0
        and initial_state.is_contiguous()
        and beta.ndim == v.ndim - 1
        and num_accepted_tokens is None
        and inplace_final_state
    )
    if use_triton and use_ssm:
        # fast path stores the state only once (after the sequence loop), which is
        # exactly the per-token last-write semantics iff the ssm index is constant
        # over the whole batch (all final-state writes go to one slot per column).
        if not bool(torch.all(ssm_state_indices == ssm_state_indices[0]).cpu()):
            use_triton = False
    if not use_triton:
        return _fused_recurrent_gated_delta_rule_fwd_python(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            scale=scale,
            initial_state=initial_state,
            inplace_final_state=inplace_final_state,
            cu_seqlens=cu_seqlens,
            ssm_state_indices=ssm_state_indices,
            use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        )

    B, T, H, K = q.shape
    HV = v.shape[2]
    V = v.shape[3]

    N = B if cu_seqlens is None else len(cu_seqlens) - 1
    if cu_seqlens is None:
        cu_seqlens = torch.arange(0, N * T + 1, T, device=q.device, dtype=torch.long)

    # NOTE: torch.empty_like on this backend does not preserve non-contiguous
    # strides; allocate a plain contiguous output and address it by its own strides.
    output = torch.empty(v.shape, dtype=v.dtype, device=v.device)
    if inplace_final_state:
        # transposed clone: only the touched state slot is overwritten by the kernel
        h_scratch = initial_state.transpose(2, 3).contiguous()
    else:
        h_scratch = torch.zeros(
            T, HV, V, K, dtype=initial_state.dtype, device=initial_state.device
        )
    final_state = initial_state
    if ssm_state_indices is None:
        ssm_state_indices = torch.zeros(1, device=q.device, dtype=torch.long)

    BK = triton.next_power_of_2(K)
    grid = (N, HV, V)
    _fused_recurrent_gated_delta_rule_fwd_kernel[grid](
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        o=output,
        h0=initial_state,
        ht=h_scratch,
        cu_seqlens=cu_seqlens,
        ssm_state_indices=ssm_state_indices,
        scale=scale,
        T=T,
        stride_q_t=q.stride(1),
        stride_q_h=q.stride(2),
        stride_q_k=q.stride(3),
        stride_k_t=k.stride(1),
        stride_k_h=k.stride(2),
        stride_k_k=k.stride(3),
        stride_v_t=v.stride(1),
        stride_v_hv=v.stride(2),
        stride_v_v=v.stride(3),
        stride_o_t=output.stride(1),
        stride_o_hv=output.stride(2),
        stride_o_v=output.stride(3),
        stride_g_t=g.stride(1),
        stride_g_hv=g.stride(2),
        stride_beta_t=beta.stride(1),
        stride_beta_hv=beta.stride(2),
        stride_cu=cu_seqlens.stride(0),
        stride_ssm=ssm_state_indices.stride(0),
        H=H,
        HV=HV,
        K=K,
        V=V,
        BK=BK,
        H0_STRIDE_S=initial_state.stride(0),
        H0_STRIDE_HV=initial_state.stride(1),
        HT_STRIDE_S=h_scratch.stride(0),
        HT_STRIDE_HV=h_scratch.stride(1),
        USE_CU=cu_seqlens is not None,
        USE_SSM=use_ssm,
        LAST_SEQ=N - 1,
        INPLACE=inplace_final_state,
        USE_L2=use_qk_l2norm_in_kernel,
        num_warps=1,
    )
    if inplace_final_state:
        initial_state.copy_(h_scratch.transpose(2, 3))
    else:
        final_state = h_scratch.transpose(2, 3).contiguous()
    return output, final_state
