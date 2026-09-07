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
