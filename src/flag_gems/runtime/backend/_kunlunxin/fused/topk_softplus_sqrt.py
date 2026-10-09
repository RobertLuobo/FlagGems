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

"""TopK Softplus-Sqrt gating kernel for Kunlunxin XPU.

XPU overlay of the generic fused op. The generic kernels run the ``topk`` loop
with ``tl.static_range`` and mutate the per-row ``scores`` vector in place; at
``BLOCK_E >= 256`` the fully-unrolled wide reductions overflow
``TritonXPUUnrollControl``. Here the loop is a dynamic ``scf.for`` (``tl.range``)
and ``scores`` stays loop-invariant: the k-th winner is found by a strictly
decreasing ``(score, index)`` cursor carried as scalars, each result written
with a per-k scalar store so no wide tensor is carried across iterations. The
renormalize scale is applied at store time (a first scalar-only pass sums the
weights) to avoid an XPU load-after-store round trip. The hash kernel is fully
scalar: ``eidx`` is read from the table per k and the single gating element is
loaded directly, so no ``BLOCK_E``/``BLOCK_K`` vector is materialized.
"""

import logging

import triton
import triton.language as tl

logger = logging.getLogger(__name__)


@triton.jit
def _fused_topk_kernel(
    gating_ptr,
    topk_weights_ptr,
    topk_indices_ptr,
    token_expert_indices_ptr,
    e_score_correction_bias_ptr,
    num_tokens,
    num_experts: tl.constexpr,
    topk: tl.constexpr,
    renormalize: tl.constexpr,
    routed_scaling_factor,
    HAS_BIAS: tl.constexpr,
    BLOCK_E: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)

    expert_offsets = tl.arange(0, BLOCK_E)
    emask = expert_offsets < num_experts

    row_base = pid * num_experts
    x = tl.load(gating_ptr + row_base + expert_offsets, mask=emask, other=0.0).to(
        tl.float32
    )

    x = tl.where(x > 20.0, x, tl.log(1.0 + tl.exp(x)))
    raw = tl.sqrt(x)

    if HAS_BIAS:
        bias = tl.load(
            e_score_correction_bias_ptr + expert_offsets, mask=emask, other=0.0
        ).to(tl.float32)
        scores = raw + bias
    else:
        scores = raw
    scores = tl.where(emask, scores, -float("inf"))

    out_base = pid * topk

    scale = routed_scaling_factor
    if renormalize:
        weight_sum = 0.0
        prev_score = float("inf")
        prev_idx = -1
        for k_idx in tl.range(topk):
            eligible = (scores < prev_score) | (
                (scores == prev_score) & (expert_offsets > prev_idx)
            )
            masked = tl.where(eligible, scores, -float("inf"))
            max_score = tl.max(masked, axis=0)
            is_max = eligible & (masked == max_score)
            match_priority = tl.where(is_max, BLOCK_E - expert_offsets, 0)
            best_slot = BLOCK_E - tl.max(match_priority, axis=0)
            if HAS_BIAS:
                w = max_score - tl.load(e_score_correction_bias_ptr + best_slot)
            else:
                w = max_score
            weight_sum += w
            prev_score = max_score
            prev_idx = best_slot
        scale = routed_scaling_factor / tl.where(weight_sum > 0.0, weight_sum, 1.0)

    prev_score = float("inf")
    prev_idx = -1
    for k_idx in tl.range(topk):
        eligible = (scores < prev_score) | (
            (scores == prev_score) & (expert_offsets > prev_idx)
        )
        masked = tl.where(eligible, scores, -float("inf"))
        max_score = tl.max(masked, axis=0)
        is_max = eligible & (masked == max_score)
        match_priority = tl.where(is_max, BLOCK_E - expert_offsets, 0)
        best_slot = BLOCK_E - tl.max(match_priority, axis=0)
        eidx = best_slot.to(tl.int32)

        if HAS_BIAS:
            w = max_score - tl.load(e_score_correction_bias_ptr + eidx)
        else:
            w = max_score

        out_off = out_base + k_idx
        tl.store(topk_weights_ptr + out_off, w * scale)
        tl.store(topk_indices_ptr + out_off, eidx)
        tl.store(token_expert_indices_ptr + out_off, out_off.to(tl.int32))

        prev_score = max_score
        prev_idx = best_slot


@triton.jit
def _hash_kernel(
    gating_ptr,
    topk_weights_ptr,
    topk_indices_ptr,
    token_expert_indices_ptr,
    e_score_correction_bias_ptr,
    input_tokens_ptr,
    hash_indices_table_ptr,
    num_tokens,
    num_experts: tl.constexpr,
    topk: tl.constexpr,
    renormalize: tl.constexpr,
    routed_scaling_factor,
    HAS_BIAS: tl.constexpr,
    BLOCK_E: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Hash mode: expert indices come from lookup table."""
    pid = tl.program_id(0)

    token_id = tl.load(input_tokens_ptr + pid)
    row_base = pid * num_experts
    tbl_base = token_id * topk
    out_base = pid * topk

    scale = routed_scaling_factor
    if renormalize:
        weight_sum = 0.0
        for k_idx in tl.range(topk):
            eidx = tl.load(hash_indices_table_ptr + tbl_base + k_idx)
            gval = tl.load(gating_ptr + row_base + eidx).to(tl.float32)
            sp = tl.where(gval > 20.0, gval, tl.log(1.0 + tl.exp(gval)))
            weight_sum += tl.sqrt(sp)
        scale = routed_scaling_factor / tl.where(weight_sum > 0.0, weight_sum, 1.0)

    for k_idx in tl.range(topk):
        eidx = tl.load(hash_indices_table_ptr + tbl_base + k_idx)
        gval = tl.load(gating_ptr + row_base + eidx).to(tl.float32)
        sp = tl.where(gval > 20.0, gval, tl.log(1.0 + tl.exp(gval)))
        w = tl.sqrt(sp)
        out_off = out_base + k_idx
        tl.store(topk_weights_ptr + out_off, w * scale)
        tl.store(topk_indices_ptr + out_off, eidx.to(tl.int32))
        tl.store(token_expert_indices_ptr + out_off, out_off.to(tl.int32))


def topk_softplus_sqrt(
    topk_weights,
    topk_indices,
    token_expert_indices,
    gating_output,
    renormalize,
    routed_scaling_factor,
    correction_bias=None,
    input_ids=None,
    tid2eid=None,
):
    """Fused topk + softplus + sqrt kernel for MoE gating (Kunlunxin overlay)."""
    logger.debug("GEMS_KUNLUNXIN TOPK_SOFTPLUS_SQRT")
    num_tokens, num_experts = gating_output.shape
    topk = topk_weights.shape[1]

    if num_tokens == 0:
        return

    BLOCK_E = triton.next_power_of_2(num_experts)
    BLOCK_K = triton.next_power_of_2(topk)
    grid = (num_tokens,)

    if input_ids is not None and tid2eid is not None:
        _hash_kernel[grid](
            gating_output,
            topk_weights,
            topk_indices,
            token_expert_indices,
            correction_bias if correction_bias is not None else gating_output,
            input_ids,
            tid2eid,
            num_tokens=num_tokens,
            num_experts=num_experts,
            topk=topk,
            renormalize=renormalize,
            routed_scaling_factor=routed_scaling_factor,
            HAS_BIAS=correction_bias is not None,
            BLOCK_E=BLOCK_E,
            BLOCK_K=BLOCK_K,
            num_warps=1,
            num_stages=1,
        )
        return

    _fused_topk_kernel[grid](
        gating_output,
        topk_weights,
        topk_indices,
        token_expert_indices,
        correction_bias if correction_bias is not None else gating_output,
        num_tokens=num_tokens,
        num_experts=num_experts,
        topk=topk,
        renormalize=renormalize,
        routed_scaling_factor=routed_scaling_factor,
        HAS_BIAS=correction_bias is not None,
        BLOCK_E=BLOCK_E,
        BLOCK_K=BLOCK_K,
        num_warps=1,
        num_stages=1,
    )
