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
from typing import Optional

import torch
import triton
import triton.language as tl

from flag_gems.ops._ctc_loss_backward import _prepare_backward
from flag_gems.ops.ctc_loss import (
    _REDUCTION_NONE,
    _ctc_loss_init_grad_kernel,
    _logaddexp3,
)
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)


@libentry()
@triton.jit
def _ctc_loss_backward_init_kernel(
    targets,
    input_lengths,
    target_lengths,
    target_offsets,
    scratch_beta,
    scratch_labels,
    T: tl.constexpr,
    MAX_TARGET: tl.constexpr,
    STATE_COUNT_MAX: tl.constexpr,
    BLANK: tl.constexpr,
    TARGET_1D: tl.constexpr,
    BLOCK_S: tl.constexpr,
):
    batch = tl.program_id(0)
    states = tl.arange(0, BLOCK_S)

    input_len = tl.load(input_lengths + batch)
    target_len = tl.load(target_lengths + batch)
    state_count = target_len * 2 + 1
    valid_state = states < state_count
    stored_state = states < STATE_COUNT_MAX

    is_blank_state = (states % 2) == 0
    target_index = (states - 1) // 2
    target_mask = (target_index >= 0) & (target_index < target_len)
    target_safe_index = tl.where(target_mask, target_index, 0)

    if TARGET_1D:
        target_origin = tl.load(target_offsets + batch)
    else:
        target_origin = batch * MAX_TARGET
    target_ptrs = targets + target_origin + target_safe_index
    target_value = tl.load(target_ptrs, mask=target_mask, other=BLANK)
    labels = tl.where(is_blank_state, BLANK, target_value)
    tl.store(scratch_labels + batch * BLOCK_S + states, labels, mask=stored_state)

    beta_init = tl.where(
        ((states == state_count - 1) | ((states == state_count - 2) & (target_len > 0)))
        & valid_state
        & (input_len > 0),
        0.0,
        -float("inf"),
    )
    # scratch_beta is laid out as [batch, T + 1, STATE_COUNT_MAX]: slot 0 holds
    # the terminal-time beta, and step s writes slot s + 1. Every slot is thus
    # written by exactly one launch and read by exactly one later launch, so the
    # only ordering needed is the implicit global flush at each kernel boundary.
    scratch_batch = scratch_beta + batch * (T + 1) * STATE_COUNT_MAX
    outer_safe_states = tl.where(stored_state, states, 0)
    tl.store(scratch_batch + outer_safe_states, beta_init, mask=stored_state)


@libentry()
@triton.jit
def _ctc_loss_backward_step_kernel(
    log_probs,
    targets,
    input_lengths,
    target_lengths,
    target_offsets,
    neg_log_likelihood,
    grad_output,
    log_alpha,
    scratch_beta,
    scratch_post,
    step,
    T: tl.constexpr,
    N: tl.constexpr,
    C: tl.constexpr,
    MAX_TARGET: tl.constexpr,
    STATE_COUNT_MAX: tl.constexpr,
    BLANK: tl.constexpr,
    TARGET_1D: tl.constexpr,
    REDUCTION: tl.constexpr,
    ZERO_INFINITY: tl.constexpr,
    BLOCK_S: tl.constexpr,
):
    batch = tl.program_id(0)
    states = tl.arange(0, BLOCK_S)

    input_len = tl.load(input_lengths + batch)
    target_len = tl.load(target_lengths + batch)
    nll = tl.load(neg_log_likelihood + batch).to(tl.float32)
    state_count = target_len * 2 + 1
    valid_state = states < state_count
    stored_state = states < STATE_COUNT_MAX

    is_blank_state = (states % 2) == 0
    target_index = (states - 1) // 2
    target_mask = (target_index >= 0) & (target_index < target_len)
    target_safe_index = tl.where(target_mask, target_index, 0)

    if TARGET_1D:
        target_origin = tl.load(target_offsets + batch)
    else:
        target_origin = batch * MAX_TARGET
    target_ptrs = targets + target_origin + target_safe_index
    target_value = tl.load(target_ptrs, mask=target_mask, other=BLANK)
    labels = tl.where(is_blank_state, BLANK, target_value)

    state1 = states + 1
    is_blank_state1 = (state1 % 2) == 0
    target_index1 = (state1 - 1) // 2
    target_mask1 = (target_index1 >= 0) & (target_index1 < target_len)
    target_safe_index1 = tl.where(target_mask1, target_index1, 0)
    target_ptrs1 = targets + target_origin + target_safe_index1
    target_value1 = tl.load(target_ptrs1, mask=target_mask1, other=BLANK)
    labels1 = tl.where(is_blank_state1, BLANK, target_value1)

    state2 = states + 2
    target_index2 = (state2 - 1) // 2
    target_mask2 = (target_index2 >= 0) & (target_index2 < target_len)
    target_safe_index2 = tl.where(target_mask2, target_index2, 0)
    target_ptrs2 = targets + target_origin + target_safe_index2
    target_value2 = tl.load(target_ptrs2, mask=target_mask2, other=BLANK)
    labels2 = target_value2

    if REDUCTION == 0:
        scale = tl.load(grad_output + batch).to(tl.float32)
    else:
        scale = tl.load(grad_output).to(tl.float32)
        if REDUCTION == 1:
            denom = tl.maximum(target_len, 1).to(tl.float32) * N
            scale = scale / denom

    if ZERO_INFINITY:
        scale = tl.where(nll == float("inf"), 0.0, scale)

    log_likelihood = tl.where(scale != 0.0, -nll, 0.0)

    scratch_batch = scratch_beta + batch * (T + 1) * STATE_COUNT_MAX
    t = input_len - 1 - step
    active = t >= 0
    safe_t = tl.where(active, t, 0)
    beta_base = scratch_batch + step * STATE_COUNT_MAX
    next_beta_base = scratch_batch + (step + 1) * STATE_COUNT_MAX
    safe_states = tl.where(stored_state, states, 0)

    beta = tl.load(
        beta_base + safe_states, mask=stored_state, other=-float("inf")
    ).to(tl.float32)
    alpha_t = tl.load(
        log_alpha
        + batch * T * STATE_COUNT_MAX
        + safe_t * STATE_COUNT_MAX
        + safe_states,
        mask=active & stored_state,
        other=-float("inf"),
    ).to(tl.float32)
    log_post = alpha_t + beta - log_likelihood
    posterior = tl.where(
        active & valid_state & (scale != 0.0),
        tl.exp(log_post),
        0.0,
    )
    # XPU3: tl.atomic_add with duplicate-address lanes inside one program does
    # not accumulate (blank/repeated labels alias the same address across state
    # lanes). Stash the masked per-state contribution; the class reduction runs
    # in the separate scatter kernel.
    contrib = tl.where(active & valid_state & stored_state, -scale * posterior, 0.0)
    tl.store(
        scratch_post + batch * T * BLOCK_S + safe_t * BLOCK_S + states,
        contrib,
        mask=active & stored_state,
    )

    stay = beta + tl.load(
        log_probs + safe_t * N * C + batch * C + labels,
        mask=active & valid_state,
        other=-float("inf"),
    ).to(tl.float32)
    state1_in = (states + 1 < state_count) & stored_state
    beta1 = tl.where(
        state1_in,
        tl.load(beta_base + tl.where(state1_in, states + 1, 0)).to(tl.float32),
        -float("inf"),
    )
    logp1 = tl.where(
        active & state1_in,
        tl.load(log_probs + safe_t * N * C + batch * C + labels1).to(tl.float32),
        -float("inf"),
    )
    next1 = beta1 + logp1
    skip_allowed = (
        (~is_blank_state)
        & (states + 2 < state_count)
        & (target_value != target_value2)
    )
    state2_in = (states + 2 < state_count) & stored_state
    beta2 = tl.where(
        state2_in,
        tl.load(beta_base + tl.where(state2_in, states + 2, 0)).to(tl.float32),
        -float("inf"),
    )
    logp2 = tl.where(
        active & skip_allowed & stored_state,
        tl.load(log_probs + safe_t * N * C + batch * C + labels2).to(tl.float32),
        -float("inf"),
    )
    next2 = beta2 + logp2

    beta_next = _logaddexp3(stay, next1, next2, skip_allowed)
    tl.store(
        next_beta_base + safe_states,
        tl.where(active, beta_next, -float("inf")),
        mask=stored_state,
    )


@libentry()
@triton.jit
def _ctc_loss_backward_scatter_kernel(
    scratch_post,
    scratch_labels,
    grad_input,
    T: tl.constexpr,
    N: tl.constexpr,
    C: tl.constexpr,
    BLOCK_S: tl.constexpr,
):
    batch = tl.program_id(0)
    t = tl.program_id(1)
    states = tl.arange(0, BLOCK_S)

    labels = tl.load(scratch_labels + batch * BLOCK_S + states)
    post_vec = tl.load(
        scratch_post + batch * T * BLOCK_S + t * BLOCK_S + states
    ).to(tl.float32)

    # Single (non-nested) loop over classes with one reduction each; this is the
    # scatter-add the recurrence deferred. grad_input already holds
    # exp(logp)*scale from the init kernel and this (batch, t) is written by
    # exactly one program, so a plain read-modify-write is race-free.
    for c in tl.range(0, C):
        csum = tl.sum(tl.where(labels == c, post_vec, 0.0))
        ptr = grad_input + t * N * C + batch * C + c
        cur = tl.load(ptr).to(tl.float32)
        tl.store(ptr, cur + csum)


def _ctc_loss_backward(
    grad: torch.Tensor,
    log_probs: torch.Tensor,
    targets: torch.Tensor,
    input_lengths: torch.Tensor,
    target_lengths: torch.Tensor,
    neg_log_likelihood: torch.Tensor,
    log_alpha: torch.Tensor,
    blank: int = 0,
    zero_infinity: bool = False,
):
    logger.debug("GEMS_KUNLUNXIN _CTC_LOSS_BACKWARD")

    setup = _prepare_backward(
        grad,
        log_probs,
        targets,
        input_lengths,
        target_lengths,
        neg_log_likelihood,
        log_alpha,
        blank,
    )

    batch_size = setup.batch_size
    original_dtype = setup.original_dtype
    unbatched = setup.unbatched
    state_count_max = setup.state_count_max
    work_grad = setup.work_grad
    work_log_probs = setup.work_log_probs
    work_targets = setup.work_targets
    work_input_lengths = setup.work_input_lengths
    work_target_lengths = setup.work_target_lengths
    work_target_offsets = setup.work_target_offsets
    work_neg_log_likelihood = setup.work_neg_log_likelihood
    work_log_alpha = setup.work_log_alpha
    target_stride = setup.target_stride
    target_1d = setup.target_1d
    block_s = setup.block_s

    grad_log_probs = torch.empty_like(work_log_probs)
    total = work_log_probs.numel()
    block = 256
    T = work_log_probs.shape[0]
    C = work_log_probs.shape[2]

    with torch_device_fn.device(log_probs.device):
        _ctc_loss_init_grad_kernel[(triton.cdiv(total, block),)](
            work_log_probs,
            work_input_lengths,
            work_target_lengths,
            work_neg_log_likelihood,
            work_grad,
            grad_log_probs,
            total,
            T,
            batch_size,
            C,
            _REDUCTION_NONE,
            zero_infinity,
            block,
        )

        scratch_beta = torch.empty(
            (batch_size, T + 1, state_count_max),
            dtype=torch.float32,
            device=log_probs.device,
        )
        scratch_post = torch.zeros(
            (batch_size, T, block_s),
            dtype=torch.float32,
            device=log_probs.device,
        )
        scratch_labels = torch.zeros(
            (batch_size, block_s),
            dtype=torch.int32,
            device=log_probs.device,
        )

        _ctc_loss_backward_init_kernel[(batch_size,)](
            work_targets,
            work_input_lengths,
            work_target_lengths,
            work_target_offsets,
            scratch_beta,
            scratch_labels,
            T,
            target_stride,
            state_count_max,
            blank,
            target_1d,
            block_s,
        )

        # XPU3: the beta recurrence exchanges data across state lanes through
        # global scratch (each state reads its neighbours' previous-step betas).
        # A device barrier inside a single kernel does not reliably make those
        # cross-lane global writes visible to the next iteration, so each
        # recurrence step is its own launch -- the kernel boundary is the only
        # dependable global sync -- and every scratch slot is written once and
        # read once.
        for step in range(T):
            _ctc_loss_backward_step_kernel[(batch_size,)](
                work_log_probs,
                work_targets,
                work_input_lengths,
                work_target_lengths,
                work_target_offsets,
                work_neg_log_likelihood,
                work_grad,
                work_log_alpha,
                scratch_beta,
                scratch_post,
                step,
                T,
                batch_size,
                C,
                target_stride,
                state_count_max,
                blank,
                target_1d,
                _REDUCTION_NONE,
                zero_infinity,
                block_s,
            )

        if T > 0:
            _ctc_loss_backward_scatter_kernel[(batch_size, T)](
                scratch_post,
                scratch_labels,
                grad_log_probs,
                T,
                batch_size,
                C,
                block_s,
            )

    if unbatched:
        grad_log_probs = grad_log_probs.squeeze(1)
    if grad_log_probs.dtype != original_dtype:
        grad_log_probs = grad_log_probs.to(original_dtype)

    return grad_log_probs


def _ctc_loss_backward_out(
    grad: torch.Tensor,
    log_probs: torch.Tensor,
    targets: torch.Tensor,
    input_lengths: torch.Tensor,
    target_lengths: torch.Tensor,
    neg_log_likelihood: torch.Tensor,
    log_alpha: torch.Tensor,
    blank: int = 0,
    zero_infinity: bool = False,
    *,
    out: Optional[torch.Tensor] = None,
):
    logger.debug("GEMS_KUNLUNXIN _CTC_LOSS_BACKWARD_OUT")

    grad_log_probs = _ctc_loss_backward(
        grad,
        log_probs,
        targets,
        input_lengths,
        target_lengths,
        neg_log_likelihood,
        log_alpha,
        blank,
        zero_infinity,
    )

    if out is not None:
        if out.shape != grad_log_probs.shape:
            out.resize_(grad_log_probs.shape)
        out.copy_(grad_log_probs)
    else:
        out = grad_log_probs

    return out
