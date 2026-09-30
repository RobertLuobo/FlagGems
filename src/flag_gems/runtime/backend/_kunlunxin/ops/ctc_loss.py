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

import sys as _sys

import torch
import triton
import triton.language as tl

import flag_gems.ops.ctc_loss  # noqa: F401  ensure canonical module is imported
from flag_gems.ops.ctc_loss import (  # noqa: F401
    _ctc_loss_init_grad_kernel,
    _debug_barrier,
    _logaddexp,
    _logaddexp3,
    ctc_loss,
)
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

_generic = _sys.modules["flag_gems.ops.ctc_loss"]


@triton.jit
def _load_masked(ptr, mask, other):
    return tl.where(mask, tl.load(ptr, mask=mask, other=other), other)


@libentry()
@triton.jit
def _ctc_loss_forward_kernel(
    log_probs,
    targets,
    input_lengths,
    target_lengths,
    target_offsets,
    neg_log_likelihood,
    log_alpha,
    T: tl.constexpr,
    N: tl.constexpr,
    C: tl.constexpr,
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
        target_base = tl.full((), 0, tl.int64)
        for prev_batch in tl.range(0, N):
            target_base += _load_masked(
                target_lengths + prev_batch, prev_batch < batch, 0
            )
        target_origin = target_base
        target_ptrs = targets + target_origin + target_safe_index
    else:
        target_origin = batch * MAX_TARGET
        target_ptrs = targets + target_origin + target_safe_index

    target_value = _load_masked(target_ptrs, target_mask, BLANK)
    labels = tl.where(is_blank_state, BLANK, target_value)

    t0_active = input_len > 0
    init_state = (states == 0) | ((states == 1) & (target_len > 0))
    init_logp = _load_masked(
        log_probs + batch * C + labels,
        init_state & stored_state & t0_active,
        0.0,
    ).to(tl.float32)
    alpha = tl.where(init_state & valid_state & t0_active, init_logp, -float("inf"))
    tl.store(
        log_alpha + batch * T * STATE_COUNT_MAX + states,
        alpha,
        mask=stored_state,
    )
    _debug_barrier()

    for t in tl.range(1, T):
        prev_base = log_alpha + batch * T * STATE_COUNT_MAX + (t - 1) * STATE_COUNT_MAX
        prev0 = _load_masked(prev_base + states, stored_state, -float("inf")).to(
            tl.float32
        )
        prev1 = _load_masked(
            prev_base + tl.where(states > 0, states - 1, 0),
            (states > 0) & stored_state,
            -float("inf"),
        ).to(tl.float32)
        prev2 = _load_masked(
            prev_base + tl.where(states > 1, states - 2, 0),
            (states > 1) & stored_state,
            -float("inf"),
        ).to(tl.float32)

        prev_target_index = tl.where(target_index > 0, target_index - 1, 0)
        prev_target_value = _load_masked(
            targets + target_origin + prev_target_index,
            target_mask & (target_index > 0),
            BLANK,
        )
        skip_allowed = (
            (~is_blank_state) & (target_index > 0) & (target_value != prev_target_value)
        )

        acc = _logaddexp3(prev0, prev1, prev2, skip_allowed)

        logp = _load_masked(
            log_probs + t * N * C + batch * C + labels,
            valid_state & (t < input_len),
            0.0,
        ).to(tl.float32)
        alpha = tl.where(valid_state & (t < input_len), acc + logp, -float("inf"))
        tl.store(
            log_alpha + batch * T * STATE_COUNT_MAX + t * STATE_COUNT_MAX + states,
            alpha,
            mask=stored_state,
        )
        _debug_barrier()

    if input_len <= 0:
        loss = tl.where(target_len == 0, 0.0, float("inf"))
    else:
        _debug_barrier()
        final_base = (
            log_alpha + batch * T * STATE_COUNT_MAX + (input_len - 1) * STATE_COUNT_MAX
        )
        last = tl.load(final_base + state_count - 1).to(tl.float32)
        prev_last = _load_masked(
            final_base + tl.where(target_len > 0, state_count - 2, 0),
            target_len > 0,
            -float("inf"),
        ).to(tl.float32)
        log_likelihood = _logaddexp(last, prev_last)
        loss = -log_likelihood

    tl.store(neg_log_likelihood + batch, loss)

@libentry()
@triton.jit
def _ctc_loss_forward_no_grad_kernel(
    log_probs,
    targets,
    input_lengths,
    target_lengths,
    target_offsets,
    neg_log_likelihood,
    scratch_alpha,
    T: tl.constexpr,
    N: tl.constexpr,
    C: tl.constexpr,
    MAX_TARGET: tl.constexpr,
    STATE_COUNT_MAX: tl.constexpr,
    BLANK: tl.constexpr,
    TARGET_1D: tl.constexpr,
    BLOCK_S: tl.constexpr,
):
    batch = tl.program_id(0)
    states = tl.arange(0, BLOCK_S)

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
        target_ptrs = targets + target_origin + target_safe_index
    else:
        target_origin = batch * MAX_TARGET
        target_ptrs = targets + target_origin + target_safe_index

    target_value = _load_masked(target_ptrs, target_mask, BLANK)
    labels = tl.where(is_blank_state, BLANK, target_value)

    input_len = tl.load(input_lengths + batch)
    init_state = (states == 0) | ((states == 1) & (target_len > 0))
    init_logp = _load_masked(
        log_probs + batch * C + labels,
        init_state & stored_state & (input_len > 0),
        0.0,
    ).to(tl.float32)
    alpha = tl.where(
        init_state & valid_state & (input_len > 0), init_logp, -float("inf")
    )
    scratch_batch = scratch_alpha + batch * 2 * STATE_COUNT_MAX
    tl.store(scratch_batch + states, alpha, mask=stored_state)
    _debug_barrier()

    for t in tl.range(1, T):
        prev_base = scratch_batch + ((t - 1) % 2) * STATE_COUNT_MAX
        cur_base = scratch_batch + (t % 2) * STATE_COUNT_MAX
        prev0 = _load_masked(prev_base + states, stored_state, -float("inf")).to(
            tl.float32
        )
        prev1 = _load_masked(
            prev_base + tl.where(states > 0, states - 1, 0),
            (states > 0) & stored_state,
            -float("inf"),
        ).to(tl.float32)
        prev2 = _load_masked(
            prev_base + tl.where(states > 1, states - 2, 0),
            (states > 1) & stored_state,
            -float("inf"),
        ).to(tl.float32)

        prev_target_index = tl.where(target_index > 0, target_index - 1, 0)
        prev_target_value = _load_masked(
            targets + target_origin + prev_target_index,
            target_mask & (target_index > 0),
            BLANK,
        )
        skip_allowed = (
            (~is_blank_state) & (target_index > 0) & (target_value != prev_target_value)
        )

        acc = _logaddexp3(prev0, prev1, prev2, skip_allowed)
        logp = _load_masked(
            log_probs + t * N * C + batch * C + labels,
            valid_state & (t < input_len),
            0.0,
        ).to(tl.float32)
        alpha = tl.where(valid_state & (t < input_len), acc + logp, -float("inf"))
        cur_prev = _load_masked(cur_base + states, stored_state, -float("inf")).to(
            tl.float32
        )
        store_val = tl.where(t < input_len, alpha, cur_prev)
        tl.store(cur_base + states, store_val, mask=stored_state)
        _debug_barrier()

    if input_len <= 0:
        loss = tl.where(target_len == 0, 0.0, float("inf"))
    else:
        _debug_barrier()
        final_base = scratch_batch + ((input_len - 1) % 2) * STATE_COUNT_MAX
        last = tl.load(final_base + state_count - 1).to(tl.float32)
        prev_last = _load_masked(
            final_base + tl.where(target_len > 0, state_count - 2, 0),
            target_len > 0,
            -float("inf"),
        ).to(tl.float32)
        loss = -_logaddexp(last, prev_last)

    tl.store(neg_log_likelihood + batch, loss)

@libentry()
@triton.jit
def _ctc_loss_forward_full_length_reduce_kernel(
    log_probs,
    targets,
    target_lengths,
    target_offsets,
    contrib,
    scratch_alpha,
    T: tl.constexpr,
    N: tl.constexpr,
    C: tl.constexpr,
    MAX_TARGET: tl.constexpr,
    STATE_COUNT_MAX: tl.constexpr,
    BLANK: tl.constexpr,
    TARGET_1D: tl.constexpr,
    REDUCTION: tl.constexpr,
    BLOCK_S: tl.constexpr,
):
    batch = tl.program_id(0)
    states = tl.arange(0, BLOCK_S)

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
        target_ptrs = targets + target_origin + target_safe_index
    else:
        target_origin = batch * MAX_TARGET
        target_ptrs = targets + target_origin + target_safe_index

    target_value = _load_masked(target_ptrs, target_mask, BLANK)
    labels = tl.where(is_blank_state, BLANK, target_value)

    init_state = (states == 0) | ((states == 1) & (target_len > 0))
    init_logp = _load_masked(
        log_probs + batch * C + labels,
        init_state & stored_state,
        0.0,
    ).to(tl.float32)
    alpha = tl.where(init_state & valid_state, init_logp, -float("inf"))
    scratch_batch = scratch_alpha + batch * 2 * STATE_COUNT_MAX
    tl.store(scratch_batch + states, alpha, mask=stored_state)
    _debug_barrier()

    for t in tl.range(1, T):
        prev_base = scratch_batch + ((t - 1) % 2) * STATE_COUNT_MAX
        cur_base = scratch_batch + (t % 2) * STATE_COUNT_MAX
        prev0 = _load_masked(prev_base + states, stored_state, -float("inf")).to(
            tl.float32
        )
        prev1 = _load_masked(
            prev_base + tl.where(states > 0, states - 1, 0),
            (states > 0) & stored_state,
            -float("inf"),
        ).to(tl.float32)
        prev2 = _load_masked(
            prev_base + tl.where(states > 1, states - 2, 0),
            (states > 1) & stored_state,
            -float("inf"),
        ).to(tl.float32)

        prev_target_index = tl.where(target_index > 0, target_index - 1, 0)
        prev_target_value = _load_masked(
            targets + target_origin + prev_target_index,
            target_mask & (target_index > 0),
            BLANK,
        )
        skip_allowed = (
            (~is_blank_state) & (target_index > 0) & (target_value != prev_target_value)
        )

        acc = _logaddexp3(prev0, prev1, prev2, skip_allowed)
        logp = _load_masked(
            log_probs + t * N * C + batch * C + labels,
            valid_state,
            0.0,
        ).to(tl.float32)
        alpha = tl.where(valid_state, acc + logp, -float("inf"))
        tl.store(cur_base + states, alpha, mask=stored_state)
        _debug_barrier()

    if T <= 0:
        loss = tl.where(target_len == 0, 0.0, float("inf"))
    else:
        _debug_barrier()
        final_base = scratch_batch + ((T - 1) % 2) * STATE_COUNT_MAX
        last = tl.load(final_base + state_count - 1).to(tl.float32)
        prev_last = _load_masked(
            final_base + tl.where(target_len > 0, state_count - 2, 0),
            target_len > 0,
            -float("inf"),
        ).to(tl.float32)
        loss = -_logaddexp(last, prev_last)

    if REDUCTION == 1:
        loss = loss / tl.maximum(target_len, 1).to(tl.float32) / N
    tl.store(contrib + batch, loss)

@libentry()
@triton.jit
def _ctc_loss_backward_beta_kernel(
    log_probs,
    targets,
    input_lengths,
    target_lengths,
    target_offsets,
    neg_log_likelihood,
    log_alpha,
    grad_output,
    scratch_beta,
    post_store,
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
        target_ptrs = targets + target_origin + target_safe_index
    else:
        target_origin = batch * MAX_TARGET
        target_ptrs = targets + target_origin + target_safe_index

    target_value = _load_masked(target_ptrs, target_mask, BLANK)
    labels = tl.where(is_blank_state, BLANK, target_value)

    state1 = states + 1
    is_blank_state1 = (state1 % 2) == 0
    target_index1 = (state1 - 1) // 2
    target_mask1 = (target_index1 >= 0) & (target_index1 < target_len)
    target_safe_index1 = tl.where(target_mask1, target_index1, 0)
    target_ptrs1 = targets + target_origin + target_safe_index1
    target_value1 = _load_masked(target_ptrs1, target_mask1, BLANK)
    labels1 = tl.where(is_blank_state1, BLANK, target_value1)

    state2 = states + 2
    target_index2 = (state2 - 1) // 2
    target_mask2 = (target_index2 >= 0) & (target_index2 < target_len)
    target_safe_index2 = tl.where(target_mask2, target_index2, 0)
    target_ptrs2 = targets + target_origin + target_safe_index2
    target_value2 = _load_masked(target_ptrs2, target_mask2, BLANK)
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

    beta_init = tl.where(
        ((states == state_count - 1) | ((states == state_count - 2) & (target_len > 0)))
        & valid_state
        & (input_len > 0),
        0.0,
        -float("inf"),
    )
    scratch_batch = scratch_beta + batch * 2 * STATE_COUNT_MAX
    tl.store(scratch_batch + states, beta_init, mask=stored_state)
    _debug_barrier()
    log_likelihood = tl.where(scale != 0.0, -nll, 0.0)

    for step in tl.range(0, T):
        t = input_len - 1 - step
        active = t >= 0
        safe_t = tl.where(active, t, 0)
        beta_base = scratch_batch + (step % 2) * STATE_COUNT_MAX
        next_beta_base = scratch_batch + ((step + 1) % 2) * STATE_COUNT_MAX
        beta = _load_masked(beta_base + states, stored_state, -float("inf")).to(
            tl.float32
        )

        alpha_t = _load_masked(
            log_alpha + batch * T * STATE_COUNT_MAX + safe_t * STATE_COUNT_MAX + states,
            active & stored_state,
            -float("inf"),
        ).to(tl.float32)
        log_post = alpha_t + beta - log_likelihood
        posterior = tl.where(
            active & valid_state & (scale != 0.0),
            tl.exp(log_post),
            0.0,
        )
        contrib = -scale * posterior
        cur_row = tl.load(
            post_store + batch * T * STATE_COUNT_MAX + safe_t * STATE_COUNT_MAX + states,
            mask=stored_state,
            other=0.0,
        )
        active_vec = stored_state & active
        store_val = tl.where(active_vec, contrib, cur_row)
        tl.store(
            post_store + batch * T * STATE_COUNT_MAX + safe_t * STATE_COUNT_MAX + states,
            store_val,
            mask=stored_state,
        )

        stay = beta + _load_masked(
            log_probs + safe_t * N * C + batch * C + labels,
            active & valid_state,
            -float("inf"),
        ).to(tl.float32)
        next1 = _load_masked(
            beta_base + states + 1,
            (states + 1 < state_count) & stored_state,
            -float("inf"),
        ).to(tl.float32) + _load_masked(
            log_probs + safe_t * N * C + batch * C + labels1,
            active & (states + 1 < state_count) & stored_state,
            -float("inf"),
        ).to(
            tl.float32
        )
        skip_allowed = (
            (~is_blank_state)
            & (states + 2 < state_count)
            & (target_value != target_value2)
        )
        next2 = _load_masked(
            beta_base + states + 2,
            (states + 2 < state_count) & stored_state,
            -float("inf"),
        ).to(tl.float32) + _load_masked(
            log_probs + safe_t * N * C + batch * C + labels2,
            active & skip_allowed & stored_state,
            -float("inf"),
        ).to(
            tl.float32
        )

        beta_next = _logaddexp3(stay, next1, next2, skip_allowed)
        tl.store(
            next_beta_base + states,
            tl.where(active, beta_next, -float("inf")),
            mask=stored_state,
        )
        _debug_barrier()


@libentry()
@triton.jit
def _ctc_loss_backward_scatter_kernel(
    targets,
    input_lengths,
    target_lengths,
    target_offsets,
    grad_input,
    post_store,
    T: tl.constexpr,
    N: tl.constexpr,
    C: tl.constexpr,
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
        target_ptrs = targets + target_origin + target_safe_index
    else:
        target_origin = batch * MAX_TARGET
        target_ptrs = targets + target_origin + target_safe_index

    target_value = _load_masked(target_ptrs, target_mask, BLANK)
    labels = tl.where(is_blank_state, BLANK, target_value)

    for tt in tl.static_range(0, T):
        active = tt < input_len
        val = _load_masked(
            post_store + batch * T * STATE_COUNT_MAX + tt * STATE_COUNT_MAX + states,
            active & stored_state,
            0.0,
        )
        for c in tl.static_range(0, C):
            sel = (labels == c) & valid_state & stored_state
            csum = tl.sum(tl.where(sel, val, 0.0))
            addr = grad_input + tt * N * C + batch * C + c
            cur = tl.load(addr)
            tl.store(addr, cur + csum)
        _debug_barrier()


def _ctc_loss_backward(ctx, grad_output):
    (
        work_log_probs,
        work_targets,
        work_input_lengths,
        work_target_lengths,
        work_target_offsets,
        neg_log_likelihood,
        log_alpha,
    ) = ctx.saved_tensors

    grad_output = grad_output.contiguous()

    grad_log_probs = torch.empty_like(work_log_probs)
    total = work_log_probs.numel()
    block = 256
    with torch_device_fn.device(work_log_probs.device):
        _ctc_loss_init_grad_kernel[(triton.cdiv(total, block),)](
            work_log_probs,
            work_input_lengths,
            work_target_lengths,
            neg_log_likelihood,
            grad_output,
            grad_log_probs,
            total,
            work_log_probs.shape[0],
            ctx.batch_size,
            work_log_probs.shape[2],
            ctx.reduction,
            ctx.zero_infinity,
            block,
        )

        scratch_beta = torch.empty(
            (ctx.batch_size, 2, ctx.state_count_max),
            dtype=torch.float32,
            device=work_log_probs.device,
        )
        post_store = torch.zeros(
            (ctx.batch_size, work_log_probs.shape[0], ctx.state_count_max),
            dtype=torch.float32,
            device=work_log_probs.device,
        )
        block_s = triton.next_power_of_2(ctx.state_count_max)
        _ctc_loss_backward_beta_kernel[(ctx.batch_size,)](
            work_log_probs,
            work_targets,
            work_input_lengths,
            work_target_lengths,
            work_target_offsets,
            neg_log_likelihood,
            log_alpha,
            grad_output,
            scratch_beta,
            post_store,
            work_log_probs.shape[0],
            ctx.batch_size,
            work_log_probs.shape[2],
            ctx.max_target,
            ctx.state_count_max,
            ctx.blank,
            ctx.target_1d,
            ctx.reduction,
            ctx.zero_infinity,
            block_s,
        )
        _ctc_loss_backward_scatter_kernel[(ctx.batch_size,)](
            work_targets,
            work_input_lengths,
            work_target_lengths,
            work_target_offsets,
            grad_log_probs,
            post_store,
            work_log_probs.shape[0],
            ctx.batch_size,
            work_log_probs.shape[2],
            ctx.max_target,
            ctx.state_count_max,
            ctx.blank,
            ctx.target_1d,
            block_s,
        )

    if ctx.unbatched:
        grad_log_probs = grad_log_probs.squeeze(1)
    if grad_log_probs.dtype != ctx.original_dtype:
        grad_log_probs = grad_log_probs.to(ctx.original_dtype)

    return grad_log_probs, None, None, None, None, None, None


_generic._ctc_loss_forward_kernel = _ctc_loss_forward_kernel
_generic._ctc_loss_forward_no_grad_kernel = _ctc_loss_forward_no_grad_kernel
_generic._ctc_loss_forward_full_length_reduce_kernel = (
    _ctc_loss_forward_full_length_reduce_kernel
)
_generic._ctc_loss_backward_beta_kernel = _ctc_loss_backward_beta_kernel
_generic._ctc_loss_backward_scatter_kernel = _ctc_loss_backward_scatter_kernel
_generic.CtcLossFunction.backward = staticmethod(_ctc_loss_backward)

__all__ = ["ctc_loss"]
