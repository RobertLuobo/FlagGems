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

import torch
import triton
import triton.language as tl

from flag_gems.ops.ctc_loss import (
    _REDUCTION_MEAN,
    _REDUCTION_NONE,
    _REDUCTION_SUM,
    _compute_dtype,
    _ctc_loss_init_grad_kernel,
    _is_integral_dtype,
    _length_stats,
    _lengths_to_tensor,
    _reduction_enum,
)
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)


@triton.jit
def _lae(a, b):
    m = tl.maximum(a, b)
    return tl.where(
        m == -float("inf"),
        -float("inf"),
        m + tl.log(1.0 + tl.exp(tl.minimum(a, b) - m)),
    )


@triton.jit
def _lae3(a, b, c, use_c):
    c = tl.where(use_c, c, -float("inf"))
    m = tl.maximum(tl.maximum(a, b), c)
    sm = tl.where(m == -float("inf"), 0.0, m)
    es = tl.exp(a - sm) + tl.exp(b - sm) + tl.exp(c - sm)
    return tl.where(m == -float("inf"), -float("inf"), m + tl.log(es))


@triton.jit
def _safe_shift(base, idx, keep, hi, other):
    # XPU masked loads ignore `other=`; clamp index in-bounds, load unmasked, select.
    safe = tl.minimum(tl.maximum(idx, 0), hi - 1)
    v = tl.load(base + safe).to(tl.float32)
    return tl.where(keep, v, other)
# PLACEHOLDER_KERNELS


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
    scratch,
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
    valid = states < state_count
    stored = states < STATE_COUNT_MAX
    is_blank = (states % 2) == 0
    tidx = (states - 1) // 2
    tmask = (tidx >= 0) & (tidx < target_len)
    tsafe = tl.where(tmask, tidx, 0)
    if TARGET_1D:
        torigin = tl.load(target_offsets + batch)
    else:
        torigin = batch * MAX_TARGET
    tval = tl.where(tmask, tl.load(targets + torigin + tsafe), BLANK)
    labels = tl.where(is_blank, BLANK, tval)
    ptidx = tl.where(tidx > 0, tidx - 1, 0)
    ptval = tl.where(tmask & (tidx > 0), tl.load(targets + torigin + ptidx), BLANK)
    skip = (~is_blank) & (tidx > 0) & (tval != ptval)

    t0 = input_len > 0
    init_state = (states == 0) | ((states == 1) & (target_len > 0))
    il = tl.where(
        init_state & stored & t0, tl.load(log_probs + batch * C + labels), 0.0
    ).to(tl.float32)
    alpha = tl.where(init_state & valid & t0, il, -float("inf"))
    sbb = scratch + batch * STATE_COUNT_MAX
    tl.store(log_alpha + batch * T * STATE_COUNT_MAX + states, alpha, mask=stored)
    for t in tl.range(1, T, loop_unroll_factor=1):
        tl.store(sbb + states, alpha, mask=stored)
        tl.debug_barrier()
        prev1 = _safe_shift(sbb, states - 1, (states > 0) & stored, STATE_COUNT_MAX, -float("inf"))
        prev2 = _safe_shift(sbb, states - 2, (states > 1) & stored, STATE_COUNT_MAX, -float("inf"))
        tl.debug_barrier()
        acc = _lae3(alpha, prev1, prev2, skip)
        logp = tl.where(
            valid & (t < input_len),
            tl.load(log_probs + t * N * C + batch * C + labels),
            0.0,
        ).to(tl.float32)
        newa = tl.where(valid & (t < input_len), acc + logp, -float("inf"))
        tl.store(
            log_alpha + batch * T * STATE_COUNT_MAX + t * STATE_COUNT_MAX + states,
            newa,
            mask=stored,
        )
        alpha = tl.where(valid & (t < input_len), acc + logp, alpha)

    if input_len <= 0:
        loss = tl.where(target_len == 0, 0.0, float("inf"))
    else:
        tl.store(sbb + states, alpha, mask=stored)
        tl.debug_barrier()
        last = tl.load(sbb + state_count - 1).to(tl.float32)
        pl = _safe_shift(
            sbb, tl.where(target_len > 0, state_count - 2, 0), target_len > 0,
            STATE_COUNT_MAX, -float("inf"),
        )
        loss = -_lae(last, pl)
    tl.store(neg_log_likelihood + batch, loss)
# PLACEHOLDER_KERNELS2


@libentry()
@triton.jit
def _ctc_loss_forward_no_grad_kernel(
    log_probs,
    targets,
    input_lengths,
    target_lengths,
    target_offsets,
    neg_log_likelihood,
    scratch,
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
    valid = states < state_count
    stored = states < STATE_COUNT_MAX
    is_blank = (states % 2) == 0
    tidx = (states - 1) // 2
    tmask = (tidx >= 0) & (tidx < target_len)
    tsafe = tl.where(tmask, tidx, 0)
    if TARGET_1D:
        torigin = tl.load(target_offsets + batch)
    else:
        torigin = batch * MAX_TARGET
    tval = tl.where(tmask, tl.load(targets + torigin + tsafe), BLANK)
    labels = tl.where(is_blank, BLANK, tval)
    ptidx = tl.where(tidx > 0, tidx - 1, 0)
    ptval = tl.where(tmask & (tidx > 0), tl.load(targets + torigin + ptidx), BLANK)
    skip = (~is_blank) & (tidx > 0) & (tval != ptval)

    t0 = input_len > 0
    init_state = (states == 0) | ((states == 1) & (target_len > 0))
    il = tl.where(
        init_state & stored & t0, tl.load(log_probs + batch * C + labels), 0.0
    ).to(tl.float32)
    alpha = tl.where(init_state & valid & t0, il, -float("inf"))
    sbb = scratch + batch * STATE_COUNT_MAX
    for t in tl.range(1, T, loop_unroll_factor=1):
        tl.store(sbb + states, alpha, mask=stored)
        tl.debug_barrier()
        prev1 = _safe_shift(sbb, states - 1, (states > 0) & stored, STATE_COUNT_MAX, -float("inf"))
        prev2 = _safe_shift(sbb, states - 2, (states > 1) & stored, STATE_COUNT_MAX, -float("inf"))
        tl.debug_barrier()
        acc = _lae3(alpha, prev1, prev2, skip)
        logp = tl.where(
            valid & (t < input_len),
            tl.load(log_probs + t * N * C + batch * C + labels),
            0.0,
        ).to(tl.float32)
        alpha = tl.where(valid & (t < input_len), acc + logp, alpha)

    if input_len <= 0:
        loss = tl.where(target_len == 0, 0.0, float("inf"))
    else:
        tl.store(sbb + states, alpha, mask=stored)
        tl.debug_barrier()
        last = tl.load(sbb + state_count - 1).to(tl.float32)
        pl = _safe_shift(
            sbb, tl.where(target_len > 0, state_count - 2, 0), target_len > 0,
            STATE_COUNT_MAX, -float("inf"),
        )
        loss = -_lae(last, pl)
    tl.store(neg_log_likelihood + batch, loss)


@libentry()
@triton.jit
def _ctc_loss_forward_full_length_reduce_kernel(
    log_probs,
    targets,
    target_lengths,
    target_offsets,
    contrib,
    scratch,
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
    valid = states < state_count
    stored = states < STATE_COUNT_MAX
    is_blank = (states % 2) == 0
    tidx = (states - 1) // 2
    tmask = (tidx >= 0) & (tidx < target_len)
    tsafe = tl.where(tmask, tidx, 0)
    if TARGET_1D:
        torigin = tl.load(target_offsets + batch)
    else:
        torigin = batch * MAX_TARGET
    tval = tl.where(tmask, tl.load(targets + torigin + tsafe), BLANK)
    labels = tl.where(is_blank, BLANK, tval)
    ptidx = tl.where(tidx > 0, tidx - 1, 0)
    ptval = tl.where(tmask & (tidx > 0), tl.load(targets + torigin + ptidx), BLANK)
    skip = (~is_blank) & (tidx > 0) & (tval != ptval)

    init_state = (states == 0) | ((states == 1) & (target_len > 0))
    il = tl.where(
        init_state & stored, tl.load(log_probs + batch * C + labels), 0.0
    ).to(tl.float32)
    alpha = tl.where(init_state & valid, il, -float("inf"))
    sbb = scratch + batch * STATE_COUNT_MAX
    for t in tl.range(1, T, loop_unroll_factor=1):
        tl.store(sbb + states, alpha, mask=stored)
        tl.debug_barrier()
        prev1 = _safe_shift(sbb, states - 1, (states > 0) & stored, STATE_COUNT_MAX, -float("inf"))
        prev2 = _safe_shift(sbb, states - 2, (states > 1) & stored, STATE_COUNT_MAX, -float("inf"))
        tl.debug_barrier()
        acc = _lae3(alpha, prev1, prev2, skip)
        logp = tl.where(
            valid, tl.load(log_probs + t * N * C + batch * C + labels), 0.0
        ).to(tl.float32)
        alpha = tl.where(valid, acc + logp, -float("inf"))

    if T <= 0:
        loss = tl.where(target_len == 0, 0.0, float("inf"))
    else:
        tl.store(sbb + states, alpha, mask=stored)
        tl.debug_barrier()
        last = tl.load(sbb + state_count - 1).to(tl.float32)
        pl = _safe_shift(
            sbb, tl.where(target_len > 0, state_count - 2, 0), target_len > 0,
            STATE_COUNT_MAX, -float("inf"),
        )
        loss = -_lae(last, pl)
    if REDUCTION == 1:
        loss = loss / tl.maximum(target_len, 1).to(tl.float32) / N
    tl.store(contrib + batch, loss)
# PLACEHOLDER_KERNELS3


@libentry()
@triton.jit
def _ctc_loss_backward_kernel(
    log_probs,
    targets,
    input_lengths,
    target_lengths,
    target_offsets,
    neg_log_likelihood,
    log_alpha,
    grad_output,
    grad_input,
    scratch,
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
    valid = states < state_count
    stored = states < STATE_COUNT_MAX
    is_blank = (states % 2) == 0
    tidx = (states - 1) // 2
    tmask = (tidx >= 0) & (tidx < target_len)
    tsafe = tl.where(tmask, tidx, 0)
    if TARGET_1D:
        torigin = tl.load(target_offsets + batch)
    else:
        torigin = batch * MAX_TARGET
    tval = tl.where(tmask, tl.load(targets + torigin + tsafe), BLANK)
    labels = tl.where(is_blank, BLANK, tval)
    s1 = states + 1
    isb1 = (s1 % 2) == 0
    ti1 = (s1 - 1) // 2
    tm1 = (ti1 >= 0) & (ti1 < target_len)
    ts1 = tl.where(tm1, ti1, 0)
    tv1 = tl.where(tm1, tl.load(targets + torigin + ts1), BLANK)
    labels1 = tl.where(isb1, BLANK, tv1)
    s2 = states + 2
    ti2 = (s2 - 1) // 2
    tm2 = (ti2 >= 0) & (ti2 < target_len)
    ts2 = tl.where(tm2, ti2, 0)
    tv2 = tl.where(tm2, tl.load(targets + torigin + ts2), BLANK)
    labels2 = tv2

    if REDUCTION == 0:
        scale = tl.load(grad_output + batch).to(tl.float32)
    else:
        scale = tl.load(grad_output).to(tl.float32)
        if REDUCTION == 1:
            scale = scale / (tl.maximum(target_len, 1).to(tl.float32) * N)
    if ZERO_INFINITY:
        scale = tl.where(nll == float("inf"), 0.0, scale)

    beta = tl.where(
        ((states == state_count - 1) | ((states == state_count - 2) & (target_len > 0)))
        & valid
        & (input_len > 0),
        0.0,
        -float("inf"),
    )
    sbb = scratch + batch * STATE_COUNT_MAX
    loglik = tl.where(scale != 0.0, -nll, 0.0)
    for step in tl.range(0, T, loop_unroll_factor=1):
        t = input_len - 1 - step
        active = t >= 0
        safe_t = tl.where(active, t, 0)
        alpha_t = _safe_shift(
            log_alpha + batch * T * STATE_COUNT_MAX + safe_t * STATE_COUNT_MAX,
            states, active & stored, STATE_COUNT_MAX, -float("inf"),
        )
        log_post = alpha_t + beta - loglik
        posterior = tl.where(
            active & valid & (scale != 0.0), tl.exp(log_post) * scale, 0.0
        )
        # XPU tl.atomic_add drops its mask and mis-sums colliding lanes; the per-batch
        # grid gives each (t, b, c) a single owner, so reduce per class and plain-RMW.
        for c in tl.static_range(0, C):
            contrib = tl.sum(tl.where((labels == c) & valid & stored, posterior, 0.0))
            cur = tl.load(grad_input + safe_t * N * C + batch * C + c).to(tl.float32)
            tl.store(grad_input + safe_t * N * C + batch * C + c, cur - contrib)
        tl.store(sbb + states, beta, mask=stored)
        tl.debug_barrier()
        lp_cur = tl.where(
            active & valid,
            tl.load(log_probs + safe_t * N * C + batch * C + labels),
            -float("inf"),
        ).to(tl.float32)
        stay = beta + lp_cur
        b1 = _safe_shift(sbb, states + 1, (states + 1 < state_count) & stored, STATE_COUNT_MAX, -float("inf"))
        lp1 = tl.where(
            active & (states + 1 < state_count) & stored,
            tl.load(log_probs + safe_t * N * C + batch * C + labels1),
            -float("inf"),
        ).to(tl.float32)
        n1 = b1 + lp1
        skip = (~is_blank) & (states + 2 < state_count) & (tval != tv2)
        b2 = _safe_shift(sbb, states + 2, (states + 2 < state_count) & stored, STATE_COUNT_MAX, -float("inf"))
        lp2 = tl.where(
            active & skip & stored,
            tl.load(log_probs + safe_t * N * C + batch * C + labels2),
            -float("inf"),
        ).to(tl.float32)
        n2 = b2 + lp2
        tl.debug_barrier()
        bn = _lae3(stay, n1, n2, skip)
        beta = tl.where(active, bn, beta)
# PLACEHOLDER_DRIVER


class CtcLossFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        log_probs,
        targets,
        input_lengths,
        target_lengths,
        blank=0,
        reduction="mean",
        zero_infinity=False,
    ):
        reduction = _reduction_enum(reduction)
        if reduction not in (_REDUCTION_NONE, _REDUCTION_MEAN, _REDUCTION_SUM):
            raise ValueError(f"ctc_loss got invalid reduction enum {reduction}")

        if log_probs.ndim not in (2, 3):
            raise RuntimeError(
                "ctc_loss expects log_probs to be a 2D or 3D tensor, "
                f"but got {log_probs.ndim}D"
            )
        if not torch.is_floating_point(log_probs):
            raise RuntimeError(f'"ctc_loss" not implemented for {log_probs.dtype}')
        if blank < 0 or blank >= log_probs.shape[-1]:
            raise RuntimeError("blank must be in label range")

        original_dtype = log_probs.dtype
        compute_dtype = _compute_dtype(original_dtype)
        unbatched = log_probs.ndim == 2
        batch_size = 1 if unbatched else log_probs.shape[1]

        work_log_probs = log_probs.unsqueeze(1) if unbatched else log_probs
        work_log_probs = work_log_probs.contiguous()
        if work_log_probs.dtype != compute_dtype:
            work_log_probs = work_log_probs.to(compute_dtype)

        if torch.is_floating_point(targets):
            work_targets = targets.to(dtype=torch.long).contiguous()
        elif _is_integral_dtype(targets.dtype):
            work_targets = targets.contiguous()
        else:
            raise RuntimeError("ctc_loss targets must be integral or floating point")
        work_input_lengths = _lengths_to_tensor(
            input_lengths, log_probs.device, "input_lengths"
        )
        work_target_lengths = _lengths_to_tensor(
            target_lengths, log_probs.device, "target_lengths"
        )
        if work_input_lengths.numel() != batch_size:
            raise RuntimeError(
                f"ctc_loss expected input_lengths to have size {batch_size}, "
                f"but got {work_input_lengths.numel()}"
            )
        if work_target_lengths.numel() != batch_size:
            raise RuntimeError(
                f"ctc_loss expected target_lengths to have size {batch_size}, "
                f"but got {work_target_lengths.numel()}"
            )
        min_input_length, max_input_length, _ = _length_stats(work_input_lengths)
        min_target_length, max_target, total_target_length = _length_stats(
            work_target_lengths
        )
        if min_input_length < 0 or max_input_length > work_log_probs.shape[0]:
            raise RuntimeError("ctc_loss input_lengths must be in [0, T]")
        if min_target_length < 0:
            raise RuntimeError("ctc_loss target_lengths must be non-negative")

        state_count_max = 2 * max_target + 1
        target_stride = max_target
        if work_targets.ndim == 1:
            target_1d = True
            if total_target_length != work_targets.numel():
                raise RuntimeError(
                    "ctc_loss expected concatenated targets length to equal "
                    "sum(target_lengths)"
                )
            work_target_offsets = (
                work_target_lengths.cumsum(0) - work_target_lengths
            ).contiguous()
        elif work_targets.ndim == 2:
            target_1d = False
            if max_target > work_targets.shape[1]:
                raise RuntimeError(
                    "ctc_loss target_lengths cannot exceed padded target width"
                )
            target_stride = work_targets.shape[1]
            work_target_offsets = work_target_lengths
        else:
            raise RuntimeError(
                "ctc_loss expects targets to be a 1D concatenated tensor or a "
                f"2D padded tensor, but got {work_targets.ndim}D"
            )

        needs_log_probs_grad = ctx.needs_input_grad[0]
        block_s = state_count_max
        T = work_log_probs.shape[0]
        C = work_log_probs.shape[2]

        if not needs_log_probs_grad:
            if (
                not unbatched
                and not zero_infinity
                and reduction in (_REDUCTION_MEAN, _REDUCTION_SUM)
                and min_input_length == T
                and T > 0
            ):
                contrib = torch.empty(
                    (batch_size,), dtype=torch.float32, device=log_probs.device
                )
                scratch = torch.empty(
                    (batch_size, state_count_max),
                    dtype=torch.float32,
                    device=log_probs.device,
                )
                with torch_device_fn.device(log_probs.device):
                    _ctc_loss_forward_full_length_reduce_kernel[(batch_size,)](
                        work_log_probs,
                        work_targets,
                        work_target_lengths,
                        work_target_offsets,
                        contrib,
                        scratch,
                        T,
                        batch_size,
                        C,
                        target_stride,
                        state_count_max,
                        blank,
                        target_1d,
                        reduction,
                        block_s,
                    )
                output = contrib.sum()
                if output.dtype != original_dtype:
                    output = output.to(original_dtype)
                return output

            raw_neg_log_likelihood = torch.empty(
                (batch_size,), dtype=torch.float32, device=log_probs.device
            )
            scratch = torch.empty(
                (batch_size, state_count_max),
                dtype=torch.float32,
                device=log_probs.device,
            )
            with torch_device_fn.device(log_probs.device):
                _ctc_loss_forward_no_grad_kernel[(batch_size,)](
                    work_log_probs,
                    work_targets,
                    work_input_lengths,
                    work_target_lengths,
                    work_target_offsets,
                    raw_neg_log_likelihood,
                    scratch,
                    T,
                    batch_size,
                    C,
                    target_stride,
                    state_count_max,
                    blank,
                    target_1d,
                    block_s,
                )
            neg_log_likelihood = raw_neg_log_likelihood
            if zero_infinity:
                neg_log_likelihood = torch.where(
                    torch.isinf(neg_log_likelihood),
                    torch.zeros(
                        (), dtype=neg_log_likelihood.dtype, device=log_probs.device
                    ),
                    neg_log_likelihood,
                )

            if reduction == _REDUCTION_NONE:
                output = neg_log_likelihood
                if unbatched:
                    output = output.squeeze(0)
            elif reduction == _REDUCTION_SUM:
                output = neg_log_likelihood.sum()
            else:
                denom = work_target_lengths.clamp_min(1)
                output = (neg_log_likelihood / denom).mean()

            if output.dtype != original_dtype:
                output = output.to(original_dtype)
            return output

        raw_neg_log_likelihood = torch.empty(
            (batch_size,), dtype=torch.float32, device=log_probs.device
        )
        log_alpha = torch.empty(
            (batch_size, T, state_count_max),
            dtype=torch.float32,
            device=log_probs.device,
        )
        scratch = torch.empty(
            (batch_size, state_count_max),
            dtype=torch.float32,
            device=log_probs.device,
        )
        with torch_device_fn.device(log_probs.device):
            _ctc_loss_forward_kernel[(batch_size,)](
                work_log_probs,
                work_targets,
                work_input_lengths,
                work_target_lengths,
                work_target_offsets,
                raw_neg_log_likelihood,
                log_alpha,
                scratch,
                T,
                batch_size,
                C,
                target_stride,
                state_count_max,
                blank,
                target_1d,
                block_s,
            )
        neg_log_likelihood = raw_neg_log_likelihood
        if zero_infinity:
            neg_log_likelihood = torch.where(
                torch.isinf(neg_log_likelihood),
                torch.zeros(
                    (), dtype=neg_log_likelihood.dtype, device=log_probs.device
                ),
                neg_log_likelihood,
            )

        if reduction == _REDUCTION_NONE:
            output = neg_log_likelihood
            if unbatched:
                output = output.squeeze(0)
            if output.dtype != original_dtype:
                output = output.to(original_dtype)
        elif reduction == _REDUCTION_SUM:
            output = neg_log_likelihood.sum()
        else:
            denom = work_target_lengths.clamp_min(1)
            output = (neg_log_likelihood / denom).mean()

        if output.dtype != original_dtype:
            output = output.to(original_dtype)

        ctx.save_for_backward(
            work_log_probs,
            work_targets,
            work_input_lengths,
            work_target_lengths,
            work_target_offsets,
            raw_neg_log_likelihood,
            log_alpha,
        )
        ctx.blank = blank
        ctx.reduction = reduction
        ctx.zero_infinity = zero_infinity
        ctx.unbatched = unbatched
        ctx.batch_size = batch_size
        ctx.original_dtype = original_dtype
        ctx.max_target = target_stride
        ctx.state_count_max = state_count_max
        ctx.target_1d = target_1d
        return output

    @staticmethod
    def backward(ctx, grad_output):
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
        T = work_log_probs.shape[0]
        C = work_log_probs.shape[2]
        with torch_device_fn.device(work_log_probs.device):
            _ctc_loss_init_grad_kernel[(triton.cdiv(total, block),)](
                work_log_probs,
                work_input_lengths,
                work_target_lengths,
                neg_log_likelihood,
                grad_output,
                grad_log_probs,
                total,
                T,
                ctx.batch_size,
                C,
                ctx.reduction,
                ctx.zero_infinity,
                block,
            )
            scratch = torch.empty(
                (ctx.batch_size, ctx.state_count_max),
                dtype=torch.float32,
                device=work_log_probs.device,
            )
            block_s = ctx.state_count_max
            _ctc_loss_backward_kernel[(ctx.batch_size,)](
                work_log_probs,
                work_targets,
                work_input_lengths,
                work_target_lengths,
                work_target_offsets,
                neg_log_likelihood,
                log_alpha,
                grad_output,
                grad_log_probs,
                scratch,
                T,
                ctx.batch_size,
                C,
                ctx.max_target,
                ctx.state_count_max,
                ctx.blank,
                ctx.target_1d,
                ctx.reduction,
                ctx.zero_infinity,
                block_s,
            )

        if ctx.unbatched:
            grad_log_probs = grad_log_probs.squeeze(1)
        if grad_log_probs.dtype != ctx.original_dtype:
            grad_log_probs = grad_log_probs.to(ctx.original_dtype)
        return grad_log_probs, None, None, None, None, None, None


def ctc_loss(
    log_probs,
    targets,
    input_lengths,
    target_lengths,
    blank=0,
    reduction="mean",
    zero_infinity=False,
):
    logger.debug("GEMS_KUNLUNXIN CTC LOSS")
    return CtcLossFunction.apply(
        log_probs,
        targets,
        input_lengths,
        target_lengths,
        blank,
        reduction,
        zero_infinity,
    )
