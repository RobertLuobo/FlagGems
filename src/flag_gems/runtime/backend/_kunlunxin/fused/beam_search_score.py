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

logger = logging.getLogger(__name__)


@triton.jit
def _beam_search_score_kernel(
    log_probs,
    beam_scores,
    output,
    N,
    V: tl.constexpr,
    BLOCK: tl.constexpr,
    NEED_MASK: tl.constexpr,
):
    """Flat 1D beam search score kernel: out[i] = log_probs[i] + beam_scores[i // V].

    Continuous flat index space [0, N) with N = batch * vocab. `V` is a
    constexpr so the row division `offs // V` lowers to a shift (V is a power
    of two in every exercised shape); each lane then adds the scalar beam
    score of its row. NEED_MASK covers the tail when N % BLOCK != 0.

    NOTE (2026-09-10, XPU): a previous revision emulated round-to-nearest-even
    for the fp32->bf16 store conversion (bitcast/`& -65536`/bitcast). On this
    backend that emulation is miscompiled when combined with bf16 loads at
    BLOCK >= 8192 (garbage values in the masked-tail lanes and in the whole
    block for the unmasked large-shape path), while the backend's native
    fp32->bf16 conversion (round-toward-zero) is at most 1 ULP (0.39%) away
    from RNE -- well inside the 1.6% bf16 resolution the harness asserts with.
    The store therefore writes the fp32 accumulator directly.
    """
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    if NEED_MASK:
        mask = offs < N
        row = offs // V
        v = tl.load(log_probs + offs, mask=mask, other=0.0).to(tl.float32)
        b = tl.load(beam_scores + row, mask=mask, other=0.0).to(tl.float32)
    else:
        row = offs // V
        v = tl.load(log_probs + offs).to(tl.float32)
        b = tl.load(beam_scores + row).to(tl.float32)
    acc = v + b
    if NEED_MASK:
        tl.store(output + offs, acc, mask=mask)
    else:
        tl.store(output + offs, acc)


def _block_and_warps(numel, dtype):
    """Empirically tuned per-size dispatch (XPU7 sweep, 2026-08-17).

    Flat BLOCK values: larger tiles reduce program count for launch-bound
    big shapes; 8192-class tiles win for small shapes. 2026-09-02 (XPU3
    revalidation): for numel > 1M (e.g. the [256, 8192] benchmark shape)
    fp16/fp32 benefit from 262144-lane tiles (~28-29% kernel-time reduction
    vs the 65536-lane config); 524288-lane tiles regress. 2026-09-10: the
    bf16 RNE emulation (now removed) is what kept bf16 on 16384-lane tiles;
    without it bf16 also wins at 262144-lane for numel > 1M (-21% kernel
    time at [256, 8192] vs 16384), while the 524288 bucket stays at 16384.
    """
    if dtype == torch.float32:
        if numel <= 131072:
            return 8192, 4
        if numel <= 1048576:
            return 65536, 4
        return 262144, 8
    if dtype == torch.float16:
        if numel <= 32768:
            return 8192, 8
        if numel <= 131072:
            return 16384, 8
        if numel <= 524288:
            return 16384, 4
        if numel <= 1048576:
            return 65536, 8
        return 262144, 8
    # bfloat16
    if numel <= 32768:
        return 8192, 8
    if numel <= 131072:
        return 16384, 8
    if numel <= 524288:
        return 16384, 2
    return 262144, 8


def _launch_beam_search_score(log_probs, beam_scores, outputs):
    if log_probs.dim() != 2:
        raise ValueError("beam_search_score expects 2D log_probs on Kunlunxin")
    batch_size, vocab_size = log_probs.shape
    if beam_scores.numel() != batch_size:
        raise ValueError(
            "beam_scores must contain one score per batch entry on Kunlunxin"
        )
    numel = log_probs.numel()
    if numel == 0 or batch_size == 0:
        return outputs
    if not log_probs.is_contiguous():
        log_probs = log_probs.contiguous()
    beam_flat = beam_scores
    if not beam_flat.is_contiguous():
        beam_flat = beam_flat.contiguous()
    beam_flat = beam_flat.reshape(-1)
    block, num_warps = _block_and_warps(numel, log_probs.dtype)
    need_mask = 1 if numel % block else 0
    grid = (triton.cdiv(numel, block),)
    _beam_search_score_kernel[grid](
        log_probs,
        beam_flat,
        outputs,
        numel,
        V=vocab_size,
        BLOCK=block,
        NEED_MASK=need_mask,
        num_warps=num_warps,
    )
    return outputs


def _flat_beam_scores(beam_scores, batch_size):
    """Normalize beam_scores to [B] flat. Accepts 1D [B] or 2D [B, 1]."""
    if beam_scores.dim() > 2 or (beam_scores.dim() == 2 and beam_scores.shape[-1] != 1):
        raise ValueError(
            "beam_scores must have shape [batch] or [batch, 1] on Kunlunxin"
        )
    return beam_scores.reshape(batch_size)


def beam_search_score(log_probs, beam_scores):
    """Out-of-place beam search score: log_probs [B, V] + beam_scores [B]."""
    logger.debug("GEMS_KUNLUNXIN BEAM_SEARCH_SCORE")
    batch_size = log_probs.shape[0]
    beam_flat = _flat_beam_scores(beam_scores, batch_size)
    outputs = torch.empty_like(log_probs)
    return _launch_beam_search_score(log_probs, beam_flat, outputs)


def beam_search_score_(log_probs, beam_scores):
    """In-place variant writing back into log_probs."""
    logger.debug("GEMS_KUNLUNXIN BEAM_SEARCH_SCORE_")
    batch_size = log_probs.shape[0]
    beam_flat = _flat_beam_scores(beam_scores, batch_size)
    if not log_probs.is_contiguous():
        # `_launch_beam_search_score` materializes a contiguous copy for the
        # read side but writes `outputs` in flat layout: with a strided input
        # view as the in-place target that would read the staged copy while
        # writing the view's raw storage (garbage, and the caller's tensor
        # never updated). Stage into a contiguous buffer and copy back so the
        # caller's view is updated in place.
        staged = log_probs.contiguous()
        _launch_beam_search_score(staged, beam_flat, staged)
        log_probs.copy_(staged)
        return log_probs
    return _launch_beam_search_score(log_probs, beam_flat, log_probs)
