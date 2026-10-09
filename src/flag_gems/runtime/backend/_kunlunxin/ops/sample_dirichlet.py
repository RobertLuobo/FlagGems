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

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils.random_utils import (
    philox_backend_seed_offset,
    uint_to_uniform_float,
)

logger = logging.getLogger(__name__)

MAX_ITERS = 32
STRIDE = (MAX_ITERS + 1) * 4


@triton.jit
def _sample_gamma_block(alpha, seed, base_counter, TINY, MAX_ITERS: tl.constexpr):
    """Marsaglia-Tsang Gamma(alpha, 1) draw for a block of alphas.

    Deterministic in (seed, base_counter) so the two normalization passes
    reproduce the identical gammas. Acceptance is tracked in an int32 tensor
    (not a tl.int1) and the loop is a fixed-count `for`: the generic kernel's
    dynamic `while (tl.sum((~done)) > 0)` early-exit lowers to a `tt.reduce`
    that is marked illegal by the XPU3 TritonXPU pipeline.
    """
    alpha_f32 = alpha.to(tl.float32)

    alpha_adj = tl.where(alpha_f32 < 1.0, alpha_f32 + 1.0, alpha_f32)
    d = alpha_adj - 0.3333333333333333
    c = 1.0 / tl.sqrt(9.0 * d)

    # For alpha < 1, Gamma(alpha) = Gamma(alpha+1) * U**(1/alpha). Draw the
    # boost uniform at a counter past the rejection loop so it never collides.
    bc = base_counter + MAX_ITERS * 4
    bc0 = (bc & 0xFFFFFFFF).to(tl.uint32)
    bc1 = ((bc >> 32) & 0xFFFFFFFF).to(tl.uint32)
    bz = bc0 * 0
    br0, _, _, _ = tl.philox(seed, bc0, bc1, bz, bz)
    ub = tl.maximum(uint_to_uniform_float(br0), 1e-7)
    boost = tl.where(alpha_f32 < 1.0, tl.exp(tl.log(ub) / alpha_f32), 1.0)

    accepted = (alpha_f32 * 0.0).to(tl.int32)
    result_f32 = alpha_f32 * 0.0

    for it in range(0, MAX_ITERS):
        counter = base_counter + it * 4
        c0 = (counter & 0xFFFFFFFF).to(tl.uint32)
        c1 = ((counter >> 32) & 0xFFFFFFFF).to(tl.uint32)
        z0 = c0 * 0
        r0, r1, r2, r3 = tl.philox(seed, c0, c1, z0, z0)
        u2 = tl.maximum(uint_to_uniform_float(r1), 1e-7)
        u3 = uint_to_uniform_float(r2)
        u4 = tl.maximum(uint_to_uniform_float(r3), 1e-7)

        z = tl.sqrt(-2.0 * tl.log(u2)) * tl.cos(2.0 * 3.141592653589793 * u3)
        v = 1.0 + c * z
        v3 = v * v * v

        accept_i = tl.where(
            (v > 0.0) & (tl.log(u4) < (0.5 * z * z + d - d * v3 + d * tl.log(v3))),
            1,
            0,
        ).to(tl.int32)
        take_i = accept_i * (1 - accepted)
        result_f32 = tl.where(take_i != 0, d * v3, result_f32)
        accepted = tl.where(accept_i != 0, 1, accepted)

    result_f32 = result_f32 * boost
    result_f32 = tl.maximum(result_f32, TINY)
    return result_f32


@libentry()
@triton.jit(do_not_specialize=["philox_seed", "philox_offset"])
def sample_dirichlet_kernel(
    alpha_ptr,
    out_ptr,
    N,
    K,
    philox_seed,
    philox_offset,
    TINY,
    BLOCK_K: tl.constexpr,
    STRIDE: tl.constexpr,
    MAX_ITERS: tl.constexpr,
):
    """Dirichlet(alpha) = Gamma(alpha_k) / sum_j Gamma(alpha_j) per row [N, K].

    Two passes over each row so wide rows (K > BLOCK_K) normalize by the full
    row sum: pass 1 accumulates the sum, pass 2 re-samples (deterministic under
    the same philox counters) and divides.
    """
    philox_seed = philox_seed.to(tl.int64)
    philox_offset = philox_offset.to(tl.int64)

    row_id = tl.program_id(0)

    gamma_sum = tl.zeros([], dtype=tl.float32)
    for k_start in range(0, K, BLOCK_K):
        k_offs = k_start + tl.arange(0, BLOCK_K)
        mask = k_offs < K
        alpha_idx = row_id * K + k_offs
        alpha = tl.load(alpha_ptr + alpha_idx, mask=mask, other=1.0)
        base_counter = philox_offset + alpha_idx.to(tl.int64) * STRIDE
        gamma = _sample_gamma_block(alpha, philox_seed, base_counter, TINY, MAX_ITERS)
        gamma_sum += tl.sum(tl.where(mask, gamma, 0.0))

    inv_sum = 1.0 / gamma_sum

    for k_start in range(0, K, BLOCK_K):
        k_offs = k_start + tl.arange(0, BLOCK_K)
        mask = k_offs < K
        alpha_idx = row_id * K + k_offs
        alpha = tl.load(alpha_ptr + alpha_idx, mask=mask, other=1.0)
        base_counter = philox_offset + alpha_idx.to(tl.int64) * STRIDE
        gamma = _sample_gamma_block(alpha, philox_seed, base_counter, TINY, MAX_ITERS)
        normalized = gamma * inv_sum
        tl.store(
            out_ptr + alpha_idx, normalized.to(out_ptr.dtype.element_ty), mask=mask
        )


def _sample_dirichlet(input, generator=None):
    logger.debug("GEMS_KUNLUNXIN SAMPLE_DIRICHLET")
    logger.debug("GEMS _SAMPLE_DIRICHLET")

    assert input.dtype in (
        torch.float16,
        torch.bfloat16,
        torch.float32,
        torch.float64,
    ), f"Unsupported dtype: {input.dtype}"
    if input.dtype == torch.float64:
        raise RuntimeError("flag_gems _sample_dirichlet does not support float64 inputs")

    orig_shape = input.shape
    if input.ndim == 0:
        input = input.unsqueeze(0)

    if input.ndim == 1:
        K = input.shape[0]
        N = 1
        inp = input.unsqueeze(0)
    else:
        inp = input.reshape(-1, input.shape[-1])
        N = inp.shape[0]
        K = inp.shape[1]

    inp = inp.contiguous()
    out = torch.empty_like(inp)

    if N == 0 or K == 0:
        return out.reshape(orig_shape)

    increment = triton.cdiv(N * K * STRIDE, 4)
    philox_seed, philox_offset = philox_backend_seed_offset(
        increment, generator=generator
    )

    # Cap BLOCK_K at 128: the two-pass (sample-sum then re-sample-normalize)
    # loop keeps a block's worth of fp32 philox temporaries live across both
    # passes, and at BLOCK_K >= 256 the XPU3 TritonXPUMemoryInplace pass runs
    # out of uni_sram (and mis-pipelines the reloaded `alpha`, "operand does
    # not dominate this use") for fp32 output on wide rows (K=4096). 128 fits.
    BLOCK_K = triton.next_power_of_2(min(K, 128))
    tiny = torch.finfo(inp.dtype).tiny

    # Constant grid tuple (grid depends only on N): avoids the per-call
    # `grid = lambda` recompile that otherwise costs hundreds of ms/call on XPU.
    grid = (N,)
    with torch_device_fn.device(inp.device):
        sample_dirichlet_kernel[grid](
            inp,
            out,
            N,
            K,
            philox_seed,
            philox_offset,
            tiny,
            BLOCK_K=BLOCK_K,
            STRIDE=STRIDE,
            MAX_ITERS=MAX_ITERS,
        )

    return out.reshape(orig_shape)
