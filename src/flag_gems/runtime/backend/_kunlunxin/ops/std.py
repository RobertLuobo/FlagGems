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

from flag_gems import runtime
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import dim_compress, libentry

logger = logging.getLogger(__name__)


@triton.jit
def _std_partial_sum_kernel(X, Tmp, N, CHUNK, BLOCK_N: tl.constexpr):
    # Each program accumulates the sum of a contiguous CHUNK of elements in
    # BLOCK_N-sized tiles (BLOCK_N < 8192: safe tl.sum on XPU). Partial sums
    # are reduced by _std_finalize_kernel. IMPORTANT: the load mask must
    # bound to the program-local end (min(start+CHUNK, N)), never the global
    # N, otherwise the last tile of each program overlaps the next program's
    # range and partial sums are inflated.
    pid = tl.program_id(0)
    start = pid * CHUNK
    end = tl.minimum(start + CHUNK, N)
    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
    for off in range(start, end, BLOCK_N):
        offset = off + tl.arange(0, BLOCK_N)
        mask = offset < end
        x = tl.load(X + offset, mask=mask, other=0.0).to(tl.float32)
        acc += x
    tl.store(Tmp + pid, tl.sum(acc, axis=0))


@triton.jit
def _std_partial_sq_kernel(X, Tmp, N, Mean, CHUNK, BLOCK_N: tl.constexpr):
    # Second pass: sum of squared deviations from the already-computed mean.
    # Two-pass (mean, then (x-mean)^2) avoids the E[x^2]-E[x]^2 catastrophic
    # cancellation that silently zeroes the variance for large N with
    # non-zero mean. Mask bounded to program-local end, see above.
    # NOTE: OOB lanes must be explicitly zeroed with tl.where AFTER the
    # subtraction. On XPU the masked-load other=0.0 value does not survive
    # the v - mean subf in tiled loops; without the where, each program
    # under-counts its mean term by ~12 lanes and the squared sum is
    # inflated ~100x for non-zero-mean data.
    pid = tl.program_id(0)
    start = pid * CHUNK
    end = tl.minimum(start + CHUNK, N)
    mean = tl.load(Mean)
    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
    for off in range(start, end, BLOCK_N):
        offset = off + tl.arange(0, BLOCK_N)
        mask = offset < end
        x = tl.load(X + offset, mask=mask, other=0.0).to(tl.float32)
        d = tl.where(mask, x - mean, 0.0)
        acc += d * d
    tl.store(Tmp + pid, tl.sum(acc, axis=0))


@triton.jit
def _std_finalize_kernel(
    Tmp, Out, N, correction, BLOCK_NUM, BLOCK_SIZE: tl.constexpr, SQRT_OUT: tl.constexpr
):
    total_acc = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    for off in range(0, BLOCK_NUM, BLOCK_SIZE):
        offset = off + tl.arange(0, BLOCK_SIZE)
        mask = offset < BLOCK_NUM
        v = tl.load(Tmp + offset, mask=mask, other=0.0).to(tl.float32)
        total_acc += v
    total = tl.sum(total_acc, axis=0)
    if SQRT_OUT:
        denom = N - correction
        var = total / tl.maximum(denom, 1e-12)
        val = tl.sqrt(tl.maximum(var, 0.0))
    else:
        val = total / N
    tl.store(Out, val.to(Out.dtype.element_ty))


@libentry()
@triton.heuristics(runtime.get_heuristic_config("softmax_inner"))
@triton.jit(do_not_specialize=["correction"])
def _std_dim_kernel_inner(
    Out,
    X,
    M,
    N,
    correction,
    TILE_N: tl.constexpr,
    ONE_TILE_PER_CTA: tl.constexpr,
):
    pid_m = tl.program_id(0)

    if ONE_TILE_PER_CTA:
        n_offsets = tl.arange(0, TILE_N)
        mask = n_offsets < N
        x = tl.load(X + pid_m * N + n_offsets, mask=mask, other=0.0).to(tl.float32)
        mean = tl.sum(x, axis=0) / N
    else:
        sum_acc = tl.zeros((TILE_N,), dtype=tl.float32)
        for start_n in range(0, N, TILE_N):
            n_offsets = start_n + tl.arange(0, TILE_N)
            mask = n_offsets < N
            x = tl.load(X + pid_m * N + n_offsets, mask=mask, other=0.0).to(tl.float32)
            sum_acc += x
        mean = tl.sum(sum_acc, axis=0) / N

    if ONE_TILE_PER_CTA:
        n_offsets = tl.arange(0, TILE_N)
        mask = n_offsets < N
        x = tl.load(X + pid_m * N + n_offsets, mask=mask, other=0.0).to(tl.float32)
        diff = tl.where(mask, x - mean, 0.0)
        sq_sum = tl.sum(diff * diff, axis=0)
    else:
        sq_acc = tl.zeros((TILE_N,), dtype=tl.float32)
        for start_n in range(0, N, TILE_N):
            n_offsets = start_n + tl.arange(0, TILE_N)
            mask = n_offsets < N
            x = tl.load(X + pid_m * N + n_offsets, mask=mask, other=0.0).to(tl.float32)
            diff = tl.where(mask, x - mean, 0.0)
            sq_acc += diff * diff
        sq_sum = tl.sum(sq_acc, axis=0)

    denom = N - correction
    var = sq_sum / tl.maximum(denom, 1e-12)
    std_dev = tl.sqrt(tl.maximum(var, 0.0))
    tl.store(Out + pid_m, std_dev.to(Out.dtype.element_ty), mask=pid_m < M)


def _std_dim_dispatch(out, x_contiguous, M, N, K, effective_correction):
    # Every dim reduction is routed through dim_compress => K is always 1 and we
    # only use the verified-correct inner kernel.
    with torch_device_fn.device(x_contiguous.device):
        grid = (M, 1, 1)
        _std_dim_kernel_inner[grid](out, x_contiguous, M, N, effective_correction)


def std(x, dim=None, *, correction=None, keepdim=False):
    logger.debug("GEMS_KUNLUNXIN STD")
    effective_correction = 1.0 if correction is None else float(correction)
    original_shape = x.shape
    input_ndim = x.ndim

    if dim is None:
        N = x.numel()
        if N == 0 or N - effective_correction <= 0:
            return torch.full([], float("nan"), device=x.device, dtype=x.dtype)
        if N == 1 and effective_correction == 0.0:
            out = torch.zeros([], device=x.device, dtype=x.dtype)
            return out.view([1] * input_ndim) if keepdim else out

        # Two-pass (mean, then sum of squared deviations). The previous
        # E[x^2]-E[x]^2 single-pass formulation suffers catastrophic
        # cancellation in fp32 for large N with non-zero mean (sum ~ N*2.5
        # has ULP ~ 0.5, destroying the ~1e-5 variance signal) and silently
        # returns std = 0. Grid is capped at 1024 programs; each program
        # walks a contiguous CHUNK in BLOCK_N tiles. Tuned on XPU:
        # BLOCK_N=4096 (vs 1024) cuts the per-program loop trip count 4x and
        # speeds up the 2^30-element global reduction ~2.5x (115ms -> 46ms
        # for fp16); GRID=min(max(cdiv(N,16384),256),1024) keeps >= 256
        # programs so even 1M-element reductions get enough parallelism.
        GRID = min(max(triton.cdiv(N, 16384), 256), 1024)
        CHUNK = triton.cdiv(N, GRID)
        BLOCK_N = 4096
        BLOCK_SIZE_REDUCE = 1024
        xc = x.contiguous()
        tmp = torch.empty((GRID,), dtype=torch.float32, device=x.device)
        mean = torch.empty(1, device=x.device, dtype=torch.float32)
        out = torch.empty([], device=x.device, dtype=x.dtype)
        with torch_device_fn.device(x.device):
            _std_partial_sum_kernel[(GRID,)](xc, tmp, N, CHUNK, BLOCK_N)
            _std_finalize_kernel[(1,)](
                tmp, mean, N, effective_correction, GRID, BLOCK_SIZE_REDUCE, False
            )
            _std_partial_sq_kernel[(GRID,)](xc, tmp, N, mean, CHUNK, BLOCK_N)
            _std_finalize_kernel[(1,)](
                tmp, out, N, effective_correction, GRID, BLOCK_SIZE_REDUCE, True
            )
        return out.view([1] * input_ndim) if keepdim else out

    if isinstance(dim, int):
        dim_list = [dim]
    else:
        dim_list = list(dim)
    dim_list_normalized = [d % input_ndim for d in dim_list]

    # Route EVERY dim reduction (single-dim AND multi-dim) through dim_compress so
    # the reduced dims land on the trailing axis => it is always a contiguous
    # (M, N) inner reduction (K == 1). We only ever launch the @libentry-cached
    # _std_dim_kernel_inner. This (a) avoids the giant 2D tile + heuristic-supplied
    # launch param IR explosion of the old _std_fused_dim_kernel path
    # (ir-std-dev5.log = 7.7M lines) and (b) avoids the non_inner (K>1) softmax
    # kernel, which was numerically wrong on XPU (std ~sqrt(K)x too small).
    x_view = dim_compress(x, dim_list_normalized)
    N = 1
    for d in dim_list_normalized:
        N *= original_shape[d]
    M = x.numel() // N

    output_shape_kept = list(original_shape)
    for d in dim_list_normalized:
        output_shape_kept[d] = 1

    if M * N > 0 and (N - effective_correction <= 0):
        final_shape = [
            s for i, s in enumerate(original_shape) if i not in dim_list_normalized
        ]
        return torch.full(
            final_shape if not keepdim else output_shape_kept,
            float("nan"),
            device=x.device,
            dtype=x.dtype,
        )
    if N == 1 and effective_correction == 0.0:
        final_shape = [
            s for i, s in enumerate(original_shape) if i not in dim_list_normalized
        ]
        return torch.zeros(
            final_shape if not keepdim else output_shape_kept,
            device=x.device,
            dtype=x.dtype,
        )

    out = torch.empty(output_shape_kept, device=x.device, dtype=x.dtype)
    if M * N == 0:
        return out.squeeze(dim=tuple(dim_list_normalized)) if not keepdim else out

    _std_dim_dispatch(out.view(-1), x_view, M, N, 1, effective_correction)
    return out.squeeze(dim=tuple(dim_list_normalized)) if not keepdim else out