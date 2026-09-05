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

# XPU override for cdist_backward.
#
# grad_x1[b, i, :] = sum_j w[b, i, j] * (x1[b, i, :] - x2[b, j, :]),
#   w[b, i, j] = grad[b, i, j] / (cdist[b, i, j] + eps).
#
# Design constraints on this Triton-XPU fork (all verified by experiment):
#   * tl.sum(axis=0) on 2D+ tiles is rejected by the backend.
#   * tl.sum(axis=1) transposed layout dies in TritonXPUCoreTiling (uni_sram OOM).
#   * tl.dot / register-tensor indexing (x[k, :]) / n2-split atomics all fail
#     to compile (tl.dot compiles but produces wrong fp32 results), so the n2
#     reduction must stay a serial scalar loop over j.
#   * BLOCK_DIM=128 (even with loop_unroll_factor=1) dies in
#     TritonXPUUnrollControl with "out of resource: uni_sram", so 64 is the
#     widest safe tile.
#
# Hence the two-kernel scheme below:
#   1) _cdist_backward_w_kernel: w = grad / (cdist + eps)  (1D elementwise;
#      BLOCK=128).
#   2) _cdist_backward_kernel: 8-way j-unroll with 8 independent accumulators.
#      Each j-iteration costs one w scalar load + one x2 vector load; the
#      dependent-accumulator chain is what serializes the loop on this
#      single-lane-per-core backend, so independent accumulators batch 8
#      loads+FMAs per iteration and give ~1.5x (small n2) to ~1.7x (large n2)
#      over the plain scalar loop, with at most 7 remaining iterations in the
#      masked-free tail.

import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)

_W_BLOCK = 128
_BLOCK_DIM = 64


@libentry()
@triton.jit
def _cdist_backward_w_kernel(grad_ptr, cdist_ptr, w_ptr, numel, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    off = pid * BLOCK + tl.arange(0, BLOCK)
    mask = off < numel
    g = tl.load(grad_ptr + off, mask=mask, other=0.0).to(tl.float32)
    c = tl.load(cdist_ptr + off, mask=mask, other=1.0).to(tl.float32)
    tl.store(w_ptr + off, g / (c + 1e-12), mask=mask)


@libentry()
@triton.jit
def _cdist_backward_kernel(
    w_ptr,
    x1_ptr,
    x2_ptr,
    grad_x1_ptr,
    batch_size,
    n1,
    n2,
    dim,
    p,
    BLOCK_DIM: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_n1 = tl.program_id(1)
    pid_dim = tl.program_id(2)
    n1_idx = pid_n1

    off_dim = pid_dim * BLOCK_DIM + tl.arange(0, BLOCK_DIM)
    mask_dim = off_dim < dim

    x1_offset = pid_b * n1 * dim + n1_idx * dim + off_dim
    x1 = tl.load(x1_ptr + x1_offset, mask=mask_dim, other=0.0).to(tl.float32)

    acc0 = tl.zeros([BLOCK_DIM], dtype=tl.float32)
    acc1 = tl.zeros([BLOCK_DIM], dtype=tl.float32)
    acc2 = tl.zeros([BLOCK_DIM], dtype=tl.float32)
    acc3 = tl.zeros([BLOCK_DIM], dtype=tl.float32)
    acc4 = tl.zeros([BLOCK_DIM], dtype=tl.float32)
    acc5 = tl.zeros([BLOCK_DIM], dtype=tl.float32)
    acc6 = tl.zeros([BLOCK_DIM], dtype=tl.float32)
    acc7 = tl.zeros([BLOCK_DIM], dtype=tl.float32)

    w_base = pid_b * n1 * n2 + n1_idx * n2
    x2_base = pid_b * n2 * dim

    nmain = n2 // 8 * 8
    for j in range(0, nmain, 8):
        w0 = tl.load(w_ptr + w_base + j).to(tl.float32)
        w1 = tl.load(w_ptr + w_base + j + 1).to(tl.float32)
        w2 = tl.load(w_ptr + w_base + j + 2).to(tl.float32)
        w3 = tl.load(w_ptr + w_base + j + 3).to(tl.float32)
        w4 = tl.load(w_ptr + w_base + j + 4).to(tl.float32)
        w5 = tl.load(w_ptr + w_base + j + 5).to(tl.float32)
        w6 = tl.load(w_ptr + w_base + j + 6).to(tl.float32)
        w7 = tl.load(w_ptr + w_base + j + 7).to(tl.float32)
        x2_0 = tl.load(
            x2_ptr + x2_base + (j + 0) * dim + off_dim, mask=mask_dim, other=0.0
        ).to(tl.float32)
        x2_1 = tl.load(
            x2_ptr + x2_base + (j + 1) * dim + off_dim, mask=mask_dim, other=0.0
        ).to(tl.float32)
        x2_2 = tl.load(
            x2_ptr + x2_base + (j + 2) * dim + off_dim, mask=mask_dim, other=0.0
        ).to(tl.float32)
        x2_3 = tl.load(
            x2_ptr + x2_base + (j + 3) * dim + off_dim, mask=mask_dim, other=0.0
        ).to(tl.float32)
        x2_4 = tl.load(
            x2_ptr + x2_base + (j + 4) * dim + off_dim, mask=mask_dim, other=0.0
        ).to(tl.float32)
        x2_5 = tl.load(
            x2_ptr + x2_base + (j + 5) * dim + off_dim, mask=mask_dim, other=0.0
        ).to(tl.float32)
        x2_6 = tl.load(
            x2_ptr + x2_base + (j + 6) * dim + off_dim, mask=mask_dim, other=0.0
        ).to(tl.float32)
        x2_7 = tl.load(
            x2_ptr + x2_base + (j + 7) * dim + off_dim, mask=mask_dim, other=0.0
        ).to(tl.float32)
        acc0 += w0 * (x1 - x2_0)
        acc1 += w1 * (x1 - x2_1)
        acc2 += w2 * (x1 - x2_2)
        acc3 += w3 * (x1 - x2_3)
        acc4 += w4 * (x1 - x2_4)
        acc5 += w5 * (x1 - x2_5)
        acc6 += w6 * (x1 - x2_6)
        acc7 += w7 * (x1 - x2_7)

    for j in range(nmain, n2):
        wj = tl.load(w_ptr + w_base + j).to(tl.float32)
        x2j = tl.load(
            x2_ptr + x2_base + j * dim + off_dim, mask=mask_dim, other=0.0
        ).to(tl.float32)
        acc0 += wj * (x1 - x2j)

    grad_x1_acc = ((acc0 + acc1) + (acc2 + acc3)) + ((acc4 + acc5) + (acc6 + acc7))

    store_offset = pid_b * n1 * dim + n1_idx * dim + off_dim
    tl.store(grad_x1_ptr + store_offset, grad_x1_acc, mask=mask_dim)


def _cdist_backward(grad, x1, x2, p, cdist):
    logger.debug("GEMS_KUNLUNXIN _cdist_backward")
    assert x1.device == x2.device == grad.device == cdist.device
    assert x1.shape[0] == x2.shape[0] == grad.shape[0] == cdist.shape[0]
    assert x1.shape[2] == x2.shape[2]
    assert x1.dtype in (
        torch.float16,
        torch.bfloat16,
        torch.float32,
    ), f"Unsupported dtype: {x1.dtype}"

    batch_size, n1, dim = x1.shape
    _, n2, _ = x2.shape

    grad = grad.contiguous()
    x1 = x1.contiguous()
    x2 = x2.contiguous()
    cdist = cdist.contiguous()

    if x1.dtype in (torch.float16, torch.bfloat16):
        grad_x1_fp32 = torch.empty_like(x1, dtype=torch.float32)
    else:
        grad_x1_fp32 = torch.empty_like(x1)

    w = torch.empty((batch_size, n1, n2), dtype=torch.float32, device=x1.device)
    numel = w.numel()

    grid = (batch_size, n1, triton.cdiv(dim, _BLOCK_DIM))

    with torch_device_fn.device(x1.device):
        _cdist_backward_w_kernel[(triton.cdiv(numel, _W_BLOCK),)](
            grad,
            cdist,
            w,
            numel,
            BLOCK=_W_BLOCK,
        )
        _cdist_backward_kernel[grid](
            w,
            x1,
            x2,
            grad_x1_fp32,
            batch_size,
            n1,
            n2,
            dim,
            float(p),
            BLOCK_DIM=_BLOCK_DIM,
        )

    if x1.dtype in (torch.float16, torch.bfloat16):
        return grad_x1_fp32.to(x1.dtype)
    return grad_x1_fp32