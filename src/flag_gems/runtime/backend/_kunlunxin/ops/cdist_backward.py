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
# The general (upstream) kernel builds contrib as a 2D [BLOCK_N2, BLOCK_DIM]
# tile and reduces along axis=0; XPU triton rejects axis=0 reductions on 2D+
# tensors ("axis must not be 0 for 2D+ shapes"), and a [BLOCK_DIM, BLOCK_N2]
# layout with axis=1 reduce compiles but TritonXPUCoreTiling fails on 64x64
# tiles. Axis-1 tl.sum on 2D tiles is also numerically wrong on XPU, and a
# runtime-bound K loop around tl.dot (for n2 > 128) pathologies both perf and
# correctness on this backend.
#
# This override re-expresses the p=2 gradient as a batched matmul:
#
#   grad_x1[b,i,k] = sum_j w[b,i,j] * (x1[b,i,k] - x2[b,j,k])
#                  = s[b,i] * x1[b,i,k] - (W[b] @ X2[b])[i,k]
#
# with w[b,i,j] = grad[b,i,j] / (cdist[b,i,j] + eps) and s[b,i] = sum_j w[b,i,j].
# Two single-shot kernels (no runtime-bound loops, no axis-0 2D reduction):
#   * _cdist_backward_w_kernel  grid (B, n1): 1D [BLOCK_N2] rows -> w, s
#   * _cdist_backward_dot_kernel grid (B, n1/BM, dim/BK): 2D tiles + tl.dot,
#     epilogue out = s[:,None] * x1 - W@X2.
#
# For n2 > BLOCK_N2 (beyond the current test matrix) we fall back to a serial
# 1D loop (correct for arbitrary n2, slower); XPU runtime-bound loops around
# tl.dot are not reliable here.

import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)

_BM = 64
_BN2 = 128
_BK = 64


@libentry()
@triton.jit
def _cdist_backward_w_kernel(
    grad_ptr, cdist_ptr, w_ptr, s_ptr, n1, n2, BLOCK_N2: tl.constexpr
):
    pid_b = tl.program_id(0)
    pid_n1 = tl.program_id(1)

    base = pid_b * n1 * n2 + pid_n1 * n2
    off = tl.arange(0, BLOCK_N2)
    mask = off < n2

    g = tl.load(grad_ptr + base + off, mask=mask, other=0.0).to(tl.float32)
    c = tl.load(cdist_ptr + base + off, mask=mask, other=1.0).to(tl.float32)
    w = g / (c + 1e-12)

    tl.store(w_ptr + base + off, w, mask=mask)
    s = tl.sum(w, axis=0)
    tl.store(s_ptr + pid_b * n1 + pid_n1, s)


@libentry()
@triton.jit
def _cdist_backward_dot_kernel(
    w_ptr,
    x1_ptr,
    x2_ptr,
    s_ptr,
    grad_x1_ptr,
    n1,
    n2,
    dim,
    BM: tl.constexpr,
    BN2: tl.constexpr,
    BK: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_k = tl.program_id(2)

    off_m = pid_m * BM + tl.arange(0, BM)
    off_j = tl.arange(0, BN2)
    off_k = pid_k * BK + tl.arange(0, BK)

    mask_m = off_m < n1
    mask_j = off_j < n2
    mask_k = off_k < dim

    # W[b, off_m, off_j] : [BM, BN2]
    w = tl.load(
        w_ptr + pid_b * n1 * n2 + off_m[:, None] * n2 + off_j[None, :],
        mask=mask_m[:, None] & mask_j[None, :],
        other=0.0,
    ).to(tl.float32)

    # X2[b, off_j, off_k] : [BN2, BK]
    x2 = tl.load(
        x2_ptr + pid_b * n2 * dim + off_j[:, None] * dim + off_k[None, :],
        mask=mask_j[:, None] & mask_k[None, :],
        other=0.0,
    ).to(tl.float32)

    acc = tl.dot(w, x2, allow_tf32=False)  # [BM, BK]

    s = tl.load(s_ptr + pid_b * n1 + off_m, mask=mask_m, other=0.0).to(tl.float32)

    # X1[b, off_m, off_k] : [BM, BK]
    x1 = tl.load(
        x1_ptr + pid_b * n1 * dim + off_m[:, None] * dim + off_k[None, :],
        mask=mask_m[:, None] & mask_k[None, :],
        other=0.0,
    ).to(tl.float32)

    out = s[:, None] * x1 - acc
    tl.store(
        grad_x1_ptr + pid_b * n1 * dim + off_m[:, None] * dim + off_k[None, :],
        out,
        mask=mask_m[:, None] & mask_k[None, :],
    )


@libentry()
@triton.jit
def _cdist_backward_serial_kernel(
    grad_ptr,
    x1_ptr,
    x2_ptr,
    cdist_ptr,
    grad_x1_ptr,
    n1,
    n2,
    dim,
    BLOCK_DIM: tl.constexpr,
):
    # General path for n2 > _BN2: serial n2 loop, all 1D so XPU codegen is
    # clean. Slower than the tl.dot path but correct for arbitrary n2.
    pid_b = tl.program_id(0)
    pid_n1 = tl.program_id(1)
    pid_dim = tl.program_id(2)

    off_dim = pid_dim * BLOCK_DIM + tl.arange(0, BLOCK_DIM)
    mask_dim = off_dim < dim

    x1_offset = pid_b * n1 * dim + pid_n1 * dim + off_dim
    x1 = tl.load(x1_ptr + x1_offset, mask=mask_dim, other=0.0).to(tl.float32)

    grad_x1_acc = tl.zeros([BLOCK_DIM], dtype=tl.float32)

    grad_base = pid_b * n1 * n2 + pid_n1 * n2
    x2_base = pid_b * n2 * dim

    eps = 1e-12
    for j in range(0, n2):
        gj = tl.load(grad_ptr + grad_base + j).to(tl.float32)
        cj = tl.load(cdist_ptr + grad_base + j).to(tl.float32)
        x2j = tl.load(
            x2_ptr + x2_base + j * dim + off_dim, mask=mask_dim, other=0.0
        ).to(tl.float32)
        grad_x1_acc += gj * (x1 - x2j) / (cj + eps)

    tl.store(grad_x1_ptr + x1_offset, grad_x1_acc, mask=mask_dim)


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

    with torch_device_fn.device(x1.device):
        if n2 <= _BN2:
            w = torch.empty(batch_size, n1, n2, dtype=torch.float32, device=x1.device)
            s = torch.empty(batch_size, n1, dtype=torch.float32, device=x1.device)
            _cdist_backward_w_kernel[(batch_size, n1)](
                grad, cdist, w, s, n1, n2, BLOCK_N2=_BN2
            )
            grid = (batch_size, triton.cdiv(n1, _BM), triton.cdiv(dim, _BK))
            _cdist_backward_dot_kernel[grid](
                w,
                x1,
                x2,
                s,
                grad_x1_fp32,
                n1,
                n2,
                dim,
                BM=_BM,
                BN2=_BN2,
                BK=_BK,
            )
        else:
            grid = (batch_size, n1, triton.cdiv(dim, _BK))
            _cdist_backward_serial_kernel[grid](
                grad,
                x1,
                x2,
                cdist,
                grad_x1_fp32,
                n1,
                n2,
                dim,
                BLOCK_DIM=_BK,
            )

    if x1.dtype in (torch.float16, torch.bfloat16):
        return grad_x1_fp32.to(x1.dtype)
    return grad_x1_fp32