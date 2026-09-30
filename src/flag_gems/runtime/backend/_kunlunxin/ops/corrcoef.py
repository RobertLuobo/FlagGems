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

from flag_gems.ops.corrcoef import _corrcoef_mean_kernel

logger = logging.getLogger(__name__)


@triton.jit
def _corrcoef_cov_pair_kernel(
    x_ptr,
    mean_ptr,
    c_ptr,
    n_rows,
    n_cols,
    stride_xn,
    stride_xm,
    stride_cn,
    stride_cm,
    BLOCK_M: tl.constexpr,
):
    # tl.dot is numerically wrong on XPU3 for these tiles; accumulate with tl.sum.
    i = tl.program_id(0)
    j = tl.program_id(1)
    mi = tl.load(mean_ptr + i)
    mj = tl.load(mean_ptr + j)
    acc = tl.zeros([BLOCK_M], dtype=tl.float32)
    for m_start in range(0, n_cols, BLOCK_M):
        offs = m_start + tl.arange(0, BLOCK_M)
        mask = offs < n_cols
        xi = tl.load(
            x_ptr + i * stride_xn + offs * stride_xm, mask=mask, other=0.0
        ).to(tl.float32)
        xj = tl.load(
            x_ptr + j * stride_xn + offs * stride_xm, mask=mask, other=0.0
        ).to(tl.float32)
        prod = (xi - mi) * (xj - mj)
        prod = tl.where(mask, prod, 0.0)
        acc += prod
    c = tl.sum(acc) / (n_cols - 1)
    tl.store(c_ptr + i * stride_cn + j * stride_cm, c)


@triton.jit
def _corrcoef_diag_std_kernel(
    c_ptr, std_ptr, n_rows, stride_cn, stride_cm, BLOCK_N: tl.constexpr
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = offs < n_rows
    diag = tl.load(c_ptr + offs * stride_cn + offs * stride_cm, mask=mask, other=0.0)
    tl.store(std_ptr + offs, tl.sqrt(diag), mask=mask)


@triton.jit
def _corrcoef_norm_kernel(
    c_ptr,
    std_ptr,
    out_ptr,
    n_rows,
    stride_cn,
    stride_cm,
    stride_on,
    stride_om,
    BLOCK_N: tl.constexpr,
):
    pid_i = tl.program_id(0)
    pid_j = tl.program_id(1)

    offs_i = pid_i * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_j = pid_j * BLOCK_N + tl.arange(0, BLOCK_N)
    # Compute bounds masks in broadcast (2D) shape: a 1D ``offs < n`` broadcast
    # to 2D trips the XPU3 legalizer (arith.cmpi operand type mismatch).
    mask_i = offs_i[:, None] < n_rows
    mask_j = offs_j[None, :] < n_rows
    mask_ij = mask_i & mask_j

    si = tl.load(std_ptr + offs_i[:, None], mask=mask_i, other=0.0)
    sj = tl.load(std_ptr + offs_j[None, :], mask=mask_j, other=0.0)

    c = tl.load(
        c_ptr + offs_i[:, None] * stride_cn + offs_j[None, :] * stride_cm,
        mask=mask_ij,
        other=0.0,
    )

    corr = c / (si * sj)

    tl.store(
        out_ptr + offs_i[:, None] * stride_on + offs_j[None, :] * stride_om,
        corr,
        mask=mask_ij,
    )


def corrcoef(input: torch.Tensor) -> torch.Tensor:
    logger.debug("GEMS_KUNLUNXIN CORRCOEF")
    logger.debug("GEMS CORRCOEF")

    if input.dim() < 2 or input.shape[0] == 1:
        return torch.ones((), dtype=input.dtype, device=input.device)

    x = input.contiguous()
    n_rows, n_cols = x.shape
    device = x.device
    out_dtype = x.dtype

    mean = torch.empty(n_rows, device=device, dtype=torch.float32)
    BLOCK_M = 128
    _corrcoef_mean_kernel[(n_rows,)](
        x, mean, n_rows, n_cols, x.stride(0), x.stride(1), BLOCK_M=BLOCK_M
    )

    cov = torch.empty(n_rows, n_rows, device=device, dtype=torch.float32)
    _corrcoef_cov_pair_kernel[(n_rows, n_rows)](
        x,
        mean,
        cov,
        n_rows,
        n_cols,
        x.stride(0),
        x.stride(1),
        cov.stride(0),
        cov.stride(1),
        BLOCK_M=BLOCK_M,
    )

    # Per-variable std = sqrt of the covariance diagonal (1D kernel).
    std = torch.empty(n_rows, device=device, dtype=torch.float32)
    BLOCK_N = 32
    grid_std = triton.cdiv(n_rows, BLOCK_N)
    _corrcoef_diag_std_kernel[(grid_std,)](
        cov, std, n_rows, cov.stride(0), cov.stride(1), BLOCK_N=BLOCK_N
    )

    out = torch.empty(n_rows, n_rows, device=device, dtype=out_dtype)
    BLOCK_N_OUT = 32
    grid_out = triton.cdiv(n_rows, BLOCK_N_OUT)
    _corrcoef_norm_kernel[(grid_out, grid_out)](
        cov,
        std,
        out,
        n_rows,
        cov.stride(0),
        cov.stride(1),
        out.stride(0),
        out.stride(1),
        BLOCK_N=BLOCK_N_OUT,
    )

    return out
