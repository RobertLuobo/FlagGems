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
def _corrcoef_mean_kernel(
    x_ptr, mean_ptr, n_rows, n_cols, stride_xn, stride_xm, BLOCK_M: tl.constexpr
):
    row = tl.program_id(0)
    acc = tl.zeros([BLOCK_M], dtype=tl.float32)
    for m_start in range(0, n_cols, BLOCK_M):
        offs = m_start + tl.arange(0, BLOCK_M)
        mask = offs < n_cols
        x = tl.load(x_ptr + row * stride_xn + offs * stride_xm, mask=mask, other=0.0)
        x = x.to(tl.float32)
        acc += x
    mean = tl.sum(acc) / n_cols
    tl.store(mean_ptr + row, mean)


@triton.jit
def _corrcoef_cov_kernel(
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
    pid_i = tl.program_id(0)
    pid_j = tl.program_id(1)

    mi = tl.load(mean_ptr + pid_i)
    mj = tl.load(mean_ptr + pid_j)

    acc = tl.zeros([BLOCK_M], dtype=tl.float32)
    for m_start in range(0, n_cols, BLOCK_M):
        offs_m = m_start + tl.arange(0, BLOCK_M)
        mask_m = offs_m < n_cols

        xi = tl.load(
            x_ptr + pid_i * stride_xn + offs_m * stride_xm, mask=mask_m, other=0.0
        ).to(tl.float32)
        xj = tl.load(
            x_ptr + pid_j * stride_xn + offs_m * stride_xm, mask=mask_m, other=0.0
        ).to(tl.float32)

        prod = (xi - mi) * (xj - mj)
        prod = tl.where(mask_m, prod, 0.0)
        acc += prod

    c = tl.sum(acc) / (n_cols - 1)
    tl.store(c_ptr + pid_i * stride_cn + pid_j * stride_cm, c)


@triton.jit
def _corrcoef_std_norm_kernel(
    c_ptr,
    out_ptr,
    n_rows,
    stride_cn,
    stride_cm,
    stride_on,
    stride_om,
):
    pid_i = tl.program_id(0)
    pid_j = tl.program_id(1)

    si = tl.load(c_ptr + pid_i * stride_cn + pid_i * stride_cm)
    sj = tl.load(c_ptr + pid_j * stride_cn + pid_j * stride_cm)
    si = tl.sqrt(si)
    sj = tl.sqrt(sj)

    c = tl.load(c_ptr + pid_i * stride_cn + pid_j * stride_cm)

    denom = si * sj
    corr = c / denom

    tl.store(out_ptr + pid_i * stride_on + pid_j * stride_om, corr)


def corrcoef(input: torch.Tensor) -> torch.Tensor:
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
    _corrcoef_cov_kernel[(n_rows, n_rows)](
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

    out = torch.empty(n_rows, n_rows, device=device, dtype=out_dtype)
    _corrcoef_std_norm_kernel[(n_rows, n_rows)](
        cov,
        out,
        n_rows,
        cov.stride(0),
        cov.stride(1),
        out.stride(0),
        out.stride(1),
    )

    return out
