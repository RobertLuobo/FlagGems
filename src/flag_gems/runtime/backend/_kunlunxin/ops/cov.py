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

from flag_gems.ops.cov import _cov_mean_kernel, _cov_weight_kernel

logger = logging.getLogger(__name__)


@triton.jit
def _cov_pair_kernel(
    x_ptr,
    w_ptr,
    mean_ptr,
    c_ptr,
    n_rows,
    n_cols,
    fact,
    stride_xn,
    stride_xm,
    stride_cn,
    stride_cm,
    HAS_W: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    # tl.dot is numerically wrong on XPU3 for these tiles; accumulate the
    # (weighted) centered cross-products with tl.sum instead. Each program
    # computes a single covariance entry c[i, j].
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
        if HAS_W:
            w = tl.load(w_ptr + offs, mask=mask, other=0.0).to(tl.float32)
            prod = prod * w
        prod = tl.where(mask, prod, 0.0)
        acc += prod
    c = tl.sum(acc) / fact
    tl.store(c_ptr + i * stride_cn + j * stride_cm, c)


def cov(
    input: torch.Tensor,
    *,
    correction: int = 1,
    fweights: torch.Tensor = None,
    aweights: torch.Tensor = None,
) -> torch.Tensor:
    logger.debug("GEMS_KUNLUNXIN COV")
    logger.debug("GEMS COV")

    if input.dim() > 2:
        raise ValueError(
            f"cov(): expected input to have two or fewer dimensions but got {input.dim()}"
        )

    x = input.view(1, -1) if input.dim() < 2 else input
    x = x.contiguous()
    n_rows, n_cols = x.shape
    device = x.device
    out_dtype = input.dtype
    single_variable = n_rows == 1

    has_wf = fweights is not None
    has_wa = aweights is not None
    has_w = has_wf or has_wa
    w = None
    if has_w:
        wf = fweights.contiguous() if has_wf else None
        wa = (
            aweights.to(device=device, dtype=torch.float32).contiguous()
            if has_wa
            else None
        )
        w = torch.empty(n_cols, device=device, dtype=torch.float32)
        WEIGHT_BLOCK = 1024
        grid_w = triton.cdiv(n_cols, WEIGHT_BLOCK)
        _cov_weight_kernel[(grid_w,)](
            wf,
            wa,
            w,
            n_cols,
            HAS_WF=has_wf,
            HAS_WA=has_wa,
            BLOCK_M=WEIGHT_BLOCK,
        )

    if has_w:
        w_sum = float(w.sum().item())
        if not has_wa:
            fact = w_sum - correction
        else:
            fact = w_sum - correction * float((w * wa).sum().item()) / w_sum
    else:
        w_sum = float(n_cols)
        fact = float(n_cols - correction)
    if fact < 0.0:
        fact = 0.0

    mean = torch.empty(n_rows, device=device, dtype=torch.float32)
    BLOCK_M = 128
    _cov_mean_kernel[(n_rows,)](
        x,
        w,
        mean,
        n_rows,
        n_cols,
        w_sum,
        x.stride(0),
        x.stride(1),
        HAS_W=has_w,
        BLOCK_M=BLOCK_M,
    )

    cov_mat = torch.empty(n_rows, n_rows, device=device, dtype=out_dtype)
    _cov_pair_kernel[(n_rows, n_rows)](
        x,
        w,
        mean,
        cov_mat,
        n_rows,
        n_cols,
        fact,
        x.stride(0),
        x.stride(1),
        cov_mat.stride(0),
        cov_mat.stride(1),
        HAS_W=has_w,
        BLOCK_M=BLOCK_M,
    )

    if single_variable:
        return cov_mat.view(())
    return cov_mat
