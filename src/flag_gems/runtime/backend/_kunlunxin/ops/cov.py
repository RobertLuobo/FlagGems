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

from .mm import mm as _xpu_mm

logger = logging.getLogger(__name__)


@triton.jit
def _cov_weight_kernel(
    wf_ptr,
    wa_ptr,
    w_ptr,
    n_cols,
    HAS_WF: tl.constexpr,
    HAS_WA: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    mask = offs < n_cols
    acc = tl.full([BLOCK_M], 1.0, dtype=tl.float32)
    if HAS_WF:
        wf = tl.load(wf_ptr + offs, mask=mask, other=0).to(tl.float32)
        acc = acc * wf
    if HAS_WA:
        wa = tl.load(wa_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        acc = acc * wa
    tl.store(w_ptr + offs, acc, mask=mask)


@triton.jit
def _cov_mean_kernel(
    x_ptr,
    w_ptr,
    mean_ptr,
    n_rows,
    n_cols,
    w_sum,
    stride_xn,
    stride_xm,
    HAS_W: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    row = tl.program_id(0)
    acc = tl.zeros([BLOCK_M], dtype=tl.float32)
    for m_start in range(0, n_cols, BLOCK_M):
        offs = m_start + tl.arange(0, BLOCK_M)
        mask = offs < n_cols
        x = tl.load(x_ptr + row * stride_xn + offs * stride_xm, mask=mask, other=0.0)
        x = x.to(tl.float32)
        if HAS_W:
            w = tl.load(w_ptr + offs, mask=mask, other=0.0).to(tl.float32)
            x = x * w
        acc += x
    mean = tl.sum(acc) / w_sum
    tl.store(mean_ptr + row, mean)


@triton.jit
def _cov_build_kernel(
    x_ptr,
    w_ptr,
    mean_ptr,
    a_ptr,
    b_ptr,
    n_rows,
    n_cols,
    inv_fact,
    stride_xn,
    stride_xm,
    stride_an,
    stride_am,
    HAS_W: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    # Materialise the two fp32 operands of the covariance GEMM:
    #   b[i, k] = x[i, k] - mean[i]                     (plain centered rows)
    #   a[i, k] = (x[i, k] - mean[i]) * w[k] * inv_fact (weighted + 1/fact scaled)
    # so that mm(a, b.T)[i, j] = sum_k w[k]*(xi-mi)*(xj-mj)/fact == torch.cov.
    # Folding w and 1/fact into one operand keeps the normalisation exact in
    # fp32 and lets the matmul itself produce the final covariance values; the
    # cross-product reduction is then done by the vendor fp32 mm (allow_tf32
    # False), avoiding the tl.dot/tl.trans + tf32x3 path that fails to lower on
    # TritonXPU.
    row = tl.program_id(0)
    m = tl.load(mean_ptr + row).to(tl.float32)
    for m_start in range(0, n_cols, BLOCK_M):
        offs = m_start + tl.arange(0, BLOCK_M)
        mask = offs < n_cols
        x = tl.load(
            x_ptr + row * stride_xn + offs * stride_xm, mask=mask, other=0.0
        ).to(tl.float32)
        centered = x - m
        tl.store(b_ptr + row * stride_an + offs * stride_am, centered, mask=mask)
        a = centered * inv_fact
        if HAS_W:
            w = tl.load(w_ptr + offs, mask=mask, other=0.0).to(tl.float32)
            a = a * w
        tl.store(a_ptr + row * stride_an + offs * stride_am, a, mask=mask)


@triton.jit
def _cov_cast_kernel(
    src_ptr,
    dst_ptr,
    n,
    BLOCK: tl.constexpr,
):
    # fp32 -> out_dtype copy done in-kernel so the final cast never dispatches
    # to the registered _to_copy override (unstable across dtypes/shapes).
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    v = tl.load(src_ptr + offs, mask=mask, other=0.0)
    tl.store(dst_ptr + offs, v, mask=mask)


def cov(
    input: torch.Tensor,
    *,
    correction: int = 1,
    fweights: torch.Tensor = None,
    aweights: torch.Tensor = None,
) -> torch.Tensor:
    logger.debug("GEMS_KUNLUNXIN COV")

    if input.dim() > 2:
        raise ValueError(
            f"cov(): expected input to have two or fewer dimensions but got {input.dim()}"
        )

    x = input.view(1, -1) if input.dim() < 2 else input
    n_rows, n_cols = x.shape
    device = x.device
    out_dtype = input.dtype
    single_variable = n_rows == 1

    has_wf = fweights is not None
    has_wa = aweights is not None
    has_w = has_wf or has_wa
    w = None
    wa = None
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
    inv_fact = 0.0 if fact == 0.0 else 1.0 / fact

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

    # Build the two fp32 GEMM operands (contiguous row-major), then route the
    # (X-mean) @ (X-mean)^T cross-product through the vendor fp32 mm.
    a_mat = torch.empty(n_rows, n_cols, device=device, dtype=torch.float32)
    b_mat = torch.empty(n_rows, n_cols, device=device, dtype=torch.float32)
    _cov_build_kernel[(n_rows,)](
        x,
        w,
        mean,
        a_mat,
        b_mat,
        n_rows,
        n_cols,
        inv_fact,
        x.stride(0),
        x.stride(1),
        a_mat.stride(0),
        a_mat.stride(1),
        HAS_W=has_w,
        BLOCK_M=BLOCK_M,
    )

    # c[i, j] = sum_k a[i, k] * b[j, k] = (a_mat) @ (b_mat)^T, exact fp32.
    cov_fp32 = _xpu_mm(a_mat, b_mat.t())

    if out_dtype == torch.float32:
        cov_mat = cov_fp32
    else:
        cov_mat = torch.empty(n_rows, n_rows, device=device, dtype=out_dtype)
        total = n_rows * n_rows
        CAST_BLOCK = 1024
        grid_c = triton.cdiv(total, CAST_BLOCK)
        _cov_cast_kernel[(grid_c,)](
            cov_fp32.reshape(-1),
            cov_mat.reshape(-1),
            total,
            BLOCK=CAST_BLOCK,
        )

    if single_variable:
        return cov_mat.view(())
    return cov_mat
