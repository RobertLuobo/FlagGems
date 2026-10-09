# Copyright 2026, The FlagOS Contributors.
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

from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as ext

logger = logging.getLogger(__name__)


@libentry()
@triton.jit
def vander_real_kernel(
    x_ptr,
    out_ptr,
    M,
    N,
    x_stride,
    increasing: tl.constexpr,
    COMPUTE_DTYPE: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    pid = ext.program_id(0)
    rows = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = rows < M
    x_val = tl.load(x_ptr + rows * x_stride, mask=mask_m, other=0).to(COMPUTE_DTYPE)

    acc = tl.full([BLOCK_M], 1, dtype=COMPUTE_DTYPE)
    for k in range(0, N):
        store_col = k if increasing else N - 1 - k
        out_off = rows * N + store_col
        tl.store(out_ptr + out_off, acc, mask=mask_m)
        acc = acc * x_val


@libentry()
@triton.jit
def vander_complex_kernel(
    xr_ptr,
    xi_ptr,
    outr_ptr,
    outi_ptr,
    M,
    N,
    x_stride,
    increasing: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    pid = ext.program_id(0)
    rows = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = rows < M
    xr = tl.load(xr_ptr + rows * x_stride, mask=mask_m, other=0.0).to(tl.float32)
    xi = tl.load(xi_ptr + rows * x_stride, mask=mask_m, other=0.0).to(tl.float32)

    accr = tl.full([BLOCK_M], 1.0, dtype=tl.float32)
    acci = tl.full([BLOCK_M], 0.0, dtype=tl.float32)
    for k in range(0, N):
        store_col = k if increasing else N - 1 - k
        out_off = rows * N + store_col
        tl.store(outr_ptr + out_off, accr, mask=mask_m)
        tl.store(outi_ptr + out_off, acci, mask=mask_m)
        new_r = accr * xr - acci * xi
        new_i = accr * xi + acci * xr
        accr = new_r
        acci = new_i


def vander(x, N=None, increasing=False):
    logger.debug("GEMS_KUNLUNXIN VANDER")

    if x.ndim != 1:
        raise RuntimeError("x must be a one-dimensional tensor.")

    if N is not None and N < 0:
        raise RuntimeError("N must be non-negative.")

    M = x.shape[0]
    if N is None:
        N = M

    if x.is_complex():
        out = torch.empty((M, N), dtype=x.dtype, device=x.device)
        if M == 0 or N == 0:
            return out
        xc = x.contiguous()
        xr = xc.real.contiguous()
        xi = xc.imag.contiguous()
        outr = torch.empty((M, N), dtype=xr.dtype, device=x.device)
        outi = torch.empty((M, N), dtype=xr.dtype, device=x.device)
        BLOCK_M = 32
        grid = (triton.cdiv(M, BLOCK_M),)
        vander_complex_kernel[grid](
            xr,
            xi,
            outr,
            outi,
            M,
            N,
            xr.stride(0),
            increasing=increasing,
            BLOCK_M=BLOCK_M,
        )
        out.real.copy_(outr)
        out.imag.copy_(outi)
        return out

    out_dtype = torch.promote_types(x.dtype, torch.long)
    out = torch.empty((M, N), dtype=out_dtype, device=x.device)

    if M == 0 or N == 0:
        return out

    x_stride = x.stride(0)
    BLOCK_M = 32
    grid = (triton.cdiv(M, BLOCK_M),)

    if x.is_floating_point():
        compute_dtype = tl.float64 if x.dtype == torch.float64 else tl.float32
    else:
        compute_dtype = tl.int64

    vander_real_kernel[grid](
        x,
        out,
        M,
        N,
        x_stride,
        increasing=increasing,
        COMPUTE_DTYPE=compute_dtype,
        BLOCK_M=BLOCK_M,
    )
    return out
