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

from flag_gems.runtime import torch_device_fn

logger = logging.getLogger(__name__)


@triton.jit
def _dequant_kernel(
    b_ptr,
    out_ptr,
    N,
    K,
    stride_bn,
    stride_bk,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < K * N
    k = offs // N
    n = offs % N
    b = tl.load(b_ptr + n * stride_bn + k * stride_bk, mask=mask, other=0).to(tl.int32)
    tl.store(out_ptr + offs, b.to(tl.float32), mask=mask)


@triton.jit
def _epilogue_kernel(
    acc_ptr,
    scale_ptr,
    out_ptr,
    M,
    N,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < M * N
    n = offs % N
    acc = tl.load(acc_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    s = tl.load(scale_ptr + n, mask=mask, other=0.0).to(tl.float32)
    acc = acc * s
    tl.store(out_ptr + offs, acc.to(out_ptr.dtype.element_ty), mask=mask)


def weight_int8pack_mm(A, B, scales):
    logger.debug("GEMS_KUNLUNXIN WEIGHT_INT8PACK_MM")

    assert (
        A.shape[1] == B.shape[1]
    ), f"incompatible K dimensions: A.shape[1]={A.shape[1]}, B.shape[1]={B.shape[1]}"
    assert (
        B.shape[0] == scales.shape[0]
    ), f"incompatible N dimensions: B.shape[0]={B.shape[0]}, scales.shape[0]={scales.shape[0]}"
    assert B.dtype == torch.int8, f"B must be int8, got {B.dtype}"

    import flag_gems

    M, K = A.shape
    N = B.shape[0]

    B = B.contiguous()
    scales = scales.reshape(-1).contiguous()

    # Route the GEMM through the backend fp32 mm: the native fused kernel feeds a
    # masked 2D load into tl.dot, which miscompiles on XPU3 and returns all-NaN.
    # int8 weights (-128..127) are exact in fp32, so dequant + fp32 mm reproduces
    # the reference (int8 -> half -> matmul) exactly up to accumulation precision.
    wdq = torch.empty((K, N), dtype=torch.float32, device=A.device)
    BLOCK = 1024
    grid_w = (triton.cdiv(K * N, BLOCK),)
    with torch_device_fn.device(A.device):
        _dequant_kernel[grid_w](
            B,
            wdq,
            N,
            K,
            B.stride(0),
            B.stride(1),
            BLOCK=BLOCK,
        )

    x2d = A.reshape(M, K).to(torch.float32)
    acc = flag_gems.mm(x2d, wdq)  # (M, N) fp32

    out = torch.empty((M, N), dtype=A.dtype, device=A.device)
    grid_e = (triton.cdiv(M * N, BLOCK),)
    with torch_device_fn.device(A.device):
        _epilogue_kernel[grid_e](
            acc,
            scales,
            out,
            M,
            N,
            BLOCK=BLOCK,
        )
    return out
