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
def _dequant_int4_kernel(
    mat2_ptr,
    scale_ptr,
    zero_ptr,
    out_ptr,
    N,
    K,
    K2,
    G,
    qGroupSize,
    stride_mn,
    stride_mk,
    stride_sg,
    stride_sn,
    stride_zg,
    stride_zn,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < K * N

    # out is row-major (K, N): offs = k * N + n
    k = offs // N
    n = offs % N

    byte_idx = k // 2
    is_high = (k % 2) == 1
    group = k // qGroupSize

    packed = tl.load(
        mat2_ptr + n * stride_mn + byte_idx * stride_mk, mask=mask, other=0
    ).to(tl.int32)
    low = packed & 0xF
    high = (packed >> 4) & 0xF
    q = tl.where(is_high, high, low).to(tl.float32)

    scale = tl.load(
        scale_ptr + group * stride_sg + n * stride_sn, mask=mask, other=1.0
    ).to(tl.float32)
    zero = tl.load(
        zero_ptr + group * stride_zg + n * stride_zn, mask=mask, other=0.0
    ).to(tl.float32)

    w = (q - zero) * scale
    tl.store(out_ptr + offs, w, mask=mask)


def _weight_int4pack_mm_with_scales_and_zeros(
    A: torch.Tensor,
    mat2: torch.Tensor,
    qGroupSize: int,
    qScale: torch.Tensor,
    qZeros: torch.Tensor,
) -> torch.Tensor:
    logger.debug("GEMS_KUNLUNXIN _WEIGHT_INT4PACK_MM_WITH_SCALES_AND_ZEROS")

    M, K = A.shape
    N = mat2.shape[0]
    K2 = mat2.shape[1]

    assert K == K2 * 2, f"K ({K}) must equal 2 * K2 ({K2}) for packed int4 format"
    assert K % qGroupSize == 0, f"K ({K}) must be divisible by qGroupSize ({qGroupSize})"
    G = K // qGroupSize
    assert qScale.shape == (G, N), f"qScale shape {qScale.shape} != expected ({G}, {N})"
    assert qZeros.shape == (G, N), f"qZeros shape {qZeros.shape} != expected ({G}, {N})"
    assert mat2.dtype == torch.uint8, f"mat2 must be uint8, got {mat2.dtype}"
    assert A.is_floating_point(), "A must be floating point"

    import flag_gems

    mat2 = mat2.contiguous()
    qScale = qScale.contiguous()
    qZeros = qZeros.contiguous()

    # The native fused kernel decodes both nibbles into half-width fragments and
    # folds them into tl.dot via join/permute/reshape, which fails to lower in the
    # XPU3 TritonToLinalg pipeline. Fully dequantize the weight to an fp32 (K, N)
    # matrix with a flat 1D pointwise kernel (no 2D tile, no tl.dot), then route the
    # GEMM through the backend fp32 mm. int4 values (0..15) and per-group
    # scales/zeros are exact in fp32, reproducing the reference up to accumulation.
    wdq = torch.empty((K, N), dtype=torch.float32, device=A.device)
    BLOCK = 1024
    grid_w = (triton.cdiv(K * N, BLOCK),)
    with torch_device_fn.device(A.device):
        _dequant_int4_kernel[grid_w](
            mat2,
            qScale,
            qZeros,
            wdq,
            N,
            K,
            K2,
            G,
            qGroupSize,
            mat2.stride(0),
            mat2.stride(1),
            qScale.stride(0),
            qScale.stride(1),
            qZeros.stride(0),
            qZeros.stride(1),
            BLOCK=BLOCK,
        )

    x2d = A.reshape(M, K).to(torch.float32)
    acc = flag_gems.mm(x2d, wdq)  # (M, N) fp32
    return acc.to(A.dtype)
