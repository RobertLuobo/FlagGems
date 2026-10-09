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

from flag_gems.runtime import torch_device_fn

from .weight_int4pack_mm_with_scales_and_zeros import (
    _BK,
    _BM,
    _BN,
    _int4pack_mm_gemm_kernel,
    _pad_rows_cols,
)

logger = logging.getLogger(__name__)


def weight_int4pack_mm(
    A: torch.Tensor,
    mat2: torch.Tensor,
    qGroupSize: int,
    qScaleAndZeros: torch.Tensor,
) -> torch.Tensor:
    """Int4 packed weight matrix multiplication (XPU overlay).

    Computes C = A @ dequant(W).T with the same conventions as the generic
    implementation:

      - A:              activation tensor (M, K), floating point.
      - mat2:           int4 packed weights (N, K//2), uint8; low nibble = even
                        column, high nibble = odd column.
      - qGroupSize:     quantization group size along K; K % qGroupSize == 0.
      - qScaleAndZeros: (K//qGroupSize, N, 2), last dim [scale, zero].

    The int-unpack fed into tl.dot mis-lowers on TritonXPU (the generic kernel
    collapses the join/permute/reshape interleave and overflows uni_sram), so
    the int4 unpack and dequant are materialized on device and only the fp32
    GEMM contraction is kept in the Triton kernel, reusing the proven vendor
    GEMM kernel from the scales-and-zeros overlay.
    """
    logger.debug("GEMS_KUNLUNXIN _WEIGHT_INT4PACK_MM")

    M, K = A.shape
    N = mat2.shape[0]
    K2 = mat2.shape[1]

    assert K == K2 * 2, f"K ({K}) must equal 2 * K2 ({K2}) for packed int4 format"
    assert (
        K % qGroupSize == 0
    ), f"K ({K}) must be divisible by qGroupSize ({qGroupSize})"
    assert qGroupSize % 2 == 0, f"qGroupSize ({qGroupSize}) must be even"
    assert mat2.dtype == torch.uint8, f"mat2 must be uint8, got {mat2.dtype}"
    assert A.dtype.is_floating_point, f"A must be a floating point type, got {A.dtype}"

    G = K // qGroupSize
    assert qScaleAndZeros.shape == (
        G,
        N,
        2,
    ), f"qScaleAndZeros shape {tuple(qScaleAndZeros.shape)} != expected ({G}, {N}, 2)"

    out_dtype = A.dtype
    A = A.contiguous()
    mat2 = mat2.contiguous()
    qScaleAndZeros = qScaleAndZeros.contiguous()

    qScale = qScaleAndZeros[..., 0]  # (G, N)
    qZeros = qScaleAndZeros[..., 1]  # (G, N)

    # Byte-pair packing: low nibble = even column (k=2j), high nibble = odd
    # column (k=2j+1); int4 values are unsigned 0..15. Dequant matches the
    # eager reference (q.float() - zero.float()) * scale.float() in float32.
    wi32 = mat2.to(torch.int32)  # (N, K2)
    lo = wi32 & 0xF  # even columns -> (N, K2)
    hi = (wi32 >> 4) & 0xF  # odd columns -> (N, K2)
    q = torch.stack([lo, hi], dim=-1).reshape(N, K)  # (N, K), interleaved on K
    q_kn = q.transpose(0, 1).to(torch.float32)  # (K, N)

    group_idx = torch.arange(K, device=A.device) // qGroupSize  # (K,)
    scale_kn = qScale[group_idx, :].to(torch.float32)  # (K, N)
    zero_kn = qZeros[group_idx, :].to(torch.float32)  # (K, N)
    w_f32 = (q_kn - zero_kn) * scale_kn  # (K, N) dequantized weight

    a_f32 = A.to(torch.float32)

    BM, BN, BK = _BM, _BN, _BK
    M_pad = triton.cdiv(M, BM) * BM
    N_pad = triton.cdiv(N, BN) * BN
    K_pad = triton.cdiv(K, BK) * BK

    # Pad all operands to launched-tile multiples so the kernel runs fully
    # unmasked / in-bounds-by-construction (masked OOB loads mis-lower on XPU3).
    ap = _pad_rows_cols(a_f32, M_pad, K_pad)
    wp = _pad_rows_cols(w_f32, K_pad, N_pad)
    out_pad = torch.empty((M_pad, N_pad), dtype=torch.float32, device=A.device)

    grid = (triton.cdiv(M_pad, BM) * triton.cdiv(N_pad, BN),)
    with torch_device_fn.device(A.device):
        _int4pack_mm_gemm_kernel[grid](
            ap,
            wp,
            out_pad,
            M_pad,
            N_pad,
            K_pad,
            ap.stride(0),
            wp.stride(0),
            out_pad.stride(0),
            BM=BM,
            BN=BN,
            BK=BK,
        )

    return out_pad[:M, :N].contiguous().to(out_dtype)
