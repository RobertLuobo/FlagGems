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

from flag_gems.runtime import torch_device_fn

logger = logging.getLogger(__name__)

_BM = 64
_BN = 64
_BK = 64


@triton.jit
def _int4pack_mm_gemm_kernel(
    a_ptr,
    w_ptr,
    c_ptr,
    M,
    N,
    K,
    stride_am,
    stride_wk,
    stride_cm,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
):
    pid = tl.program_id(0)
    grid_n = tl.cdiv(N, BN)
    pid_m = pid // grid_n
    pid_n = pid % grid_n

    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_n = pid_n * BN + tl.arange(0, BN)
    offs_k = tl.arange(0, BK)

    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :]
    w_ptrs = w_ptr + offs_k[:, None] * stride_wk + offs_n[None, :]

    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for _ in range(0, tl.cdiv(K, BK)):
        a = tl.load(a_ptrs)
        w = tl.load(w_ptrs)
        acc += tl.dot(a, w, out_dtype=tl.float32, allow_tf32=False)
        a_ptrs += BK
        w_ptrs += BK * stride_wk

    out_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :]
    tl.store(out_ptrs, acc.to(c_ptr.dtype.element_ty))


def _pad_rows_cols(src, rows, cols, pad_value=0.0):
    r, c = src.shape
    if r == rows and c == cols:
        return src
    dst = torch.full((rows, cols), pad_value, device=src.device, dtype=src.dtype)
    torch.ops.aten._copy_from(src, dst[:r, :c], False)
    return dst


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
    assert (
        K % qGroupSize == 0
    ), f"K ({K}) must be divisible by qGroupSize ({qGroupSize})"
    G = K // qGroupSize
    assert qScale.shape == (G, N), f"qScale shape {qScale.shape} != expected ({G}, {N})"
    assert qZeros.shape == (G, N), f"qZeros shape {qZeros.shape} != expected ({G}, {N})"
    assert mat2.dtype == torch.uint8, f"mat2 must be uint8, got {mat2.dtype}"
    assert A.is_floating_point(), "A must be floating point"

    out_dtype = A.dtype
    A = A.contiguous()
    mat2 = mat2.contiguous()
    qScale = qScale.contiguous()
    qZeros = qZeros.contiguous()

    # Host-side int4 unpack + dequant. On TritonXPU the generic kernel's
    # int-unpack fed into tl.dot mis-lowers (the pipeline fails outright in
    # TritonToLinalgExperimental) and the tl.join/permute/reshape interleave
    # overflows uni_sram. We therefore materialize the dequantized (K, N) float
    # weight on the host and keep only the fp32 GEMM contraction in the kernel.
    #
    # Byte-pair packing: low nibble = even column (k=2j), high nibble = odd
    # column (k=2j+1); int4 values are unsigned 0..15. The arithmetic exactly
    # mirrors the eager reference (int.float() - zero.float()) * scale.float(),
    # computed in float32.
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
