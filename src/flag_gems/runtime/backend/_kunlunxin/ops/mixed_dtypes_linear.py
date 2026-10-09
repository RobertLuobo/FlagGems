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

_ACT_MAP = {"none": 0, None: 0, "relu": 1, "silu": 2}

_BM = 128
_BN = 128
_BK = 64
_GROUP_M = 8


@triton.jit
def _mixed_dtypes_linear_kernel(
    x_ptr,
    w_ptr,
    scale_ptr,
    bias_ptr,
    out_ptr,
    M,
    N,
    K,
    stride_xm,
    stride_wn,
    stride_om,
    HAS_BIAS: tl.constexpr,
    ACT: tl.constexpr,
    EPI_SCALE: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BM)
    num_pid_n = tl.cdiv(N, BN)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = tl.minimum(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_n = pid_n * BN + tl.arange(0, BN)
    offs_k = tl.arange(0, BK)

    # w_ptr holds a dense (K, N) weight materialized on the host (see the
    # launcher), so the kernel is a plain fp32-accumulate GEMM plus a small
    # epilogue. Two things are deliberately kept off the device:
    #   * int bit-ops / tl.interleave of the generic kernel: `(int32 - bias)`
    #     fed into tl.dot mis-lowers to all NaN on TritonXPU, and tl.interleave
    #     overflows uni_sram at these tile sizes.
    #   * the bf16 per-tile dequant multiply: triton's in-kernel `.to(bf16)`
    #     rounds differently from torch's, so an in-loop `(w*scale).to(bf16)`
    #     diverges from the reference at cancellation points (measured 5/5
    #     failures). For bf16 the scaled weight is therefore pre-rounded on the
    #     host (EPI_SCALE=False -> w already == dequantized wdq, no scale here);
    #     for fp16 the integer weight is passed and the per-column scale is
    #     folded into the fp32 epilogue, matching the reference's placement.
    x_ptrs = x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :]
    w_ptrs = w_ptr + offs_k[:, None] * stride_wn + offs_n[None, :]

    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for _ in range(0, tl.cdiv(K, BK)):
        a = tl.load(x_ptrs)
        w = tl.load(w_ptrs)
        acc += tl.dot(a, w, out_dtype=tl.float32, allow_tf32=False)
        x_ptrs += BK
        w_ptrs += BK * stride_wn

    if EPI_SCALE:
        s_epi = tl.load(scale_ptr + offs_n)
        acc = acc * s_epi.to(tl.float32)[None, :]
    if HAS_BIAS:
        b = tl.load(bias_ptr + offs_n)
        acc = acc + b.to(tl.float32)[None, :]
    if ACT == 1:
        acc = tl.maximum(acc, 0.0)
    elif ACT == 2:
        acc = acc * tl.sigmoid(acc)

    out_ptrs = out_ptr + offs_m[:, None] * stride_om + offs_n[None, :]
    tl.store(out_ptrs, acc.to(out_ptr.dtype.element_ty))


def _pad_rows_cols(src, rows, cols, pad_value=0):
    """Materialize ``src`` into a (rows, cols) contiguous buffer, zero-padded.

    TritonXPU mis-lowers masked loads/stores whose addresses leave the
    allocation (see the vendor mm_kernel comment), so instead of relying on
    in-kernel masks we pad every operand to the launched-tile multiples on the
    host and run the kernel fully unmasked / in-bounds-by-construction. The
    copy goes through the native engine (not overridden by gems); padded
    rows/cols produce values that are discarded by the host-side output slice.
    """
    r, c = src.shape
    if r == rows and c == cols:
        return src
    dst = torch.full((rows, cols), pad_value, device=src.device, dtype=src.dtype)
    torch.ops.aten._copy_from(src, dst[:r, :c], False)
    return dst


def _pad_vec(src, n, pad_value=0):
    (m,) = src.shape
    if m == n:
        return src
    dst = torch.full((n,), pad_value, device=src.device, dtype=src.dtype)
    torch.ops.aten._copy_from(src, dst[:m], False)
    return dst


def mixed_dtypes_linear(input, weight, scale, bias=None, activation=None):
    logger.debug("GEMS_KUNLUNXIN _MIXED_DTYPES_LINEAR")

    assert input.dtype in (
        torch.float16,
        torch.bfloat16,
    ), f"input must be float16 or bfloat16, got {input.dtype}"
    assert weight.dtype == torch.uint8, f"weight must be uint8, got {weight.dtype}"
    assert (
        scale.dtype == input.dtype
    ), f"scale dtype {scale.dtype} must match input dtype {input.dtype}"
    act = activation if activation is not None else "none"
    assert act in _ACT_MAP, f"unsupported activation: {activation}"

    orig = input.shape
    K = orig[-1]
    M = input.numel() // K
    N = scale.shape[0]
    ncols = weight.shape[1]
    assert ncols == N or 2 * ncols == N, (
        f"weight columns {ncols} must equal scale size {N} (int8) "
        f"or half of it (packed int4)"
    )
    int4 = ncols != N

    x2d = input.reshape(M, K)

    # Host-side dequant. int8: FasterTransformer biased converter (byte - 128);
    # int4: unpack low/high nibbles (nibble - 8) interleaved to the output
    # column order. This is pure weight prep (the GEMM stays in Triton) and
    # keeps the int bit-ops / interleave off the device, where TritonXPU either
    # NaNs ((int32 - bias) -> tl.dot) or overflows uni_sram (tl.interleave).
    if int4:
        wi32 = weight.to(torch.int32)
        lo = (wi32 & 0xF) - 8
        hi = ((wi32 >> 4) & 0xF) - 8
        w_int = torch.stack([lo, hi], dim=-1).reshape(K, N)
    else:
        w_int = weight.to(torch.int32) - 128

    epi_scale = input.dtype == torch.float16
    if epi_scale:
        # fp16: pass the (exact, integer-valued) weight; the per-column scale is
        # folded into the fp32 epilogue in-kernel, matching the reference.
        wmat = w_int.to(input.dtype)
    else:
        # bf16: pre-round the dequantized weight on the host so its bf16
        # rounding matches the reference exactly (triton's in-kernel .to(bf16)
        # rounds differently and diverges at cancellation points). Scale is
        # fully baked in here, so the kernel does no scaling for bf16.
        wmat = (w_int.to(torch.float32) * scale.to(torch.float32)[None, :]).to(
            input.dtype
        )

    BM, BN, BK, GROUP_M = _BM, _BN, _BK, _GROUP_M
    M_pad = triton.cdiv(M, BM) * BM
    N_pad = triton.cdiv(N, BN) * BN
    K_pad = triton.cdiv(K, BK) * BK

    # Pad every operand to the launched-tile multiples so the kernel runs
    # fully unmasked and in-bounds-by-construction on TritonXPU.
    xp = _pad_rows_cols(x2d, M_pad, K_pad)
    wp = _pad_rows_cols(wmat, K_pad, N_pad)
    scale_p = _pad_vec(scale, N_pad)
    if bias is not None:
        bias_p = _pad_vec(bias, N_pad)
    else:
        bias_p = scale_p  # unused by the kernel (HAS_BIAS=False)

    out_pad = torch.empty((M_pad, N_pad), dtype=input.dtype, device=input.device)

    grid = (triton.cdiv(M_pad, BM) * triton.cdiv(N_pad, BN),)
    with torch_device_fn.device(input.device):
        _mixed_dtypes_linear_kernel[grid](
            xp,
            wp,
            scale_p,
            bias_p,
            out_pad,
            M_pad,
            N_pad,
            K_pad,
            xp.stride(0),
            wp.stride(0),
            out_pad.stride(0),
            HAS_BIAS=(bias is not None),
            ACT=_ACT_MAP[act],
            EPI_SCALE=epi_scale,
            BM=BM,
            BN=BN,
            BK=BK,
            GROUP_M=GROUP_M,
            num_warps=8,
            num_stages=4,
        )

    out = out_pad[:M, :N].contiguous()
    return out.reshape(*orig[:-1], N)
