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
import importlib
import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn

logger = logging.getLogger(__name__)

_generic = importlib.import_module("flag_gems.ops.mixed_dtypes_linear")
_ACT_MAP = _generic._ACT_MAP


@triton.jit
def _dequant_kernel(
    w_ptr,
    scale_ptr,
    out_ptr,
    K,
    N,
    ncols,
    INT4: tl.constexpr,
    EPI_SCALE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < K * N
    n = offs % N
    k = offs // N
    if INT4:
        byte_col = n // 2
        nibble = n % 2
        byte_off = k * ncols + byte_col
        wb = tl.load(w_ptr + byte_off, mask=mask, other=0).to(tl.int32)
        wi = ((wb >> (nibble * 4)) & 0xF) - 8
    else:
        byte_off = k * ncols + n
        wb = tl.load(w_ptr + byte_off, mask=mask, other=0).to(tl.int32)
        wi = wb - 128
    if EPI_SCALE:
        # fp16: cast the integer weight to fp16 then fp32 (scale is applied in
        # the epilogue). int8 (-128..127) / int4 (-8..7) are exact in fp16.
        wf = wi.to(tl.float16).to(tl.float32)
    else:
        # bf16: per-column dequant in fp32; the round-to-bf16 that defines the
        # reference boundary is done by a torch `.to()` cast on the caller side
        # (triton's bf16 rounding mode differs on XPU3).
        s = tl.load(scale_ptr + n, mask=mask, other=0.0).to(tl.float32)
        wf = wi.to(tl.float32) * s
    tl.store(out_ptr + offs, wf, mask=mask)


@triton.jit
def _epilogue_kernel(
    acc_ptr,
    scale_ptr,
    bias_ptr,
    out_ptr,
    M,
    N,
    EPI_SCALE: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    ACT: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < M * N
    n = offs % N
    acc = tl.load(acc_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    if EPI_SCALE:
        s = tl.load(scale_ptr + n, mask=mask, other=0.0).to(tl.float32)
        acc = acc * s
    if HAS_BIAS:
        b = tl.load(bias_ptr + n, mask=mask, other=0.0).to(tl.float32)
        acc = acc + b
    if ACT == 1:
        acc = tl.maximum(acc, 0.0)
    elif ACT == 2:
        acc = acc * tl.sigmoid(acc)
    tl.store(out_ptr + offs, acc.to(out_ptr.dtype.element_ty), mask=mask)


def mixed_dtypes_linear(input, weight, scale, bias=None, activation=None):
    logger.debug("GEMS_KUNLUNXIN MIXED_DTYPES_LINEAR")

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
    epi_scale = input.dtype == torch.float16

    # Route the GEMM through the backend fp32 mm (exact vs torch fp32 matmul on
    # XPU3). The native fused kernel miscompiles the masked 2D load into tl.dot
    # here and returns all-NaN; dequant + fp32 mm is in-bounds by construction.
    import flag_gems

    wdq = torch.empty((K, N), dtype=torch.float32, device=input.device)
    BLOCK = 1024
    grid_w = (triton.cdiv(K * N, BLOCK),)
    with torch_device_fn.device(input.device):
        _dequant_kernel[grid_w](
            weight,
            scale,
            wdq,
            K,
            N,
            ncols,
            INT4=int4,
            EPI_SCALE=epi_scale,
            BLOCK=BLOCK,
        )

    x2d = input.reshape(M, K).to(torch.float32)
    if not epi_scale:
        # Match the reference bf16 rounding boundary: round the dequantized
        # weight to bf16 (torch round-to-nearest-even) then back to fp32.
        wdq = wdq.to(torch.bfloat16).to(torch.float32)
    acc = flag_gems.mm(x2d, wdq)  # (M, N) fp32

    out = torch.empty((*orig[:-1], N), dtype=input.dtype, device=input.device)
    grid_e = (triton.cdiv(M * N, BLOCK),)
    with torch_device_fn.device(input.device):
        _epilogue_kernel[grid_e](
            acc,
            scale,
            bias if bias is not None else input,
            out,
            M,
            N,
            EPI_SCALE=epi_scale,
            HAS_BIAS=(bias is not None),
            ACT=_ACT_MAP[act],
            BLOCK=BLOCK,
        )
    return out
