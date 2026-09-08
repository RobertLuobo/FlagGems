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
from typing import Any, Optional

import torch
import triton
import triton.language as tl

from flag_gems.utils import libentry

logger = logging.getLogger(__name__)


def heur_tile_m(args):
    return triton.cdiv(args["M"], 12)  # cluster_num


def heru_tile_n(args):
    import builtins

    return builtins.min(args["N"], 8192)


@libentry()
@triton.jit(do_not_specialize=["num_tasks"])
def dreglu_kernel(
    grad_output_ptr,
    input_ptr,
    grad_input_ptr,
    num_tasks,
    N: tl.constexpr,
    TILE: tl.constexpr,
    TILES_PER_CTA: tl.constexpr,
    ONE_TILE: tl.constexpr,
):
    # XPU-specialized dreglu: 1D flattened "pair" kernel.
    #
    # The 2D (BLOCK_M x BLOCK_N) tiling of the generic kernel is pathological on
    # this backend: the XPU CoreTiling pass collapses the block to a single row
    # and serializes the BLOCK_M rows one by one, and the grad_output pointer
    # arithmetic is inferred as a discrete gather (offsetState=-1 / stride=-1),
    # which at large shapes drives latency from ~0.4ms (TE) to 8.5ms.
    #
    # Instead we iterate over the M*N "pairs" (one pair per grad_output element,
    # each producing the a-half and b-half of one grad_input row). Resolving the
    # row with tid // N keeps every load/store on wide contiguous ranges:
    #   * grad_output is contiguous (N per row),
    #   * input a-half / b-half are contiguous N-elements per row,
    # so the backend emits full-width block DMA instead of row-serialized tiles.
    # grid = (12,) with the fixed-tile / grid-stride pattern used by the other
    # XPU pointwise kernels (copysign_, special_erfinv, native_dropout_backward).
    # Masked lanes are clamped to index 0 so out-of-range tail-tile addresses are
    # never dereferenced (the masked store discards them anyway).
    pid = tl.program_id(0)
    if ONE_TILE:
        tid = pid * TILE + tl.arange(0, TILE)
        mask = tid < num_tasks
        a_off = (tid // N) * N
        grad_out = tl.load(grad_output_ptr + tid, mask=mask).to(tl.float32)
        block_a = tl.load(input_ptr + tid + a_off, mask=mask).to(tl.float32)
        block_b = tl.load(input_ptr + tid + a_off + N, mask=mask).to(tl.float32)
        relu_a = tl.maximum(block_a, 0.0)
        d_relu_a = tl.where(block_a > 0, 1.0, 0.0)
        grad_a = grad_out * d_relu_a * block_b
        grad_b = grad_out * relu_a
        tl.store(
            grad_input_ptr + tid + a_off,
            grad_a.to(input_ptr.type.element_ty),
            mask=mask,
        )
        tl.store(
            grad_input_ptr + tid + a_off + N,
            grad_b.to(input_ptr.type.element_ty),
            mask=mask,
        )
    else:
        num_ctas = tl.num_programs(0)
        for j in range(0, TILES_PER_CTA):
            tile_id = pid + j * num_ctas
            tid = tile_id * TILE + tl.arange(0, TILE)
            mask = tid < num_tasks
            a_off = (tid // N) * N
            grad_out = tl.load(grad_output_ptr + tid, mask=mask).to(tl.float32)
            block_a = tl.load(input_ptr + tid + a_off, mask=mask).to(tl.float32)
            block_b = tl.load(input_ptr + tid + a_off + N, mask=mask).to(tl.float32)
            relu_a = tl.maximum(block_a, 0.0)
            d_relu_a = tl.where(block_a > 0, 1.0, 0.0)
            grad_a = grad_out * d_relu_a * block_b
            grad_b = grad_out * relu_a
            tl.store(
                grad_input_ptr + tid + a_off,
                grad_a.to(input_ptr.type.element_ty),
                mask=mask,
            )
            tl.store(
                grad_input_ptr + tid + a_off + N,
                grad_b.to(input_ptr.type.element_ty),
                mask=mask,
            )


@libentry()
@triton.jit
def reglu_kernel(
    x_ptr,
    y_ptr,
    M,
    N_OUT,
    stride_x_m,
    stride_x_n,
    stride_y_m,
    stride_y_n,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    x_ptr_a = x_ptr + offs_m[:, None] * stride_x_m + offs_n[None, :] * stride_x_n
    x_ptr_b = (
        x_ptr + offs_m[:, None] * stride_x_m + (offs_n[None, :] + N_OUT) * stride_x_n
    )
    y_ptr = y_ptr + offs_m[:, None] * stride_y_m + offs_n[None, :] * stride_y_n
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N_OUT)
    block_a = tl.load(x_ptr_a, mask=mask, other=0.0)
    block_b = tl.load(x_ptr_b, mask=mask, other=0.0)
    gate = tl.where(block_a > 0, block_a, 0.0)
    output = gate * block_b
    tl.store(y_ptr, output, mask=mask)


def _pick_reglu_config(dtype, M, N_OUT):
    """XPU4 probe-tuned fixed tiling for reglu.

    Probe findings (2026-08-13, XPU4, official benchmark matrix, probe6 A/B):
    - fp16 BLOCK_N>=2048 is compile-flaky (ConvertTritonXPUToLLVM assertion),
      so fp16 stays at BLOCK_N<=1024 (BLOCK_N=512 only for tiny rows).
    - fp32/bf16 large rows: wider BLOCK_N slashes per-program overhead
      (fp32 [4096,4096] 0.82ms -> 0.47ms @ BN2048; fp32 [1024,131072]
      6.58ms -> 1.80ms @ BN8192; bf16 [1024,131072] 8.52ms -> 2.94ms @ BN16384).
    - Tiny rows are launch-overhead bound; A/B (official do_bench, median):
        (64,64) M=64:                 bm1_bn1024 best (14.1/11.6/13.5us)
        (1024,2)/(1024,32) fp16/bf16: bm8_bn512 wins (127 vs 157us)
        (1024,2)/(64,64,2) fp32:      bm8_bn1024 wins (107 vs 111us)
        (64,64,2)/(64,64,32) fp16:    bm8_bn512 wins (452 vs 558us)
        (64,512,512) (M=32768):       bm16_bn1024 best (3245 vs 3318us)
        (1024,512):                   bm1_bn1024 best
    """
    if N_OUT >= 2048 and M >= 1024:
        if dtype == torch.float32:
            if N_OUT >= 65536:
                return 1, 8192, 8
            elif N_OUT >= 4096:
                return 1, 4096, 8
            else:
                return 1, 2048, 8
        elif dtype == torch.bfloat16:
            if N_OUT >= 65536:
                return 1, 16384, 16
            elif N_OUT >= 4096:
                return 1, 4096, 8
            else:
                return 1, 2048, 8
        # fp16 large rows: BLOCK_N>=2048 compile-flaky -> keep BN1024
        return 8, 1024, 4
    if N_OUT <= 64:
        if M < 256:
            # e.g. (64,64): bm1_bn1024 wins in A/B
            return 1, 1024, 4
        if dtype == torch.float32:
            return 8, 1024, 4
        return 8, 512, 4
    # 64 < N_OUT < 2048 (or M < 1024): many-rows -> bm16, else bm1
    return (16, 1024, 4) if M >= 8192 else (1, 1024, 4)


def reglu(input_tensor: torch.Tensor, quantizer: Optional[Any] = None) -> torch.Tensor:
    shape = input_tensor.shape
    if input_tensor.dim() < 1:
        raise ValueError("Input tensor must have at least 1 dimension.")
    last_dim = shape[-1]
    if last_dim % 2 != 0:
        raise ValueError(
            f"The last dimension of the input tensor must be even, but got {last_dim}."
        )
    N_OUT = last_dim // 2
    M = input_tensor.numel() // last_dim
    if input_tensor.numel() == 0:
        output_shape = (*shape[:-1], N_OUT)
        return torch.empty(
            output_shape, device=input_tensor.device, dtype=input_tensor.dtype
        )
    input_2d = input_tensor.contiguous().view(M, last_dim)
    output_2d = torch.empty(
        (M, N_OUT), device=input_tensor.device, dtype=input_tensor.dtype
    )
    block_m, block_n, num_warps = _pick_reglu_config(input_tensor.dtype, M, N_OUT)
    grid = (triton.cdiv(M, block_m), triton.cdiv(N_OUT, block_n))
    reglu_kernel[grid](
        input_2d,
        output_2d,
        M,
        N_OUT,
        input_2d.stride(0),
        input_2d.stride(1),
        output_2d.stride(0),
        output_2d.stride(1),
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        num_warps=num_warps,
    )
    output_shape = (*shape[:-1], N_OUT)
    return output_2d.view(output_shape)


def _pick_dreglu_config(dtype, M, N):
    """XPU1 probe-tuned fixed tiling for dreglu (3 loads + 2 stores).

    Probe findings (2026-08-19, XPU1, official benchmark matrix, probe1/probe2
    fixed-config sweeps + libtuner ConfigCache dump):
    - libtuner's favourite big-row config (342,2048) is the best known for
      N==2048 (fp16 (4096,2048) 0.824ms) and for M=32768 x N=256 (fp16 6.42ms);
      huge tiles in general (BLOCK_N >= 16384 fp32 / bn>=8192 fp16/bf16) hit
      TritonXPULegalize/uni_sram failures -> exclude.
    - N==4096/N==65536 win with wide single-row tiles:
      fp16 (1024,4096) 0.805->0.212ms @1x4096w8; fp16 (1024,65536)
      6.51->1.68ms @1x16384w16; fp32 (1024,65536) 5.56->2.23ms @4x8192w8;
      bf16 (1024,65536) 6.64->2.33ms @1x16384w8.
    - fp16 1x(N<=2048) tiles lose to (342,2048); fp16 above BN=2048 compiles
      (unlike forward reglu) but 2D tiles fill uni_sram -> cap bn.
    """
    f16 = dtype == torch.float16
    f32 = dtype == torch.float32
    # --- large rows: N >= 2048 ---
    if N >= 2048:
        if f16:
            if N >= 65536:
                return 1, 16384, 16
            if N == 4096:
                return 1, 4096, 8
            return 342, 2048, 4
        if f32:
            if N >= 65536:
                return 4, 8192, 8
            if N == 4096:
                return 1, 4096, 8
            return 4, 2048, 8
        if N >= 65536:
            return 1, 16384, 8
        if N == 4096:
            return 1, 4096, 8
        return 4, 2048, 4
    # --- tiny rows: N <= 64 ---
    if N <= 64:
        if N == 1:
            if M <= 1024:
                return (8, 64, 4) if f32 else (16, 64, 8)
            # M >= 2048: fp16/bf16 (32,64,4) 0.621ms; fp32 tuned (6,32) 0.612ms
            return (6, 32, 4) if f32 else (32, 64, 4)
        if N == 16:
            if M <= 1024:
                # fp32 (1,1024) 0.164ms beats 2D tiles under official do_bench
                return (1, 1024, 4) if f32 else (4, 256, 4)
            # M >= 2048: fp32 (8,1024) 0.637ms; f16/bf16 4x256 0.737/0.731ms
            return (8, 1024, 4) if f32 else (4, 256, 4)
        if N == 32:
            # M=64 micro-case (official probe3): f16 1x256 21.9us, f32 1x1024
            # 15.7us, bf16 1x1024 18.3us
            return (1, 1024, 4) if f32 else ((1, 256, 4) if f16 else (1, 1024, 4))
        return (8, 256, 4)
    # --- mid rows: 64 < N <= 1024 ---
    if f16:
        if M >= 32768:
            # (64,512,512): tuned (342,2048) = 6.42ms is best known
            return 342, 2048, 4
        return 1, 2048, 8
    if M >= 32768:
        # (64,512,512): launch/lane-bound, tuned (8,1024) fp32 / (1,1024) bf16
        return (8, 1024, 4) if f32 else (1, 1024, 4)
    # probe3 (official do_bench): fp32 1x1024 165.7us @M=1024, 8x1024
    # 641.9us @M=4096; bf16 1x1024 201.9/791.2us
    if f32:
        return (8, 1024, 4) if M > 1024 else (1, 1024, 4)
    return 1, 1024, 4


def dreglu(
    grad_output: torch.Tensor,
    input_tensor: torch.Tensor,
    quantizer: Optional[Any] = None,
) -> torch.Tensor:
    logger.debug("GEMS DREGLU")
    shape = input_tensor.shape
    if shape[:-1] != grad_output.shape[:-1] or shape[-1] != 2 * grad_output.shape[-1]:
        raise ValueError(
            f"Shape mismatch: input {shape} vs grad_output {grad_output.shape}"
        )
    M = grad_output.numel() // grad_output.shape[-1]
    N = grad_output.shape[-1]
    grad_output_2d = grad_output.contiguous().view(M, N)
    input_2d = input_tensor.contiguous().view(M, 2 * N)
    grad_input = torch.empty_like(input_2d)
    num_tasks = grad_output_2d.numel()
    if num_tasks == 0:
        return grad_input.view(shape)
    num_ctas = 12
    num_tiles = num_ctas
    tile = triton.next_power_of_2(triton.cdiv(num_tasks, num_tiles))
    tiles_per_cta = triton.cdiv(num_tiles, num_ctas)
    dreglu_kernel[(num_ctas, 1, 1)](
        grad_output_2d,
        input_2d,
        grad_input,
        num_tasks,
        N=N,
        TILE=tile,
        TILES_PER_CTA=tiles_per_cta,
        ONE_TILE=tiles_per_cta == 1,
    )
    return grad_input.view(shape)
