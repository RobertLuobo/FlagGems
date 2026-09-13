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

from flag_gems.utils import libentry, tl_extra_shim

erf = tl_extra_shim.erf
exp = tl_extra_shim.exp
tanh = tl_extra_shim.tanh
# NOTE: `tl_extra_shim.pow` must not be called with an *integer* exponent on
# this backend. `pow(x, 2)` lowers to an `Unsupported` external symbol and the
# XPU3 ELF converter fails at link time with
# `ld.lld: error: undefined symbol: Unsupported`, so every geglu/dgeglu launch
# aborted before this fix. A *float* exponent links fine -- verified in
# isolation on XPU2: `pow(x, 2.0)` / `pow(x, 3.0)` / `pow(x, 0.5)` all compile
# and match `x * x` bit-for-bit, and vendor `fused/gelu_and_mul.py` relies on
# `pow(x_fp32, 2.0)`. The restriction is on the exponent's type, not on `pow`
# itself. The x**2 terms here are written as `x * x`.

logger = logging.getLogger(__name__)


@libentry()
@triton.jit
def geglu_kernel(
    input_ptr,
    output_ptr,
    M,
    H,
    stride_in_m,
    stride_in_h,
    stride_out_m,
    stride_out_h,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_H: tl.constexpr,
    NEED_MASK: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_h = tl.program_id(1)

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_h = pid_h * BLOCK_SIZE_H + tl.arange(0, BLOCK_SIZE_H)

    input_a_ptr = (
        input_ptr + offs_m[:, None] * stride_in_m + offs_h[None, :] * stride_in_h
    )
    input_b_ptr = (
        input_ptr + offs_m[:, None] * stride_in_m + (offs_h[None, :] + H) * stride_in_h
    )
    output_ptr = (
        output_ptr + offs_m[:, None] * stride_out_m + offs_h[None, :] * stride_out_h
    )

    if NEED_MASK:
        mask = (offs_m[:, None] < M) & (offs_h[None, :] < H)
        x_a = tl.load(input_a_ptr, mask=mask, other=0.0).to(tl.float32)
        x_b = tl.load(input_b_ptr, mask=mask, other=0.0).to(tl.float32)
        gelu_out = 0.5 * x_a * (1 + tanh(0.79788456 * x_a * (1 + 0.044715 * x_a * x_a)))
        tl.store(output_ptr, gelu_out * x_b, mask=mask)
    else:
        x_a = tl.load(input_a_ptr).to(tl.float32)
        x_b = tl.load(input_b_ptr).to(tl.float32)
        gelu_out = 0.5 * x_a * (1 + tanh(0.79788456 * x_a * (1 + 0.044715 * x_a * x_a)))
        tl.store(output_ptr, gelu_out * x_b)


@triton.jit
def dgeglu_kernel(
    grad_out_ptr,
    input_ptr,
    grad_in_ptr,
    M,
    H,
    stride_grad_out_m,
    stride_grad_out_h,
    stride_in_m,
    stride_in_h,
    stride_grad_in_m,
    stride_grad_in_h,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_H: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_h = tl.program_id(1)

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_h = pid_h * BLOCK_SIZE_H + tl.arange(0, BLOCK_SIZE_H)

    mask = (offs_m[:, None] < M) & (offs_h[None, :] < H)

    grad_out_ptr = (
        grad_out_ptr
        + offs_m[:, None] * stride_grad_out_m
        + offs_h[None, :] * stride_grad_out_h
    )
    input_a_ptr = (
        input_ptr + offs_m[:, None] * stride_in_m + offs_h[None, :] * stride_in_h
    )
    input_b_ptr = (
        input_ptr + offs_m[:, None] * stride_in_m + (offs_h[None, :] + H) * stride_in_h
    )
    grad_a_ptr = (
        grad_in_ptr
        + offs_m[:, None] * stride_grad_in_m
        + offs_h[None, :] * stride_grad_in_h
    )
    grad_b_ptr = (
        grad_in_ptr
        + offs_m[:, None] * stride_grad_in_m
        + (offs_h[None, :] + H) * stride_grad_in_h
    )

    grad_out = tl.load(grad_out_ptr, mask=mask, other=0.0).to(tl.float32)
    x_a = tl.load(input_a_ptr, mask=mask, other=0.0).to(tl.float32)
    x_b = tl.load(input_b_ptr, mask=mask, other=0.0).to(tl.float32)

    tanh_out = tanh(0.79788456 * x_a * (1 + 0.044715 * x_a * x_a))
    gelu_out = 0.5 * x_a * (1 + tanh_out)

    # dgelu/dx
    sech2 = 1 - tanh_out * tanh_out
    dgelu = 0.5 * (1 + tanh_out) + 0.5 * x_a * sech2 * 0.79788456 * (
        1 + 3 * 0.044715 * x_a * x_a
    )

    grad_a = grad_out * x_b * dgelu
    grad_b = grad_out * gelu_out

    tl.store(grad_a_ptr, grad_a.to(x_a.dtype), mask=mask)
    tl.store(grad_b_ptr, grad_b.to(x_a.dtype), mask=mask)


@libentry()
@triton.jit
def geglu_pair_kernel(
    input_ptr,
    output_ptr,
    num_tasks,
    H: tl.constexpr,
    TILE: tl.constexpr,
):
    """1D flattened "pair" kernel for geglu (2 loads + 1 store).

    XPU-specialized variant used for H <= 1024 (see `_pick_geglu_pair_tile`).
    The 2D (BLOCK_SIZE_M x BLOCK_SIZE_H) tiling above is pathological on this
    backend whenever the row count M dwarfs the half-width H: with only a few
    useful lanes per row the CoreTiling pass serializes BLOCK_SIZE_M rows one
    by one and the many small programs stay launch/over-read bound, so e.g.
    (16,7,57,32,30) fp32 runs ~19.7ms and (64,64,2) fp16 ~0.38ms.

    Instead (same idea as the dreglu pair kernel) we iterate over the M*H
    output elements, one "pair" per element: element t of row t//H reads
    input[2H*(t//H) + t%H] (the a-half) and input[2H*(t//H) + t%H + H] (the
    b-half) and writes output[t]. Every load/store then stays on wide
    contiguous ranges (N elements per row), so the backend emits full-width
    block DMA instead of row-serialized tiles; a handful of programs covers
    the whole tensor.

    H is a tl.constexpr (not a runtime arg) on purpose: with a runtime H the
    (tid // H) * H division lowers to a hardware divide and, more importantly,
    a subset of (dtype, TILE) combinations mis-lower and return wrong values
    (probe 2026-09-10, see `_pick_geglu_pair_tile`). With a compile-time H the
    division folds to a shift and every (H, TILE) cell in the probe grid is
    numerically bit-identical to the 2D kernel.
    """
    pid = tl.program_id(0)
    tid = pid * TILE + tl.arange(0, TILE)
    mask = tid < num_tasks
    a_off = (tid // H) * H
    x_a = tl.load(input_ptr + tid + a_off, mask=mask).to(tl.float32)
    x_b = tl.load(input_ptr + tid + a_off + H, mask=mask).to(tl.float32)
    gelu_out = 0.5 * x_a * (1 + tanh(0.79788456 * x_a * (1 + 0.044715 * x_a * x_a)))
    tl.store(output_ptr + tid, gelu_out * x_b, mask=mask)


def _pick_geglu_config(dtype, M, H):
    """XPU2 probe-tuned fixed tiling for the geglu forward (2 loads + 1 store).

    Probe: 2026-08-29, XPU 2, official benchmark matrix (12 (M, H) cells x
    fp16/fp32/bf16), `triton.testing.do_bench(return_mode="median")` with the
    same warmup/rep as `benchmark/base.py`, every timing gated by a CPU-fp64
    tanh-approx oracle at the accuracy-test tolerances
    (`/tmp/geglu_xpu2_probe/tile_sweep{2,3,4,5}.log`).

    Findings that drive the bands below:
    - Wide half-rows want one row per program and the widest legal tile:
      (1024, 131072) fp16 9.05ms -> 1.32ms @ (1, 16384, w8),
      (1024, 8192) fp16 0.57 -> 0.148ms @ (1, 4096, w8),
      (4096, 4096) fp16 1.13 -> 0.462ms @ (4, 2048, w8).
      fp16 BLOCK_H >= 2048 compiles fine here (unlike the forward `reglu`
      note), so fp16 is *not* capped at 1024.
    - **fp32 + BLOCK_SIZE_H == 1024 + masked tile mis-lowers on this backend**:
      it silently returns wrong values for lanes inside the mask
      ((1024,512) 16223, (4096,512) 65044, (64,512,512) 5.2e5,
      (1024,32) 1020, (64,64,2) 4079 wrong elements). fp16/bf16 at BLOCK_H=1024
      and fp32 at BLOCK_H 512/2048/4096/8192/16384 are all clean, so fp32 mid
      rows are pinned to BLOCK_H = 512.
    - Mask-free narrow tiles (BLOCK_H = H when H <= 16) are both wrong
      (BLOCK_H 8/16 mis-lower) and 3-10x slower, and shrinking BLOCK_H to
      32/64/128 for tiny H is uniformly slower than a 512-wide masked read
      (contiguous DMA dominates the wasted lanes). Tiny H therefore keeps a
      512-wide tile.
    - `H <= 64` cells stay launch/over-read bound at ~0.9us per program
      (0.12ms @ M=1024, 0.47ms @ M=4096) and `M=32768, H=256` stays at
      ~4ms; both are the documented XPU floors for many-rows/short-columns
      2D tiles and no tile in the sweep beats them.
    """
    f32 = dtype == torch.float32
    bf16 = dtype == torch.bfloat16
    # --- wide half-rows (H >= 2048): widest legal tile, ~1 row per program ---
    if H >= 2048:
        if H >= 8192:
            if f32:
                return 1, 8192, 8
            return (1, 16384, 8) if H % 16384 == 0 else (1, 8192, 8)
        if H >= 4096:
            return 1, 4096, 8
        return (1, 2048, 8) if bf16 else (4, 2048, 8)
    # --- mid half-rows (64 < H < 2048) ---
    if H > 64:
        if f32:
            # BLOCK_H = 1024 is numerically unsafe for fp32 masked tiles
            if M >= 32768:
                return 64, 512, 4
            return (8, 512, 4) if M >= 4096 else (2, 512, 4)
        if M >= 32768:
            return 16, 1024, 4
        return (8, 1024, 4) if M >= 4096 else (1, 1024, 4)
    # --- tiny half-rows (H <= 64) ---
    if H == 1:
        # H == 1 keeps a single useful lane per row, so a narrower 128-wide
        # contiguous read wins: (1024,2) fp32 0.124 -> 0.098ms,
        # (64,64,2) fp16 0.490 -> 0.381ms versus a 512-wide tile.
        if f32:
            return 8, 128, 4
        return (32, 128, 4) if M <= 1024 else (16, 64, 4)
    if M < 256:
        return (1, 256, 4) if f32 else (1, 512, 4)
    if M > 1024:
        # (64,64,32) 0.481 -> 0.448ms; also covers the (M=131072, H=30) and
        # (M=204288, H=15) accuracy-test shapes.
        return 64, 256, 4
    return 8, 512, 4


def _pick_geglu_pair_tile(dtype, M, H):
    """XPU2 probe-tuned fixed tile for the 1D geglu pair kernel (H <= 1024).

    Probe: 2026-09-10, XPU2 (card 3), the official benchmark + accuracy-test
    matrix, constexpr-H pair kernel (`/tmp/ab_geglu4.py`,
    `/tmp/warp_probe.py` -- timing, `triton.testing.do_bench(median)` with the
    host-sync removed so small-kernel figures are not inflated) and a dense
    (M x H) correctness grid (`/tmp/sweep_clean_{fp16,fp32,bf16}.py`, 168 cells
    x 4 tiles, `d < 1e-4` vs the 2D kernel output -- with identical FP
    expressions the pair kernel is bit-identical to the 2D kernel whenever it
    does not mis-lower, so this is an exact codegen check).

    Why the pair kernel wins for small H: on every H <= 1024 cell probed it
    beats the 2D tiling, from 1.5x-2x at H ~ 256-1024
    ((64,64,512) fp16 0.59ms -> 0.12ms; (4096,1024) fp32 0.65ms -> 0.18ms)
    up to 15x-130x on the wide/short cells
    ((16,7,57,32,30) fp32 19.7ms -> 0.15ms @ T=2048; (64,64,2) fp16 0.38ms
    -> 0.006ms; (1024,2) fp16 0.11ms -> 0.006ms).

    Numerics / the TILE minefield (constexpr-H, w4; the same pattern with a
    runtime H keeps a subset of these mis-lowering cells):
    - fp32: every (M, H, TILE) in the grid is clean -> free to pick the fastest.
    - fp16: TILE 256/512/1024 clean everywhere; TILE 2048 mis-lowers on most
      cells -> capped at 1024.
    - bf16: clean-TILE depends on H: H <= 64 -> 512; 64 < H <= 1024 -> 1024.
      (T=1024 is wrong at H <= 64 for large M, T=512 / T=2048 are wrong at
      H in (64, 1024] for M >= ~1024; T=1024 is the only clean size in that
      band.)
    - num_warps is 4 (dreglu-consistent): A/B showed w4 == w8 within noise.
    - H > 1024 keeps the 2D kernel below: the 2D wide-row tiles are as fast as
      (H=1024/2048) or faster (H >= 4096) than the pair kernel there, e.g.
      (1024,8192) 0.14ms 2D vs 0.9ms+ pair.
    """
    if dtype == torch.bfloat16:
        # bf16: T=512 is clean for H <= 64 at every M in the grid; T=1024
        # clean for 64 < H <= 1024 at every M in the grid + ext probe.
        return 512 if H <= 64 else 1024
    if dtype == torch.float32 and M >= 65536:
        # fp32 large-M short-H cells (e.g. (16,7,57,32,30) H=15, M=204288)
        # want the widest tile: 0.235ms @ T=1024 vs 0.150ms @ T=2048.
        return 2048
    return 1024


def geglu(input_tensor: torch.Tensor, quantizer: Optional[Any] = None) -> torch.Tensor:
    shape = input_tensor.shape
    if input_tensor.dim() < 1:
        raise ValueError("Input tensor must have at least 1 dimension.")
    last_dim = shape[-1]
    if last_dim % 2 != 0:
        raise ValueError(
            f"The last dimension of the input tensor must be even, but got {last_dim}."
        )
    H = last_dim // 2
    output_shape = (*shape[:-1], H)
    if input_tensor.numel() == 0:
        return torch.empty(
            output_shape, device=input_tensor.device, dtype=input_tensor.dtype
        )
    M = input_tensor.numel() // last_dim

    input_2d = input_tensor.contiguous().view(M, last_dim)
    output_2d = torch.empty(M, H, device=input_tensor.device, dtype=input_tensor.dtype)

    if H <= 1024:
        tile = _pick_geglu_pair_tile(input_tensor.dtype, M, H)
        num_tasks = M * H
        geglu_pair_kernel[(triton.cdiv(num_tasks, tile),)](
            input_2d,
            output_2d,
            num_tasks,
            H=H,
            TILE=tile,
            num_warps=4,
        )
        return output_2d.view(output_shape)

    block_m, block_h, num_warps = _pick_geglu_config(input_tensor.dtype, M, H)
    need_mask = (M % block_m != 0) or (H % block_h != 0)
    grid = (triton.cdiv(M, block_m), triton.cdiv(H, block_h))

    geglu_kernel[grid](
        input_2d,
        output_2d,
        M,
        H,
        input_2d.stride(0),
        input_2d.stride(1),
        output_2d.stride(0),
        output_2d.stride(1),
        BLOCK_SIZE_M=block_m,
        BLOCK_SIZE_H=block_h,
        NEED_MASK=need_mask,
        num_warps=num_warps,
    )
    return output_2d.view(output_shape)


def _pick_dgeglu_config(dtype, M, H):
    """XPU2 probe-tuned fixed tiling for the dgeglu backward (2 loads + 2 stores).

    Probe: 2026-09-10, XPU2 (card 5), the official benchmark matrix
    (12 (M, H) cells x fp16/fp32/bf16), steady-state host timing
    (`/tmp/probe_dgeglu_tiles.py`, `/tmp/probe_dgeglu_mid{,2}.py`), every
    candidate validated against the previous 64x64 tiling and an fp64 oracle
    (`/tmp/probe_dgeglu_correct.py`).

    The old fixed 64x64 tile is catastrophically slow on wide rows:
      (16384, 4096) fp16 258ms -> 4.42ms @ (1, 4096, w8),   ~58x
      (1024, 65536) fp32 257ms -> 3.25ms @ (1, 8192, w8),  ~79x
      (4096, 2048)  fp16 41.7ms -> 0.81ms @ (4, 2048, w8), ~52x
      (64, 32)      fp16  0.15ms -> 0.033ms @ (8, 512, w4) ~5x
    As with the forward `_pick_geglu_config`, wide half-rows want one row per
    program and the widest legal tile; 2D 8x256/64x64 tiles cost 3-6x more
    for the same element count.

    Numerics (all within the accuracy-test windows; the accuracy-test
    shapes, H <= 128, all land on the bitwise-identical 8x512/8x128 paths):
    - fp32 at BLOCK_H >= 1024 differs from 64x64 by <= 7.1e-6 (FMA
      reassociation; 0 elements beyond atol=1e-4/rtol=1.3e-6), unlike the
      *forward* geglu where fp32 + BLOCK_H == 1024 masked tiles mis-lower.
    - fp16/bf16 at BLOCK_H >= 1024 differ by <= 1 ULP on <= 0.06% of
      elements (within the 0.016/1e-3 rtol windows).
    - BLOCK_H <= 512 paths are bitwise-identical to the previous 64x64 run.
    """
    if H >= 2048:
        if H % 8192 == 0:
            return 1, 8192, 8
        if H % 4096 == 0:
            return 1, 4096, 8
        return 4, 2048, 8
    if H >= 256:
        return 1, 1024, 8
    if H == 1:
        return 8, 128, 4
    return 8, 512, 4


def dgeglu(
    grad_output: torch.Tensor,
    input_tensor: torch.Tensor,
    quantizer: Optional[Any] = None,
) -> torch.Tensor:
    shape = input_tensor.shape
    H = shape[-1] // 2
    M = input_tensor.numel() // (2 * H)

    grad_out_2d = grad_output.contiguous().view(M, H)
    input_2d = input_tensor.contiguous().view(M, 2 * H)
    grad_in_2d = torch.empty_like(input_2d)

    block_m, block_h, num_warps = _pick_dgeglu_config(input_tensor.dtype, M, H)
    grid = (triton.cdiv(M, block_m), triton.cdiv(H, block_h))

    dgeglu_kernel[grid](
        grad_out_2d,
        input_2d,
        grad_in_2d,
        M,
        H,
        grad_out_2d.stride(0),
        grad_out_2d.stride(1),
        input_2d.stride(0),
        input_2d.stride(1),
        grad_in_2d.stride(0),
        grad_in_2d.stride(1),
        BLOCK_SIZE_M=block_m,
        BLOCK_SIZE_H=block_h,
        num_warps=num_warps,
    )
    # print(dgeglu)
    return grad_in_2d.view_as(input_tensor)
