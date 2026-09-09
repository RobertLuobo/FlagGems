# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

# NOTE: hand-written flat-1D kernels instead of the pointwise_dynamic codegens.
#  - The generic codegen (512-tile) was the baseline's slow path (~0.77x average:
#    per-element integer-division indexing + always-true predicated access).
#  - The vendor codegen (kunlunAutoGrid) derives an unbounded 1D tile
#    tile = next_power_of_2(cdiv(numel, 12)): (16,128,64,1280) and (4096,4096)
#    land on 2^24 entries and deterministically hang the device (NOC idle
#    timeout, observed 2026-09-08 on XPU 5; same red-zone hazard as
#    special_chebyshev_polynomial_w_out recorded the same day).
#  - Masked loads must NOT pass `other=` (mis-lowered on this backend: a handful
#    of interior lanes return `other`/zero -> U_n(0) garbage; the sibling
#    generic codegen also omits `other`).  Masked-off lanes load garbage but are
#    never stored (store is masked), which is safe.
# Math: U_0=1, U_1=2x, U_k=2x*U_{k-1}-U_{k-2}, selected per element by the
# (integer, guard-validated [0,5]) degree n; computed in fp32.

import logging

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)

_BLOCK = 2048
_NUM_WARPS = 8


@triton.jit
def _chebyshev_polynomial_u_tensor_n_kernel(
    x_ptr,
    n_ptr,
    out_ptr,
    numel,
    BLOCK: tl.constexpr,
    NEED_MASK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    if NEED_MASK:
        mask = offs < numel
        x = tl.load(x_ptr + offs, mask=mask)
        n_f32 = tl.load(n_ptr + offs, mask=mask).to(tl.float32)
    else:
        x = tl.load(x_ptr + offs)
        n_f32 = tl.load(n_ptr + offs).to(tl.float32)
    x_f32 = x.to(tl.float32)

    ukm2 = x_f32 * 0.0 + 1.0  # U_0
    ukm1 = 2.0 * x_f32  # U_1
    result = tl.where(n_f32 < 0.5, ukm2, ukm1)

    for k in tl.static_range(2, 6):
        uk = 2.0 * x_f32 * ukm1 - ukm2
        result = tl.where(tl.abs(n_f32 - k) < 0.5, uk, result)
        ukm2, ukm1 = ukm1, uk

    if NEED_MASK:
        tl.store(out_ptr + offs, result.to(x.dtype), mask=mask)
    else:
        tl.store(out_ptr + offs, result.to(x.dtype))


@triton.jit
def _chebyshev_polynomial_u_scalar_n_kernel(
    x_ptr,
    n_idx: tl.constexpr,
    out_ptr,
    numel,
    BLOCK: tl.constexpr,
    NEED_MASK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    if NEED_MASK:
        mask = offs < numel
        x = tl.load(x_ptr + offs, mask=mask)
    else:
        x = tl.load(x_ptr + offs)
    x_f32 = x.to(tl.float32)

    # n_idx is a compile-time constant (guard guarantees [0, 5]); the branch is
    # resolved at compile time, so there is no runtime degree selection at all.
    if n_idx == 0:
        result = x_f32 * 0.0 + 1.0
    elif n_idx == 1:
        result = 2.0 * x_f32
    elif n_idx == 2:
        t = x_f32 * x_f32
        result = 4.0 * t - 1.0
    elif n_idx == 3:
        t = x_f32 * x_f32
        result = (8.0 * t - 4.0) * x_f32
    elif n_idx == 4:
        t = x_f32 * x_f32
        result = (16.0 * t - 12.0) * t + 1.0
    else:  # n_idx == 5
        t = x_f32 * x_f32
        result = (32.0 * t - 32.0) * t * x_f32 + 6.0 * x_f32

    if NEED_MASK:
        tl.store(out_ptr + offs, result.to(x.dtype), mask=mask)
    else:
        tl.store(out_ptr + offs, result.to(x.dtype))


def _launch_flat(x, n, numel):
    out = torch.empty_like(x)
    grid = (triton.cdiv(numel, _BLOCK),)
    need_mask = (numel % _BLOCK) != 0
    if isinstance(n, torch.Tensor):
        _chebyshev_polynomial_u_tensor_n_kernel[grid](
            x, n, out, numel,
            BLOCK=_BLOCK, NEED_MASK=need_mask, num_warps=_NUM_WARPS,
        )
    else:
        _chebyshev_polynomial_u_scalar_n_kernel[grid](
            x, n, out, numel,
            BLOCK=_BLOCK, NEED_MASK=need_mask, num_warps=_NUM_WARPS,
        )
    return out


def special_chebyshev_polynomial_u(x, n):
    logger.debug("GEMS_KUNLUNXIN SPECIAL_CHEBYSHEV_POLYNOMIAL_U")
    if x.dtype not in (torch.float32, torch.float64):
        raise ValueError(
            f"special_chebyshev_polynomial_u only supports float32/float64, got {x.dtype}"
        )
    if x.numel() == 0:
        return torch.empty_like(x)
    x = x.contiguous()

    if isinstance(n, torch.Tensor):
        # Range guard on a CPU copy (avoid dispatching back into gems lt/gt).
        n_ref = n.detach().to("cpu", dtype=torch.int32)
        n_min = int(n_ref.amin().item())
        n_max = int(n_ref.amax().item())
        n = n.to(device=x.device, dtype=torch.int32)
    else:
        n_min = n_max = int(n)

    if n_max > 5 or n_min < 0:
        raise ValueError(
            f"Chebyshev polynomial order n must be in [0, 5], "
            f"got values in [{n_min}, {n_max}]"
        )

    return _launch_flat(x, n, x.numel())