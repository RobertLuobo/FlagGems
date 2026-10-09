# Copyright 2026, The FlagOS Contributors.
# SPDX-License-Identifier: Apache-2.0

"""Kunlunxin(XPU) backend override for aten::_dyn_quant_matmul_4bit.

Semantics are identical to flag_gems.ops._dyn_quant_matmul_4bit. Two
XPU-specific codegen workarounds are applied:

1. Masked tl.load with 64-bit pointer arithmetic drops its fill: masked lanes
   keep out-of-bounds garbage instead of ``other``. Every masked load clamps
   its index to an in-bounds lane and applies ``tl.where`` afterwards.
2. A single kernel that scans the activation rows twice (range reduction, then
   quantize-and-store) miscompiles the first-pass max reduction (FLT_MAX leaks
   in, collapsing the scale to ~1e36 and producing NaN output). The activation
   quantizer is split into two single-pass kernels (_quantize_stats computes
   scale/offset/multiplier, _quantize_apply writes the INT8 rows).
"""

import logging
from contextlib import nullcontext

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn

logger = logging.getLogger(__name__)


@triton.jit
def _quantization_divide(
    a,  # Dividend.
    b,  # Divisor.
):
    # Recover the residual with an explicit FMA before rounding the quotient.
    q = tl.div_rn(a, b)
    residual = tl.fma(-q, b, a)
    corrected = tl.fma(residual, tl.div_rn(1.0, b), q)
    # Preserve a/inf == 0 instead of the NaN residual from 0*inf.
    return tl.where(residual == residual, corrected, q)


@triton.jit
def _quantize_stats(
    X,  # Input activations [M, K].
    Scale,  # Activation dequantization scale per row.
    Offset,  # Negative activation zero point per row.
    Mult,  # Activation quantization multiplier per row (scratch for apply pass).
    M,  # Number of input rows.
    K: tl.constexpr,  # Number of input features.
    SM: tl.constexpr,  # Input row stride, in elements.
    SK: tl.constexpr,  # Input feature stride, in elements.
    B: tl.constexpr,  # Features processed per quantization tile.
):
    cols = tl.arange(0, B)
    lanes = tl.arange(0, 8)
    # One program per row, no runtime loop: a tl.min/tl.max inside a grid-stride loop corrupts later rows here.
    row = tl.program_id(0)
    row64 = row.to(tl.int64)
    lo = tl.full((B,), 0.0, tl.float32)
    hi = tl.full((B,), 0.0, tl.float32)
    for start in range(tl.cdiv(K, B)):
        k = start * B + cols
        kmask = k < K
        k_safe = tl.where(kmask, k, 0)
        x = tl.load(X + row64 * SM + k_safe.to(tl.int64) * SK).to(tl.float32)
        x = tl.where(kmask, x, 0.0)
        lo = tl.minimum(lo, x)
        hi = tl.maximum(hi, x)
    rmin = tl.min(lo, 0)
    rmax = tl.max(hi, 0)
    multiplier = tl.where(rmin == rmax, 1.0, _quantization_divide(255.0, rmax - rmin))
    scale = tl.where(multiplier != 0.0, _quantization_divide(1.0, multiplier), 0.0)
    scaled_min = rmin * multiplier
    scaled_max = rmax * multiplier
    zp = tl.where(
        (-128.0 + scaled_min) + (127.0 + scaled_max) > 0.0,
        -128.0 - scaled_min,
        127.0 - scaled_max,
    )
    zp = tl.minimum(tl.maximum(zp, -128.0), 127.0)
    # lrintf: nearest, ties to even.
    lower = tl.floor(zp)
    frac = zp - lower
    zp_int = lower.to(tl.int32) + (
        (frac > 0.5) | ((frac == 0.5) & ((lower.to(tl.int32) & 1) != 0))
    ).to(tl.int32)
    # Masked single-lane stores: an unmasked 0-d/arange(0,1) store widens and overwrites neighboring rows here.
    keep = lanes < 1
    tl.store(Scale + row64 + lanes, tl.where(keep, scale, 0.0), keep)
    tl.store(Offset + row64 + lanes, tl.where(keep, -zp_int, 0), keep)
    tl.store(Mult + row64 + lanes, tl.where(keep, multiplier, 0.0), keep)


@triton.jit
def _quantize_apply(
    X,  # Input activations [M, K].
    Q,  # Quantized INT8 activations [M, K].
    Offset,  # Negative activation zero point per row.
    Mult,  # Activation quantization multiplier per row.
    M,  # Number of input rows.
    K: tl.constexpr,  # Number of input features.
    SM: tl.constexpr,  # Input row stride, in elements.
    SK: tl.constexpr,  # Input feature stride, in elements.
    B: tl.constexpr,  # Features processed per quantization tile.
):
    cols = tl.arange(0, B)
    row = tl.program_id(0)
    row64 = row.to(tl.int64)
    # Scalar loads: a length-1 vector multiplier/zero-point only broadcasts into lane 0 here, leaving the tile unscaled.
    multiplier = tl.load(Mult + row64)
    zp_int = -tl.load(Offset + row64)
    for start in range(tl.cdiv(K, B)):
        k = start * B + cols
        kmask = k < K
        k_safe = tl.where(kmask, k, 0)
        x = tl.load(X + row64 * SM + k_safe.to(tl.int64) * SK).to(tl.float32)
        x = tl.where(kmask, x, 0.0)
        v = x * multiplier
        rounded = tl.floor(tl.abs(v) + 0.5)
        rounded = tl.where(v < 0, -rounded, rounded).to(tl.int32)
        q = tl.minimum(tl.maximum(rounded + zp_int, -128), 127)
        tl.store(Q + row64 * K + k_safe, q.to(tl.int8), kmask)


@triton.jit
def _w4a8_matmul(
    Q,  # Quantized INT8 activations [M, K].
    Scale,  # Activation dequantization scale per row.
    Offset,  # Negative activation zero point per row.
    Packed,  # Portable weights, weight scales, and optional bias.
    Y,  # Output matrix [M, N].
    N: tl.constexpr,  # Number of output features.
    K: tl.constexpr,  # Number of input features.
    GROUP: tl.constexpr,  # Input features per weight scale group.
    HAS_BIAS: tl.constexpr,  # Whether the packed buffer includes bias.
    BK: tl.constexpr,  # Reduction features per accumulation tile.
):
    # Per-element 2D grid with no runtime loop; per-group integer tl.sum folded with each float weight scale; masked single-lane store.
    groups: tl.constexpr = K // GROUP
    weight_count: tl.constexpr = N * (K // 2)
    ks = tl.arange(0, BK)
    lanes = tl.arange(0, 8)
    row = tl.program_id(0)
    col = tl.program_id(1)
    row64 = row.to(tl.int64)
    col64 = col.to(tl.int64)
    offset = tl.load(Offset + row)
    scale = tl.load(Scale + row)
    total = 0.0
    for group in range(groups):
        ws = tl.load(Packed + weight_count + col64 * groups + group)
        iacc = tl.zeros((BK,), tl.int32)
        for chunk in range(tl.cdiv(GROUP, BK)):
            local_k = chunk * BK + ks
            k = group * GROUP + local_k
            valid_k = local_k < GROUP
            k_safe = tl.where(valid_k, k, 0)
            q = tl.load(Q + row64 * K + k_safe.to(tl.int64)).to(tl.int32)
            q = tl.where(valid_k, q, 0)
            centered = tl.where(valid_k, q + offset, 0)
            p_off = col64 * (K // 2) + (k_safe.to(tl.int64) // 2)
            packed = tl.load(Packed + p_off).to(tl.int32)
            w = ((packed >> ((k % 2) * 4)) & 15) - 8
            w = tl.where(valid_k, w, 0)
            iacc += tl.where(valid_k, centered * w, 0)
        total += tl.sum(iacc, 0).to(tl.float32) * ws
    total = total * scale
    if HAS_BIAS:
        total += tl.load(Packed + weight_count + N * groups + col64)
    keep = lanes < 1
    tl.store(Y + row64 * N + col64 + lanes, tl.where(keep, total, 0.0), keep)


def _dyn_quant_matmul_4bit(inp, packed_weights, block_size, in_features, out_features):
    """W4A8 linear using the FP32 portable _dyn_quant_pack_4bit_weight ABI."""
    if not torch.compiler.is_compiling():
        logger.debug("GEMS_KUNLUNXIN _DYN_QUANT_MATMUL_4BIT")
    if inp.ndim != 2:
        raise RuntimeError("inp must be two-dimensional")
    if inp.dtype not in (torch.float32, torch.bfloat16):
        raise RuntimeError("inp must have float32 or bfloat16 dtype")
    if in_features <= 0 or in_features % 2:
        raise RuntimeError("in_features must be positive and even")
    if out_features < 0:
        raise RuntimeError("out_features must be nonnegative")
    if inp.shape[1] != in_features:
        raise RuntimeError("inp.size(1) must equal in_features")
    if block_size <= 0 or (
        block_size != in_features
        and (block_size % 32 != 0 or in_features % block_size != 0)
    ):
        raise RuntimeError(
            "block_size must equal in_features or divide it as a multiple of 32"
        )
    if inp.dtype == torch.bfloat16 and block_size != in_features:
        raise RuntimeError("bfloat16 requires block_size == in_features")
    if packed_weights.dtype == torch.uint8:
        raise RuntimeError(
            "opaque uint8 packed_weights are unsupported; repack on the target device with FlagGems"
        )
    if packed_weights.dtype != torch.float32:
        raise RuntimeError("packed_weights must use the float32 portable ABI")
    if packed_weights.device != inp.device:
        raise RuntimeError("packed_weights and inp must be on the same device")
    if packed_weights.ndim != 1 or not packed_weights.is_contiguous():
        raise RuntimeError("packed_weights must be one-dimensional and contiguous")
    base = out_features * (in_features // 2 + in_features // block_size)
    if packed_weights.numel() not in (base, base + out_features):
        raise RuntimeError(
            "packed_weights length does not match weights, scales, and optional bias"
        )
    m = inp.shape[0]
    output = torch.empty((m, out_features), dtype=inp.dtype, device=inp.device)
    if m == 0 or out_features == 0:
        return output
    q = torch.empty((m, in_features), dtype=torch.int8, device=inp.device)
    scale = torch.empty((m,), dtype=torch.float32, device=inp.device)
    offset = torch.empty((m,), dtype=torch.int32, device=inp.device)
    mult = torch.empty((m,), dtype=torch.float32, device=inp.device)
    # Bounded vector tiles support rows larger than a single program's limits.
    quant_block = max(32, min(triton.next_power_of_2(in_features), 1024))
    matmul_block = max(32, min(triton.next_power_of_2(block_size), 1024))
    device_context = (
        nullcontext()
        if torch.compiler.is_compiling()
        else torch_device_fn.device(inp.device)
    )
    with device_context:
        _quantize_stats[(min(m, 65535),)](
            inp,
            scale,
            offset,
            mult,
            m,
            in_features,
            inp.stride(0),
            inp.stride(1),
            quant_block,
            enable_fp_fusion=False,
        )
        _quantize_apply[(min(m, 65535),)](
            inp,
            q,
            offset,
            mult,
            m,
            in_features,
            inp.stride(0),
            inp.stride(1),
            quant_block,
            enable_fp_fusion=False,
        )
        _w4a8_matmul[(m, out_features)](
            q,
            scale,
            offset,
            packed_weights,
            output,
            out_features,
            in_features,
            block_size,
            packed_weights.numel() == base + out_features,
            matmul_block,
            enable_fp_fusion=False,
        )
    return output
