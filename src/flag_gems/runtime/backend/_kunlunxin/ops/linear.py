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
#
# Kunlunxin(XPU) specific linear override.
#
# y = x @ W^T + b  with x (M, K), W (N, K), b (N,), y (M, N).
#
# History: the initial vendor kernel (batch 20260908) loaded the W tile as a
# (BLOCK_K, BLOCK_N) block indexed by (k, n) = k*stride_wk + n*stride_wn with
# stride_wk = 1 / stride_wn = K, i.e. column-major (non-coalesced, one scalar
# load per lane) accesses.  That costs ~3x vs the vendor BLAS on the
# benchmark's large GEMMs (e.g. fp16 4096^3: 1.97ms vs 0.63ms) and puts the
# equal-weight mean far below the 0.8 bar.  It also autotuned 8 configs per
# unique (M, N, K) key (~360 autotune launches for the benchmark matrix),
# which is why a single benchmark run took ~39 minutes.
#
# This variant loads the W tile as a (BLOCK_N, BLOCK_K) block (row-major over
# W, coalesced / block-DMA-able) and transposes it in registers for the dot
# (tl.trans; a pattern already in use by _kunlunxin/ops/attention.py).  Fixed
# mm.py-style tiles (128^3 w4 for M,N <= 512, 256^3 w8 otherwise) replace the
# autotuner, so nothing is shape-compiled on the hot path and the launch cost
# stays inside the vendor-BLAS ballpark.  EVEN is a tl.constexpr so each
# specialization compiles only one load path; fp32 accumulate and
# allow_tf32=False are kept (same numerical semantics as the generic kernel).
# The bf16 XMLIR_MATMUL_FAST_MODE flag mirrors mm.py for large-K GEMMs.
#
# Closure fix (2026-09-11): the transposed-kernel variant above cannot compile
# bf16 shapes with K >= 2048 on this backend ('tt.trans' element-type mismatch
# / OutOfResources: uni_sram; reproduced for 128^3 and 256^3 tiles, with and
# without XMLIR_MATMUL_FAST_MODE, and by a clamp+where bias rewrite which
# produced a device-side illegal memory access).  The batch-20260908 kernel
# below (trans-free, autotuned) is baseline-proven for every benchmark shape,
# so these shapes dispatch to it; the trans kernel still serves all fp16/fp32
# and bf16 K < 2048 shapes.
#
# Closure fix 2 (2026-09-11): the trans kernel additionally HANGS the device
# at launch for fp32 shapes with K >= 32768 (deterministic
# cudaErrorLaunchTimeout -> kl3 status-702 after ~15-20 min of bus-wait;
# reproduced on cards 6 and 7 for K = 32768/65536/128256/151936/152064, all
# with M = 1848 masked + N = 1536; K <= 18944 verified OK on the same
# M-masked pattern; fp16 huge-K (e.g. 151936) is unaffected and passes) and
# for large input sizes M*K >= 2**25 (e.g. (8192, 3584, 18944) = 1.55e8;
# (8192, 3584, 4096) = 2^25 exactly is the largest verified-OK trans shape
# family, while (1848, 1536, 32768) = 6.05e7 hangs).  The M*K rule is
# dtype-fp32 specific: the identical shapes pass for fp16 (242/242
# benchmark).  All these fp32 shapes also dispatch to the legacy kernel.
import logging
import os

import torch
import triton
import triton.language as tl

from flag_gems import runtime
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry, libtuner

logger = logging.getLogger(__name__)

_FAST_MODE_ENV = "XMLIR_MATMUL_FAST_MODE"


def _set_matmul_fast_mode(a_dtype, M, N, K):
    """Mirror of mm.py: XMLIR_MATMUL_FAST_MODE=1 speeds the bf16 tl.dot
    lowering for large-K GEMMs (K >= 2048, M/N >= 128); small bf16 shapes
    regress, so apply it selectively."""
    if a_dtype == torch.bfloat16 and K >= 2048 and M >= 128 and N >= 128:
        saved = os.environ.get(_FAST_MODE_ENV)
        os.environ[_FAST_MODE_ENV] = "1"
        return saved
    return None


def _restore_matmul_fast_mode(saved):
    if saved is None:
        os.environ.pop(_FAST_MODE_ENV, None)
    else:
        os.environ[_FAST_MODE_ENV] = saved


@libentry()
@triton.jit
def linear_kernel(
    input_ptr,
    weight_ptr,
    bias_ptr,
    output_ptr,
    M,
    N,
    K,
    stride_im,
    stride_ik,
    stride_wn,
    stride_wk,
    stride_om,
    stride_on,
    stride_bn,
    # Bias is present or not
    BIAS: tl.constexpr,
    # All tile dimensions divide the shape exactly: masks are all-true
    EVEN: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    """
    Linear kernel: y = x @ W^T + b
    - input: (M, K) where M is batch size (flattened), K is in_features
    - weight: (N, K) where N is out_features
    - bias: (N,) optional
    - output: (M, N)

    W is row-major (N, K): load the (BLOCK_N, BLOCK_K) tile in its natural
    (coalesced) order and transpose in registers for the dot.  Loading the
    (BLOCK_K, BLOCK_N) tile directly (the pre-20260909 indexing) is one scalar
    load per lane on this backend (inner stride K) and costs ~3x.
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    input_ptrs = input_ptr + (offs_m[:, None] * stride_im + offs_k[None, :] * stride_ik)

    # W tile: (BLOCK_N, BLOCK_K), row-major -> coalesced; trans for the dot.
    weight_ptrs = weight_ptr + (
        offs_n[:, None] * stride_wn + offs_k[None, :] * stride_wk
    )

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        if EVEN:
            # Aligned (M%BLOCK_M==0, N%BLOCK_N==0, K%BLOCK_K==0): every mask is
            # all-true; unmasked block loads lower to block DMA instead of the
            # per-lane select/predication of the generic masked path.
            a = tl.load(input_ptrs)
            w = tl.load(weight_ptrs)
        else:
            # Load input block
            input_mask_m = offs_m < M
            input_mask_k = offs_k < (K - k * BLOCK_SIZE_K)
            input_mask = input_mask_m[:, None] & input_mask_k[None, :]

            a = tl.load(input_ptrs, mask=input_mask, other=0.0)

            # Load weight block
            weight_mask_k = offs_k < (K - k * BLOCK_SIZE_K)
            weight_mask_n = offs_n < N
            weight_mask = weight_mask_n[:, None] & weight_mask_k[None, :]

            w = tl.load(weight_ptrs, mask=weight_mask, other=0.0)

        b = tl.trans(w)

        # Compute dot product
        accumulator += tl.dot(a, b, allow_tf32=False)

        # Move to next block
        input_ptrs += BLOCK_SIZE_K * stride_ik
        weight_ptrs += BLOCK_SIZE_K * stride_wk

    # Compute output offset
    offs_om = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_on = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    output_ptrs = output_ptr + (
        offs_om[:, None] * stride_om + offs_on[None, :] * stride_on
    )

    # Add bias if present
    if BIAS:
        bias_ptrs = bias_ptr + offs_on
        if EVEN:
            bias = tl.load(bias_ptrs)
        else:
            bias = tl.load(bias_ptrs, mask=offs_n < N, other=0.0)
        accumulator = accumulator + bias

    # Store result
    output = accumulator.to(output_ptr.dtype.element_ty)
    if EVEN:
        tl.store(output_ptrs, output)
    else:
        output_mask_m = offs_m < M
        output_mask_n = offs_n < N
        output_mask = output_mask_m[:, None] & output_mask_n[None, :]
        tl.store(output_ptrs, output, mask=output_mask)


# Fixed tiles (mm.py 2026-08-13 XPU probe): 128^3 w4 for M,N <= 512
# (launch-bound preferred), 256^3 w8 otherwise.  EVEN is computed against the
# launched blocks, so the unmasked path never leaves the allocation.
def _block_m(M):
    return 128 if M <= 512 else 256


def _block_n(N):
    return 128 if N <= 512 else 256


def _block_k(M, N):
    if M <= 512 and N <= 512:
        return 128
    return 256


def _num_warps(M, N):
    return 4 if (M <= 512 and N <= 512) else 8


def linear(input, weight, bias=None):
    """
    Applies a linear transformation to the incoming data: y = xA^T + b

    Args:
        input: Input tensor of shape (*, in_features) where * means any number of
               additional dimensions, including none.
        weight: Weight tensor of shape (out_features, in_features)
        bias: Bias tensor of shape (out_features), optional

    Returns:
        Output tensor of shape (*, out_features)
    """
    logger.debug("GEMS KUNLUNXIN LINEAR")

    # bf16 with K >= 2048 cannot compile the transposed-tile kernel (see the
    # header note); fp32 with K >= 32768 or M*K >= 2**25 hangs the device at
    # launch (same note).  Both take the baseline-proven autotuned kernel.
    if (input.dtype == torch.bfloat16 and input.shape[-1] >= 2048) or (
        input.dtype == torch.float32
        and (input.shape[-1] >= 32768 or input.numel() >= 2**25)
    ):
        return _linear_legacy_kernel(input, weight, bias)

    input_dim = input.dim()
    if input_dim == 1:
        # Single 1D input: treat as (1, in_features)
        input = input.unsqueeze(0)
        single_1d = True
    else:
        single_1d = False

    # Flatten batch dimensions: (*, in_features) -> (batch, in_features)
    batch_dims = input.shape[:-1]
    batch_size = 1
    for dim in batch_dims:
        batch_size *= dim
    M = batch_size
    K = input.shape[-1]  # in_features
    N = weight.shape[0]  # out_features

    # Flatten input: (*, K) -> (M, K)
    input_flat = input.view(M, K)

    # Ensure weight is contiguous and properly shaped
    weight = weight.contiguous()

    # Allocate output
    output = torch.empty((M, N), device=input.device, dtype=input.dtype)

    # Launch kernel
    blk_m = _block_m(M)
    blk_n = _block_n(N)
    blk_k = _block_k(M, N)
    num_warps = _num_warps(M, N)
    grid = lambda META: (
        triton.cdiv(M, blk_m),
        triton.cdiv(N, blk_n),
    )

    saved = _set_matmul_fast_mode(input.dtype, M, N, K)
    try:
        with torch_device_fn.device(input.device):
            linear_kernel[grid](
                input_flat,
                weight,
                bias if bias is not None else weight,  # Pass dummy ptr if no bias
                output,
                M,
                N,
                K,
                input_flat.stride(0),
                input_flat.stride(1),
                weight.stride(0),
                weight.stride(1),
                output.stride(0),
                output.stride(1),
                bias.stride(0) if bias is not None else 0,
                BIAS=bias is not None,
                EVEN=(
                    (M % blk_m == 0) and (N % blk_n == 0) and (K % blk_k == 0)
                ),
                BLOCK_SIZE_M=blk_m,
                BLOCK_SIZE_N=blk_n,
                BLOCK_SIZE_K=blk_k,
                num_warps=num_warps,
            )
    finally:
        _restore_matmul_fast_mode(saved)

    # Reshape output: (M, N) -> (*, N)
    output = output.view(*batch_dims, N)

    # If original input was 1D, squeeze the batch dim
    if single_1d:
        output = output.squeeze(0)

    return output


# ---------------------------------------------------------------------------
# Batch-20260908 vendor kernel (trans-free, autotuned) kept verbatim as the
# bf16 K >= 2048 fallback; it is the exact code that produced the 726/726
# baseline benchmark run.
# ---------------------------------------------------------------------------

@libentry()
@libtuner(
    configs=runtime.get_tuned_config("linear"),
    key=["M", "N", "K"],
    strategy=["align32", "align32", "align32"],
    warmup=5,
    rep=10,
)
@triton.jit
def linear_kernel_legacy(
    input_ptr,
    weight_ptr,
    bias_ptr,
    output_ptr,
    M,
    N,
    K,
    stride_im,
    stride_ik,
    stride_wn,
    stride_wk,
    stride_om,
    stride_on,
    stride_bn,
    # Bias is present or not
    BIAS: tl.constexpr,
    # All tile dimensions divide the shape exactly: masks are all-true
    EVEN: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    input_ptrs = input_ptr + (offs_m[:, None] * stride_im + offs_k[None, :] * stride_ik)

    weight_ptrs = weight_ptr + (
        offs_k[:, None] * stride_wk + offs_n[None, :] * stride_wn
    )

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        if EVEN:
            a = tl.load(input_ptrs)
            b = tl.load(weight_ptrs)
        else:
            input_mask_m = offs_m < M
            input_mask_k = offs_k < (K - k * BLOCK_SIZE_K)
            input_mask = input_mask_m[:, None] & input_mask_k[None, :]

            a = tl.load(input_ptrs, mask=input_mask, other=0.0)

            weight_mask_k = offs_k < (K - k * BLOCK_SIZE_K)
            weight_mask_n = offs_n < N
            weight_mask = weight_mask_k[:, None] & weight_mask_n[None, :]

            b = tl.load(weight_ptrs, mask=weight_mask, other=0.0)

        accumulator += tl.dot(a, b, allow_tf32=False)

        input_ptrs += BLOCK_SIZE_K * stride_ik
        weight_ptrs += BLOCK_SIZE_K * stride_wk

    offs_om = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_on = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    output_ptrs = output_ptr + (
        offs_om[:, None] * stride_om + offs_on[None, :] * stride_on
    )

    if BIAS:
        bias_ptrs = bias_ptr + offs_on
        if EVEN:
            bias = tl.load(bias_ptrs)
        else:
            bias = tl.load(bias_ptrs, mask=offs_n < N, other=0.0)
        accumulator = accumulator + bias

    output = accumulator.to(output_ptr.dtype.element_ty)
    if EVEN:
        tl.store(output_ptrs, output)
    else:
        output_mask_m = offs_m < M
        output_mask_n = offs_n < N
        output_mask = output_mask_m[:, None] & output_mask_n[None, :]
        tl.store(output_ptrs, output, mask=output_mask)


def _even(matrix, block):
    return (matrix % block) == 0


# Max tile sizes of the autotune config set returned by
# runtime.get_tuned_config("linear"): BLOCK_SIZE_M in {32,64,128},
# BLOCK_SIZE_N in {32,64,128,256}, BLOCK_SIZE_K in {32,64}.  EVEN may only be
# True when the shape is divisible by the LARGEST tile, because any of the 8
# configs can be selected for an EVEN=True shape; otherwise the BLOCK_N=256 /
# BLOCK_M=128 config would issue unmasked out-of-bounds loads.
_EVEN_M = 128
_EVEN_N = 256
_EVEN_K = 64


def _linear_legacy_kernel(input, weight, bias=None):
    """Verbatim batch-20260908 vendor wrapper (autotuned trans-free kernel)."""
    input_dim = input.dim()
    if input_dim == 1:
        input = input.unsqueeze(0)
        single_1d = True
    else:
        single_1d = False

    batch_dims = input.shape[:-1]
    batch_size = 1
    for dim in batch_dims:
        batch_size *= dim
    M = batch_size
    K = input.shape[-1]  # in_features
    N = weight.shape[0]  # out_features

    input_flat = input.view(M, K)
    weight = weight.contiguous()

    output = torch.empty((M, N), device=input.device, dtype=input.dtype)

    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_SIZE_M"]),
        triton.cdiv(N, META["BLOCK_SIZE_N"]),
    )

    with torch_device_fn.device(input.device):
        linear_kernel_legacy[grid](
            input_flat,
            weight,
            bias if bias is not None else weight,  # Pass dummy ptr if no bias
            output,
            M,
            N,
            K,
            input_flat.stride(0),
            input_flat.stride(1),
            weight.stride(0),
            weight.stride(1),
            output.stride(0),
            output.stride(1),
            bias.stride(0) if bias is not None else 0,
            BIAS=bias is not None,
            EVEN=(
                _even(M, _EVEN_M) and _even(N, _EVEN_N) and _even(K, _EVEN_K)
            ),
        )

    output = output.view(*batch_dims, N)

    if single_1d:
        output = output.squeeze(0)

    return output