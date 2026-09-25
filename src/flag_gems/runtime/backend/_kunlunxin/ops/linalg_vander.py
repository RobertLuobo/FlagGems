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

# Kunlunxin (XPU/P800) override for torch.linalg.vander.
#
# The generic implementation (flag_gems/ops/linalg_vander.py) uses a polar-form
# complex power that calls libdevice-style externs tl_extra_shim.{cos,sin,atan2}.
# On the TritonXPU backend those externs lower to an ``llvm.call`` with the wrong
# operand count and the MLIR pass ``ConvertTritonXPUToLLVM`` crashes at compile
# time ("'llvm.call' op incorrect number of operands (1) for callee (expecting:
# 2)") -> all 60 complex cases fail before any numerical comparison.
#
# The real (float) kernel uses tl_extra_shim.pow, which compiles and passes on
# XPU, so it is kept verbatim. The complex kernel is rewritten to compute
# (a + b j)**p with integer p in [0, N) via exponentiation-by-squaring (pure
# complex multiplication). This avoids every crashing extern and is also exact
# for integer exponents (better than the polar form).

import logging

import torch
import triton
import triton.language as tl

from flag_gems.utils import tl_extra_shim

logger = logging.getLogger(__name__)


@triton.jit
def vander_kernel(
    x_ptr,
    out_ptr,
    N,
    total_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total_elements

    # offsets map to flat output: out_flat[k*N + j] = x_flat[k] ** j
    col = offsets % N
    row = offsets // N

    x_val = tl.load(x_ptr + row, mask=mask)
    # Compute in the output dtype: fp64 for float64, fp32 otherwise (fp16/bf16
    # are upcast to fp32 since libdevice pow is not accurate at low precision).
    compute_dtype = tl.float64 if out_ptr.dtype.element_ty == tl.float64 else tl.float32
    result = tl_extra_shim.pow(x_val.to(compute_dtype), col.to(compute_dtype))

    tl.store(out_ptr + offsets, result, mask=mask)


@triton.jit
def vander_complex_kernel(
    x_ri_ptr,
    out_ri_ptr,
    N,
    total_elements,
    BLOCK_SIZE: tl.constexpr,
    STORE_FP64: tl.constexpr,
    POW_BITS: tl.constexpr,
):
    """Complex vander: each output element is (a + b j)**p with integer p in [0, N).

    Instead of the polar form (r**p * (cos(p*theta) + j sin(p*theta))), which on
    XPU requires the crashing tl_extra_shim.{cos,sin,atan2} externs, this computes
    the integer power directly by exponentiation-by-squaring using complex
    multiplication. This is exact for integer exponents and touches no externs.

    Inputs/outputs are viewed as interleaved real/imag float arrays via
    torch.view_as_real so the kernel only handles real dtypes (Triton cannot
    specialise on complex tensors).
    """
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total_elements

    # offsets index the flat *complex* output; each complex element occupies
    # two consecutive floats in the view_as_real layout.
    col = offsets % N  # exponent p (integer)
    row = offsets // N  # index into the flat complex input

    base = row * 2
    real = tl.load(x_ri_ptr + base, mask=mask, other=0.0)
    imag = tl.load(x_ri_ptr + base + 1, mask=mask, other=0.0)

    # Carry the arithmetic in the widest available float (fp64 requested; the
    # XPU stack silently runs it as fp32, matching the generic intent).
    base_re = real.to(tl.float64)
    base_im = imag.to(tl.float64)

    # result = 1 + 0 j; exponentiation by squaring over the bits of col.
    res_re = tl.zeros_like(base_re) + 1.0
    res_im = tl.zeros_like(base_im)
    c = col
    for _ in tl.static_range(POW_BITS):
        odd = (c & 1) == 1
        # result *= base   (only when the current bit is set)
        nr = res_re * base_re - res_im * base_im
        ni = res_re * base_im + res_im * base_re
        res_re = tl.where(odd, nr, res_re)
        res_im = tl.where(odd, ni, res_im)
        # base *= base
        br = base_re * base_re - base_im * base_im
        bi = 2.0 * base_re * base_im
        base_re = br
        base_im = bi
        c = c >> 1

    # col == 0 must give exactly 1 + 0 j; the loop already yields that (result
    # stays untouched when every bit is zero), so no special-casing is needed.
    store_dtype = tl.float64 if STORE_FP64 else tl.float32
    out_real_store = res_re.to(store_dtype)
    out_imag_store = res_im.to(store_dtype)

    out_base = offsets * 2
    tl.store(out_ri_ptr + out_base, out_real_store, mask=mask)
    tl.store(out_ri_ptr + out_base + 1, out_imag_store, mask=mask)


def linalg_vander(x, N=None):
    logger.debug("GEMS LINALG_VANDER")

    # fmt: off
    assert x.dtype.is_floating_point or x.dtype.is_complex, f"Unsupported dtype {x.dtype}"
    # fmt: on

    # Handle N parameter
    if N is None:
        N = x.shape[-1]

    # Get input shape info
    batch_dims = x.shape[:-1]
    n = x.shape[-1]

    # Flatten batch dims
    x_flat = x.reshape(-1)

    # Output shape: (*, n, N)
    final_shape = batch_dims + (n, N)

    # Run triton kernel
    total_elements = x_flat.numel() * N
    BLOCK_SIZE = 256
    grid = (triton.cdiv(total_elements, BLOCK_SIZE),)

    if x.is_complex():
        # Triton cannot specialise on complex tensor pointers, so we view the
        # complex input/output as interleaved real/imag float arrays and run
        # a kernel that operates purely on real dtypes. Allocate the complex
        # output via torch.empty_strided (not patched by flag_gems) since the
        # patched torch.empty kernel does not accept complex dtypes either.
        x_ri = torch.view_as_real(x_flat.contiguous()).reshape(-1)

        # Build contiguous strides for final_shape: strides[i] = prod(shape[i+1:]).
        strides = []
        stride = 1
        for dim in reversed(final_shape):
            strides.append(stride)
            stride *= dim
        strides.reverse()

        out = torch.empty_strided(
            final_shape, tuple(strides), dtype=x.dtype, device=x.device
        )
        out_flat = out.reshape(-1)
        out_ri = torch.view_as_real(out_flat).reshape(-1)

        store_fp64 = x.dtype == torch.complex128
        # Enough squaring iterations to cover any exponent in [0, N).
        pow_bits = max(1, int(N - 1).bit_length())
        vander_complex_kernel[grid](
            x_ri,
            out_ri,
            N,
            total_elements,
            BLOCK_SIZE=BLOCK_SIZE,
            STORE_FP64=store_fp64,
            POW_BITS=pow_bits,
        )
    else:
        out = torch.empty(final_shape, dtype=x.dtype, device=x.device)
        vander_kernel[grid](x_flat, out, N, total_elements, BLOCK_SIZE=BLOCK_SIZE)

    return out
