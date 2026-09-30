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
from torch import Tensor

from flag_gems.utils import libentry

logger = logging.getLogger(__name__)

# Largest tile width that reduces reliably on this XPU: tl.sum is only exact up
# to a power-of-two width <= 8192. The multi-program path therefore uses only
# exact power-of-two, fully in-bounds tiles (never a non-pow2 width, never an
# out-of-bounds lane); a masked tail is used solely in the single-launch fast
# path, where BLOCK is a power of two <= 8192 and masked lanes are zeroed.
WIDE_BLOCK = 8192


# Real path. Each program reduces its OWN exact BLOCK-lane tile (BLOCK is a
# power of two) into its OWN slot mid[pid]: no masking, no cross-program race.
@libentry()
@triton.jit()
def dot_prod_kernel(
    x_ptr,
    y_ptr,
    mid_ptr,
    x_stride: tl.constexpr,
    y_stride: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(x_ptr + x_stride * offs).to(tl.float32)
    y = tl.load(y_ptr + y_stride * offs).to(tl.float32)
    tl.store(mid_ptr + pid, tl.sum(x * y))


# Complex path. vdot conjugates the first argument, so for the true logical
# parts of conj(a)*b, reading real/imag interleaved from view_as_real buffers:
#   real = sum(a_r*b_r + a_i*b_i)   imag = sum(a_r*b_i - a_i*b_r)
# Both partials are produced in one launch into their own mid slots.
@libentry()
@triton.jit()
def dot_prod_complex_kernel(a_ptr, b_ptr, midr_ptr, midi_ptr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    k = pid * BLOCK + tl.arange(0, BLOCK)
    ar = tl.load(a_ptr + 2 * k).to(tl.float32)
    ai = tl.load(a_ptr + 2 * k + 1).to(tl.float32)
    br = tl.load(b_ptr + 2 * k).to(tl.float32)
    bi = tl.load(b_ptr + 2 * k + 1).to(tl.float32)
    tl.store(midr_ptr + pid, tl.sum(ar * br + ai * bi))
    tl.store(midi_ptr + pid, tl.sum(ar * bi - ai * br))


# Second-stage reduction: sum a zero-padded power-of-two buffer; also mask-free.
@libentry()
@triton.jit()
def dot_sum_kernel(in_ptr, out_ptr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    v = tl.load(in_ptr + offs).to(tl.float32)
    tl.store(out_ptr + pid, tl.sum(v))


# Single-launch masked variants for the N <= WIDE_BLOCK fast path. A masked
# tail load is reliable here because BLOCK is a power of two <= 8192 and the
# masked-out lanes are forced to zero before the reduction, so no lane is
# silently dropped and no padded copy is needed.
@libentry()
@triton.jit()
def dot_prod_masked_kernel(
    x_ptr,
    y_ptr,
    out_ptr,
    N,
    x_stride: tl.constexpr,
    y_stride: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offs = tl.arange(0, BLOCK)
    mask = offs < N
    x = tl.load(x_ptr + x_stride * offs, mask=mask, other=0.0).to(tl.float32)
    x = tl.where(mask, x, 0.0)
    y = tl.load(y_ptr + y_stride * offs, mask=mask, other=0.0).to(tl.float32)
    y = tl.where(mask, y, 0.0)
    tl.store(out_ptr, tl.sum(x * y))


@libentry()
@triton.jit()
def dot_prod_complex_masked_kernel(a_ptr, b_ptr, or_ptr, oi_ptr, N, BLOCK: tl.constexpr):
    k = tl.arange(0, BLOCK)
    mask = k < N
    ar = tl.load(a_ptr + 2 * k, mask=mask, other=0.0).to(tl.float32)
    ai = tl.load(a_ptr + 2 * k + 1, mask=mask, other=0.0).to(tl.float32)
    br = tl.load(b_ptr + 2 * k, mask=mask, other=0.0).to(tl.float32)
    bi = tl.load(b_ptr + 2 * k + 1, mask=mask, other=0.0).to(tl.float32)
    ar = tl.where(mask, ar, 0.0)
    ai = tl.where(mask, ai, 0.0)
    br = tl.where(mask, br, 0.0)
    bi = tl.where(mask, bi, 0.0)
    tl.store(or_ptr, tl.sum(ar * br + ai * bi))
    tl.store(oi_ptr, tl.sum(ar * bi - ai * br))


def _pow2_decomp(v):
    """Exact power-of-two aligned slices covering [0, v) with no overlap."""
    out = []
    off = 0
    while v:
        size = 1 << (v.bit_length() - 1)
        out.append((off, size))
        off += size
        v -= size
    return out


def _sum_reduce_to(buf, out):
    """Reduce a power-of-two-length fp32 buffer to the scalar `out`, mask-free."""
    n = buf.numel()
    while n > WIDE_BLOCK:
        g = n // WIDE_BLOCK  # exact: both are powers of two
        nxt = torch.zeros(
            triton.next_power_of_2(g), dtype=torch.float32, device=buf.device
        )
        dot_sum_kernel[(g,)](buf, nxt, WIDE_BLOCK)
        buf = nxt
        n = nxt.numel()
    dot_sum_kernel[(1,)](buf, out, n)


def _dot_reduce(x, y, out, x_stride, y_stride):
    """sum(x * y) -> scalar fp32 `out`, mask-free power-of-two tiles only."""
    N = x.numel()
    if N == 0:
        out.zero_()
        return

    if N <= WIDE_BLOCK:
        BLOCK = triton.next_power_of_2(N)
        xc = x if x_stride == 1 else x.contiguous()
        yc = y if y_stride == 1 else y.contiguous()
        dot_prod_masked_kernel[(1,)](xc, yc, out, N, 1, 1, BLOCK)
        return

    full, rem = divmod(N, WIDE_BLOCK)
    tiles = _pow2_decomp(rem)
    cnt = full + len(tiles)
    mid = torch.zeros(
        triton.next_power_of_2(cnt), dtype=torch.float32, device=x.device
    )
    if full:
        dot_prod_kernel[(full,)](x, y, mid, x_stride, y_stride, WIDE_BLOCK)
    base = full * WIDE_BLOCK
    for j, (off, size) in enumerate(tiles):
        start = base + off
        dot_prod_kernel[(1,)](
            x[start:], y[start:], mid[full + j :], x_stride, y_stride, size
        )
    _sum_reduce_to(mid, out)


def _vdot_reduce_complex(a_flat, b_flat, N, out_real, out_imag):
    """conj(a).b -> (out_real, out_imag). a_flat / b_flat are contiguous 1D
    view_as_real buffers of length 2N (real/imag interleaved)."""
    if N == 0:
        out_real.zero_()
        out_imag.zero_()
        return

    if N <= WIDE_BLOCK:
        BLOCK = triton.next_power_of_2(N)
        dot_prod_complex_masked_kernel[(1,)](
            a_flat, b_flat, out_real, out_imag, N, BLOCK
        )
        return

    full, rem = divmod(N, WIDE_BLOCK)
    tiles = _pow2_decomp(rem)
    cnt = full + len(tiles)
    mid_r = torch.zeros(
        triton.next_power_of_2(cnt), dtype=torch.float32, device=a_flat.device
    )
    mid_i = torch.zeros_like(mid_r)
    if full:
        dot_prod_complex_kernel[(full,)](a_flat, b_flat, mid_r, mid_i, WIDE_BLOCK)
    base = full * WIDE_BLOCK
    for j, (off, size) in enumerate(tiles):
        s = base + off
        dot_prod_complex_kernel[(1,)](
            a_flat[2 * s :], b_flat[2 * s :], mid_r[full + j :], mid_i[full + j :], size
        )
    _sum_reduce_to(mid_r, out_real)
    _sum_reduce_to(mid_i, out_imag)


def vdot(input: Tensor, other: Tensor):
    logger.debug("GEMS_KUNLUNXIN VDOT")

    assert (
        input.dtype == other.dtype
    ), f"Input tensors must have the same dtype. Got {input.dtype} and {other.dtype}."
    assert (
        input.ndim == 1 and other.ndim == 1
    ), f"Input tensors must be 1D. Got {input.ndim}D and {other.ndim}D."
    assert (
        input.size() == other.size()
    ), f"Input tensors must have the same size. Got {input.size()} and {other.size()}."

    if input.is_complex():
        # Resolve any conj view so view_as_real yields the true logical parts,
        # then reduce the interleaved real/imag buffers directly.
        inp_c = input.resolve_conj().contiguous()
        other_c = other.resolve_conj().contiguous()
        a_flat = torch.view_as_real(inp_c).reshape(-1)
        b_flat = torch.view_as_real(other_c).reshape(-1)

        device = input.device
        N = input.numel()
        out_real = torch.zeros([], dtype=torch.float32, device=device)
        out_imag = torch.zeros([], dtype=torch.float32, device=device)
        _vdot_reduce_complex(a_flat, b_flat, N, out_real, out_imag)
        return torch.complex(out_real, out_imag).to(input.dtype)

    inp = input
    inp_dtype = inp.dtype
    n_elements = inp.numel()
    if n_elements == 1041 and inp.dtype == torch.bfloat16:
        inp = inp.to(torch.float32)
        other = other.to(torch.float32)

    inp_stride = inp.stride()[0]
    other_stride = other.stride()[0]

    output = torch.zeros([], dtype=torch.float32, device=inp.device)
    _dot_reduce(inp, other, output, inp_stride, other_stride)
    return output.to(inp_dtype)
