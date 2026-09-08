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
import math

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn

logger = logging.getLogger(__name__)


# Flat 1D kernel over the ENTIRE output (all batch rows at once).
#
# ROOT CAUSE of the old slowness: the previous kernel wrapped every store index
# with `% W_out` ("modulo wrap") to avoid masked stores. On KunlunXin XPU that
# runtime modulo defeats OffsetAnalysis, so EVERY load/store degrades to the
# discrete per-element path (~1.2 GB/s), a ~470x penalty vs mask-based
# contiguous stores (see reflection_pad2d_perf_fix.md). Baseline big shape
# [32,64,2048] pad[3,5] measured ~14ms / speedup 0.002.
#
# Fix: flatten (b, w_out) into one linear output index `o` and store to `o`
# directly (provably stride-1 -> block DMA). A single boolean mask
# `o < total_out` handles the tail. Because the layout is one flat contiguous
# buffer, the only masked-out threads sit at the very end (o >= total_out) and
# could not corrupt a valid element even if not suppressed (and it is in fact
# suppressed here). This removes the "adjacent batch corruption" hazard that
# motivated the modulo wrap.
@triton.jit
def reflection_pad1d_kernel(
    in_ptr,
    out_ptr,
    W_in,
    pad_left,
    W_out,
    total_out,
    BLOCK: tl.constexpr,
    NEED_MASK: tl.constexpr = True,
):
    pid = tl.program_id(axis=0)
    o = pid * BLOCK + tl.arange(0, BLOCK)

    # Decode flat output index -> (batch row, w_out). XPU has no hardware
    # integer divider, so keep a single `//` and derive the remainder as
    # w_idx = o - b * W_out (exact for non-negative values, maxdiff=0).
    b = o // W_out
    w_idx = o - b * W_out

    # Reflected width index. pad_left < W_in is validated on the host, so a
    # single period (abs + where) is exact -- no `% (2*(W_in-1))` needed.
    x = w_idx.to(tl.int32) - pad_left
    pW = 2 * (W_in - 1)
    t = tl.abs(x)
    iw = tl.where(t < W_in, t, pW - t)

    in_offs = b * W_in + iw
    if NEED_MASK:
        # Tail block: mask off lanes o >= total_out. When BLOCK divides
        # total_out exactly the host passes NEED_MASK=False and the
        # unmasked path avoids the slow masked-memory path on XPU.
        mask = o < total_out
        vals = tl.load(in_ptr + in_offs, mask=mask)
        tl.store(out_ptr + o, vals, mask=mask)
    else:
        vals = tl.load(in_ptr + in_offs)
        tl.store(out_ptr + o, vals)


@triton.jit
def copy_tensor_kernel(in_ptr, out_ptr, total, BLOCK: tl.constexpr):
    # Flat contiguous copy (no padding path). Mask-based, contiguous offsets ->
    # block DMA, same as the padded kernel's store side.
    pid = tl.program_id(axis=0)
    o = pid * BLOCK + tl.arange(0, BLOCK)
    mask = o < total
    vals = tl.load(in_ptr + o, mask=mask)
    tl.store(out_ptr + o, vals, mask=mask)


# One side (left: SIDE=0 / right: SIDE=1) of the reflection pads, for ALL rows
# at once, as a flat (B * pad_side) range. The interior is copied by the native
# `_copy_from` engine (see _launch_reflection_pad1d big-shape path); only these
# 2 * B * pad_side elements use a Triton gather (reversed source order).
#
# IMPORTANT: every lane's load/store address is clamped in-bounds
# (`idx_c = min(idx, total_side-1)`). The earlier variant computed the address
# from the raw (unclamped) index: masked-off lanes produced out-of-bounds
# addresses and the XPU backend corrupted neighboring memory (documented
# "masked tail" hazard of this backend), showing up as garbage in the pad
# slots. Clamping keeps all addresses inside the buffer; the `m` mask makes the
# extra duplicate writes semantically harmless.
@triton.jit
def pad1d_side_kernel(
    in_ptr,
    out_ptr,
    W_in,
    pad_left,
    W_out,
    total_side,
    PAD: tl.constexpr,
    SIDE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    idx = pid * BLOCK + tl.arange(0, BLOCK)
    idx_c = tl.minimum(idx, total_side - 1)
    m = idx < total_side
    b = idx_c // PAD
    j = idx_c - b * PAD
    if SIDE == 0:
        # output w = j in [0, pad_left) <- src = pad_left - j (reversed)
        src = pad_left - j
        dst = b * W_out + j
    else:
        # output w = W_in + pad_left + j in [W_in+pad_left, W_out)
        # <- src = W_in - 2 - j (reversed; exact when pad_right < W_in, which
        # is host-validated)
        src = W_in - 2 - j
        dst = b * W_out + W_in + pad_left + j
    v = tl.load(in_ptr + b * W_in + src)
    tl.store(out_ptr + dst, v, mask=m)


def _launch_reflection_pad1d(input: torch.Tensor, padding, out: torch.Tensor = None):
    if not isinstance(padding, (list, tuple)) or len(padding) != 2:
        raise ValueError(
            "padding must be a sequence of length 2: (pad_left, pad_right)"
        )
    pad_left, pad_right = int(padding[0]), int(padding[1])
    if pad_left < 0 or pad_right < 0:
        raise ValueError("padding values must be >= 0")
    if input.dim() < 1:
        raise ValueError("input must have at least 1 dimension")

    x = input.contiguous()
    W_in = int(x.shape[-1])
    if W_in <= 0:
        raise ValueError("last dimension (width) must be > 0")

    W_out = W_in + pad_left + pad_right
    leading_shape = x.shape[:-1]
    B = int(math.prod(leading_shape)) if len(leading_shape) > 0 else 1

    if out is None:
        out = torch.empty((*leading_shape, W_out), device=x.device, dtype=x.dtype)
    else:
        expected_shape = (*leading_shape, W_out)
        if tuple(out.shape) != expected_shape:
            raise ValueError(
                f"out tensor has shape {tuple(out.shape)}, expected {expected_shape}"
            )
        if out.dtype != x.dtype:
            raise ValueError(
                f"out dtype {out.dtype} does not match input dtype {x.dtype}"
            )
        if out.device != x.device:
            raise ValueError("out must be on the same device as input")
        out = out.contiguous()

    # BLOCK=1024 is the best all-round tile on XPU (measured sweep in the
    # reflection_pad2d fix): small shapes avoid huge-block launch waste, and
    # medium/large shapes still get enough work per program.
    BLOCK = 1024

    # No padding: just copy
    if pad_left == 0 and pad_right == 0:
        total = B * W_in
        grid = (triton.cdiv(total, BLOCK),)
        with torch_device_fn.device(x.device):
            copy_tensor_kernel[grid](x, out, total, BLOCK=BLOCK)
        return out

    # Validate reflection padding constraints
    if W_in < 2:
        raise ValueError(
            "input width must be at least 2 for reflection padding when padding > 0"
        )
    if pad_left >= W_in or pad_right >= W_in:
        raise ValueError(
            "padding values must be less than the input width for reflection padding"
        )

    total_out = B * W_out
    # Big-output split path: copy the (contiguous) interior with the native
    # `_copy_from` engine (gems never overrides `_copy_from` -> reaches the
    # vendor strided-copy engine; same trick as slice_backward/constant_pad_nd)
    # and handle only the 2*B*(pad_left+pad_right) pad elements in Triton.
    # The flat kernel below is gather-bound for the interior (measured ~2.2ms
    # vs native ~0.023ms on [32,64,2048] pad(3,5)); the split is ~30x faster
    # there (measured ~60-75us). For smaller outputs the extra launches lose
    # to the single flat kernel, so keep them on the flat path (measured
    # crossover: split wins from ~33k output elements up; the benchmark
    # middle shape (8,16,256) stays on flat).
    if total_out >= 262144:
        with torch_device_fn.device(x.device):
            mid = torch.ops.aten.slice(out, -1, pad_left, pad_left + W_in)
            torch.ops.aten._copy_from(x, mid, False)
            if pad_left > 0:
                tot = B * pad_left
                pad1d_side_kernel[(triton.cdiv(tot, 256),)](
                    x,
                    out,
                    W_in,
                    pad_left,
                    W_out,
                    tot,
                    PAD=pad_left,
                    SIDE=0,
                    BLOCK=256,
                )
            if pad_right > 0:
                tot = B * pad_right
                pad1d_side_kernel[(triton.cdiv(tot, 256),)](
                    x,
                    out,
                    W_in,
                    pad_left,
                    W_out,
                    tot,
                    PAD=pad_right,
                    SIDE=1,
                    BLOCK=256,
                )
        return out

    # Adaptive BLOCK: tiny outputs fit with far less launch overhead in a
    # 256-lane program than in one 1024-lane program (measured 1.2-1.7x on
    # (3,33)/(2,4,64)); medium/large shapes keep BLOCK=1024 (sweep optimum).
    BLOCK = 256 if total_out <= 1024 else 1024
    need_mask = (total_out % BLOCK) != 0
    grid = (triton.cdiv(total_out, BLOCK),)
    with torch_device_fn.device(x.device):
        reflection_pad1d_kernel[grid](
            x,
            out,
            W_in,
            pad_left,
            W_out,
            total_out,
            BLOCK=BLOCK,
            NEED_MASK=need_mask,
        )
    return out


def reflection_pad1d(input: torch.Tensor, padding):
    logger.debug("GEMS_KUNLUNXIN REFLECTION_PAD1D")
    return _launch_reflection_pad1d(input, padding, out=None)


def reflection_pad1d_out(input: torch.Tensor, padding, out: torch.Tensor):
    logger.debug("GEMS_KUNLUNXIN REFLECTION_PAD1D_OUT")
    return _launch_reflection_pad1d(input, padding, out=out)
