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


# Kunlunxin (XPU) override of replication_pad1d / replication_pad1d.out.
#
# Performance reconstruction (2026-08-21, device XPU 5, benchmark matrix
# 4 shapes x 3 float dtypes):
# - The generic 1D flat Triton kernel decodes every output index with int64
#   div/mod and does a per-lane clamp gather (discrete access); on XPU it
#   measures 0.05x on the large benchmark shape (8,32,256) (~150us vs ~7us
#   torch) and ~0.4x on the mid shapes - the same int64-decode + gather
#   penalty as replication_pad2d.
# - Measured alternatives on the same matrix (do_bench, fresh cache):
#     flat int32 clamp kernel (BLOCK 256/512 by total):  best below ~100K
#        output elems (2,3,7: 6.5us vs generic 7.6us; 4,16,64: 8.9us vs
#        18.4-19.4us; 8,32,256: 42us vs 137-151us);
#     vendor `_copy_from` 3-segment path (1 interior + 2 narrow edge stripes):
#       best above ~100K (16,64,256: 55us vs flat ~154us); the two edge
#       segments stay narrow (pad width), so the vendor engine serves them
#       cheaply.
# - Negative padding (crop semantics) and total_out >= 2^31 fall back to the
#   flat clamp kernel (int32 / int64 variants); the native XPU engine asserts
#   pad >= 0 on crops, so crop cases are only reachable through the clamp
#   kernels.
#
# Performance reconstruction round 2 (2026-09-04, device XPU 3, official
# triton.testing.do_bench, benchmark matrix + large shapes):
# - The flat clamp kernel's `o // W_out` + per-lane clamp gather still costs
#   42-66us at (8,32,256) (fp16/bf16 ~2.5x the fp32 time), while a plain
#   contiguous flat copy of the same size takes 8.8us and the vendor
#   `_copy_from` interior block 6.3us (torch reference 6.65us).
# - The old 3-segment `_copy_from` path was dominated by the two edge
#   segments' `expand` (stride-0) source views (~80us each on XPU): the edge
#   columns are now written by one flat Triton kernel
#   (`_replication_pad1d_edge_kernel`, total_nc*(pad_l+pad_r) lanes, all
#   contiguous 1..pad-wide runs). New fast path =
#     interior `_copy_from` (vendor engine, ~6.3us) + edge kernel (~2us work +
#     one ~6us launch): (8,32,256) 42.1-65.0 -> 15.2-15.6us,
#     (16,64,256) 57.5 -> 20.9us, (32,64,256) 62.3 -> 29.8us,
#     (2,1000,100) 50.6 -> 35.2us.
# - Crossover (fp32): <=10K elems flat clamp (256/512/1024 buckets) wins,
#   >10K the new pair wins. fp16/bf16 at (32,256) 8320 elems marginally
#   prefer the pair (13.9 vs 15-16.3us) but flat is <1us cheaper at
#   (4,8,256) 8288, so a single 10K threshold is used.


@triton.jit
def _replication_pad1d_kernel_clamp_i64(
    x_ptr,
    out_ptr,
    W_in,
    W_out,
    pad_l,
    total_out,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    o = pid * BLOCK + tl.arange(0, BLOCK)
    o64 = o.to(tl.int64)
    mask = o < total_out

    # Decode flat output index -> (nc, w_out).
    nc = o64 // W_out
    w_out = o64 % W_out

    # Replication clamp handles both pad directions: forward padding clamps to
    # the edge, negative padding (crop) shifts the source window.
    iw = w_out - pad_l
    iw = tl.where(iw < 0, 0, iw)
    iw = tl.where(iw > W_in - 1, W_in - 1, iw)

    in_offs = (nc * W_in + iw).to(tl.int64)
    vals = tl.load(x_ptr + in_offs, mask=mask)
    tl.store(out_ptr + o64, vals, mask=mask)


@triton.jit
def _replication_pad1d_kernel_clamp_i32(
    x_ptr,
    out_ptr,
    W_in,
    W_out,
    pad_l,
    total_out,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    o = pid * BLOCK + tl.arange(0, BLOCK)
    mask = o < total_out

    # int32 arithmetic is safe because the host only takes this path when
    # total_out < 2^31.
    nc = o // W_out
    w_out = o % W_out

    iw = w_out - pad_l
    iw = tl.where(iw < 0, 0, iw)
    iw = tl.where(iw > W_in - 1, W_in - 1, iw)

    in_offs = nc * W_in + iw
    vals = tl.load(x_ptr + in_offs, mask=mask)
    tl.store(out_ptr + o, vals, mask=mask)


@triton.jit
def _replication_pad1d_edge_kernel(
    x_ptr,
    out_ptr,
    W_in,
    W_out,
    pad_l,
    pad_r,
    total_nc,
    BLOCK: tl.constexpr,
):
    # One flat pass over all edge (replicated) columns of every row.
    # Edge index n = nc * (pad_l + pad_r) + e; e < pad_l is the left-edge
    # element (source column 0), e >= pad_l the right-edge element (source
    # column W_in - 1). Both the interior block copy and this edge kernel
    # therefore use only contiguous source/destination runs (the interior
    # runs of a row are contiguous; the edge runs are 1..pad columns wide),
    # unlike the flat clamp kernel whose per-lane `o // W_out` decode +
    # clamp makes every load a discrete gather on XPU (measured ~1.5-2 GB/s
    # big-shape penalty, ~2.5-3x on fp16/bf16).
    n = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    per = pad_l + pad_r
    total_e = total_nc * per
    mask = n < total_e
    nc = n // per
    e = n - nc * per
    is_l = e < pad_l
    dst = nc * W_out + tl.where(is_l, e, pad_l + W_in + (e - pad_l))
    v = tl.load(x_ptr + nc * W_in + tl.where(is_l, 0, W_in - 1), mask=mask)
    tl.store(out_ptr + dst, v, mask=mask)


def _pad2(padding):
    if isinstance(padding, torch.Tensor):
        padding = tuple(int(p) for p in padding.tolist())
    if isinstance(padding, int):
        return (padding, padding)
    if not isinstance(padding, (tuple, list)) or len(padding) != 2:
        raise ValueError(
            "padding must be a sequence of 2 integers: (pad_left, pad_right)"
        )
    return tuple(int(p) for p in padding)


def _launch_flat_clamp(x, out3, W_in, W_out, pad_l, total_out):
    # int64 index arithmetic when the flat index would overflow int32.
    if total_out >= 2**31:
        BLOCK = 1024
        grid = (triton.cdiv(total_out, BLOCK),)
        _replication_pad1d_kernel_clamp_i64[grid](
            x,
            out3,
            W_in,
            W_out,
            pad_l,
            total_out,
            BLOCK=BLOCK,
        )
    else:
        # BLOCK sweep (2026-08-21, XPU 5, do_bench): tiny totals want 256
        # lanes, mid totals want 512; both beat 1024 on the benchmark matrix.
        # Re-swept 2026-09-04 (XPU 3, official do_bench): the 8K-17K range
        # prefers 1024 (8288: 512=12.0us vs 1024=12.0us tie; 8320: 512=11.5
        # vs 1024=11.9us), while 4352 still prefers 512 (8.4 vs 11.7us) and
        # <=2048 prefers 256. Three buckets, all measured.
        if total_out <= 2048:
            BLOCK = 256
        elif total_out <= 6000:
            BLOCK = 512
        else:
            BLOCK = 1024
        grid = (triton.cdiv(total_out, BLOCK),)
        _replication_pad1d_kernel_clamp_i32[grid](
            x,
            out3,
            W_in,
            W_out,
            pad_l,
            total_out,
            BLOCK=BLOCK,
        )


# Measured crossover (2026-09-04, XPU 3, official do_bench): below ~10K
# output elements the flat i32 clamp kernel wins (1 launch, no vendor-copy
# dispatch); above it the interior `_copy_from` + edge-kernel pair wins
# (8288 elems tie at ~12us, 16768 elems 14.1us vs flat 15.5-18.7us, 66304
# elems 15.2us vs flat 42.1-65.0us, 265216 elems 20.9us vs old 3-seg 55us).
FLAT_LIMIT = 10_000

# Edge kernel tile: the edge workload total_nc*(pad_l+pad_r) is usually tiny
# (a few thousand lanes), one 1024-lane program is enough; huge edges (very
# many rows) still split into cdiv() programs.
EDGE_BLOCK = 1024


def launch_replication_pad1d(input: torch.Tensor, padding, out: torch.Tensor = None):
    pad_l, pad_r = _pad2(padding)

    dim = input.dim()
    if dim not in (2, 3):
        raise ValueError("replication_pad1d expects 2D (C, W) or 3D (N, C, W) input")

    x = input.contiguous()
    is_2d = dim == 2
    if is_2d:
        x = x.unsqueeze(0)

    N, C, W_in = x.shape
    W_out = W_in + pad_l + pad_r

    # Match the reference: N may be 0 (empty batch), but C and W must be
    # positive.
    if C <= 0:
        raise RuntimeError(
            "Expected 2D or 3D (batch mode) tensor with possibly 0 batch size "
            "and other non-zero dimensions for input"
        )
    if W_in <= 0:
        raise ValueError("Input width must be greater than 0 for replication padding")
    if W_out <= 0:
        raise RuntimeError(
            f"replication_pad1d: output spatial dimension is non-positive: "
            f"output size {W_out}"
        )

    if out is None:
        out3 = torch.empty((N, C, W_out), device=x.device, dtype=x.dtype)
    else:
        expected = (C, W_out) if is_2d else (N, C, W_out)
        if tuple(out.shape) != expected:
            raise ValueError(
                f"Provided out tensor has shape {tuple(out.shape)}, expected {expected}"
            )
        if out.device != x.device:
            raise ValueError("Input and out must be on the same device")
        if out.dtype != x.dtype:
            raise ValueError("Input and out must have the same dtype")
        out3 = out.unsqueeze(0) if is_2d else out

    total_out = N * C * W_out
    if total_out == 0:
        return out3.squeeze(0) if is_2d else out3

    has_neg_pad = pad_l < 0 or pad_r < 0

    # Dispatch: flat clamp kernel under FLAT_LIMIT output elements and for
    # any negative padding (crop semantics); vendor `_copy_from` interior
    # block + flat Triton edge kernel above that (measured 2026-09-04, XPU 3,
    # official do_bench: (8,32,256) 66304 elems 42.1-65.0us -> 15.2-15.6us,
    # (16,64,256) 57.5 -> 20.9us, (32,64,256) 62.3 -> 29.8us; the old
    # 3-segment `_copy_from` path paid ~80us per edge `expand` (stride-0)
    # copy, replaced by the edge kernel).
    if has_neg_pad or total_out <= FLAT_LIMIT:
        kout = out3 if out3.is_contiguous() else torch.empty_like(out3)
        with torch_device_fn.device(x.device):
            _launch_flat_clamp(x, kout, W_in, W_out, pad_l, total_out)
        if kout is not out3:
            with torch_device_fn.device(x.device):
                torch.ops.aten._copy_from(kout, out3)
        return out3.squeeze(0) if is_2d else out3

    # Fast path: vendor strided-copy of the whole interior block + one flat
    # Triton kernel for the replicated edge columns. The interior copy is a
    # contiguously-strided 2D block (each row W_in contiguous), served by the
    # vendor engine at ~40 GB/s; the edge kernel covers total_nc*(pad_l+pad_r)
    # lanes with contiguous 1..pad-wide destination runs. Non-contiguous out
    # writes to a contiguous temp first, then `_copy_from` back (same as the
    # flat path).
    with torch_device_fn.device(x.device):
        dst3 = out3 if out3.is_contiguous() else torch.empty_like(out3)
        torch.ops.aten._copy_from(x, dst3[:, :, pad_l : pad_l + W_in])
        if pad_l or pad_r:
            total_e = N * C * (pad_l + pad_r)
            grid_e = (triton.cdiv(total_e, EDGE_BLOCK),)
            _replication_pad1d_edge_kernel[grid_e](
                x,
                dst3,
                W_in,
                W_out,
                pad_l,
                pad_r,
                N * C,
                BLOCK=EDGE_BLOCK,
            )
        if dst3 is not out3:
            torch.ops.aten._copy_from(dst3, out3)

    return out3.squeeze(0) if is_2d else out3


def replication_pad1d(input: torch.Tensor, padding):
    logger.debug("GEMS_KUNLUNXIN REPLICATION_PAD1D")
    return launch_replication_pad1d(input, padding, out=None)


def replication_pad1d_out(input: torch.Tensor, padding, out: torch.Tensor):
    logger.debug("GEMS_KUNLUNXIN REPLICATION_PAD1D_OUT")
    return launch_replication_pad1d(input, padding, out=out)
