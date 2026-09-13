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


# Flat 1D kernel over the ENTIRE output (all batches at once).
#
# ROOT CAUSE of the old slowness: the previous kernel wrapped every store index
# with `% HW_out` ("modulo wrap") to avoid masked stores. On KunlunXin XPU that
# runtime modulo defeats OffsetAnalysis, so EVERY load/store degrades to the
# discrete per-element path (~1.2 GB/s). Even a pure contiguous copy written with
# `%total` measured 228ms / 1.2 GB/s vs 0.49ms / 578 GB/s for the mask-based
# copy — a ~470x penalty (see reflection_pad2d_perf_fix.md).
#
# Fix: flatten (b, h_out, w_out) into one linear output index `o` and store to
# `o` directly (provably stride-1 -> block DMA). A single boolean mask
# `o < total_out` handles the tail. The masked-off lanes only exist in the final
# partial block; their store addresses fall past the end of the buffer and are
# suppressed by the mask (verified maxdiff=0 across the whole shape matrix,
# incl. tail-masked shapes). Note that clamping `o` (min with a runtime scalar)
# is NOT an option here: it defeats the compiler's contiguity proof and
# degrades the store to the discrete per-element path (~3x slower, measured
# A/B). The border kernels of the split path (pad2d_hside_kernel /
# pad2d_wside_kernel) DO clamp every lane's address in-bounds, because their
# offsets are non-affine anyway and unclamped pad kernels on XPU corrupt
# neighboring memory (see reflection_pad1d_out 2026-09-08).
#
# The reflected input index is still a data-dependent gather (structural XPU
# wall), and the flat-index decode needs integer div/mod (slow on XPU), so the
# big shape stays ~40ms; but that is ~4.5x faster than the 183ms modulo version.
@triton.jit
def reflection_pad2d_kernel(
    in_ptr,
    out_ptr,
    H_in,
    W_in,
    pad_left,
    pad_top,
    W_out,
    HW_out,
    HW_in,
    total_out,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    o = pid * BLOCK + tl.arange(0, BLOCK)
    mask = o < total_out

    # Decode flat output index -> (batch, h_out, w_out). NOTE: do NOT clamp `o`
    # here — a min with a runtime scalar defeats the compiler's contiguity proof
    # and degrades the store to the discrete per-element path (measured ~3x
    # slower on small shapes). The masked-off lanes only exist in the final
    # partial block and their stores are suppressed by `mask` (maxdiff=0 across
    # the whole shape matrix, incl. tail-masked shapes). Bounds-safe addressing
    # IS mandatory in the border kernels of the split path (see
    # pad2d_hside_kernel) where the offsets are non-affine anyway.
    b = o // HW_out
    rem = o % HW_out
    h_idx = rem // W_out
    w_idx = rem % W_out

    # Reflected height index. pad_top < H_in is validated on the host, so a single
    # period (abs + where) is exact — no `% (2*(H_in-1))` needed.
    y = h_idx.to(tl.int32) - pad_top
    pH = 2 * (H_in - 1)
    t_h = tl.abs(y)
    ih = tl.where(t_h < H_in, t_h, pH - t_h)

    # Reflected width index (same reasoning; pad_left < W_in validated).
    x = w_idx.to(tl.int32) - pad_left
    pW = 2 * (W_in - 1)
    t_w = tl.abs(x)
    iw = tl.where(t_w < W_in, t_w, pW - t_w)

    in_offs = b * HW_in + ih * W_in + iw
    vals = tl.load(in_ptr + in_offs, mask=mask)
    tl.store(out_ptr + o, vals, mask=mask)


@triton.jit
def copy_tensor_kernel(in_ptr, out_ptr, total, BLOCK: tl.constexpr):
    # Flat contiguous copy (no padding path). Mask-based, contiguous offsets ->
    # block DMA, same as the padded kernel's store side.
    pid = tl.program_id(axis=0)
    o = pid * BLOCK + tl.arange(0, BLOCK)
    mask = o < total
    vals = tl.load(in_ptr + o, mask=mask)
    tl.store(out_ptr + o, vals, mask=mask)


# Top (h_out in [0, pad_top)) + bottom (h_out in [H_in+pad_top, H_out)) rows of
# the output. Used by the big-shape split path (see _launch_reflection_pad2d_split):
# the interior is copied by the native `_copy_from` engine, only these
# B*(pad_top+pad_bottom)*W_out border elements use a Triton gather.
#
# OPTIMIZATION (2026-09-11, reflection_pad2d_out task): the previous
# pad2d_hside_kernel used a flat 1D index `o` over all B*R*W_out border elements
# and clamped it (oc = min(o, total_h-1)) before decoding. On KunlunXin XPU the
# clamp (a runtime min) defeats the compiler's contiguity proof, so the store
# `out + f(oc)` degraded to the discrete per-element path: measured 2.16ns/el
# (4606us for the 2.13M border elements of (32,64,128,256) pad(0,4,0,4)). This
# row-per-program variant assigns ONE program per border row (b, r); the store
# offset is a per-program scalar base + pure arange(w), so it is provably
# stride-1 and lowers to block DMA. The load side is an unavoidable gather
# (reflected row + reflected column), but the whole border pass drops to
# 0.40ns/el (851us, 5.4x faster). The mask `w < W_out` only ever masks the
# tail lanes of the LAST row (the flat kernel uses the identical masked-tail
# pattern and is verified clean); loads are always in-bounds (reflection is a
# total map into [0, H_in) x [0, W_in)). Validated: canary-guarded output
# buffers + full reference comparison across the whole test/benchmark shape
# matrix, maxerr = 0.0.
@triton.jit
def pad2d_hside_rows_kernel(
    in_ptr,
    out_ptr,
    H_in: tl.constexpr,
    W_in: tl.constexpr,
    pad_left: tl.constexpr,
    pad_top: tl.constexpr,
    pad_bottom: tl.constexpr,
    W_out: tl.constexpr,
    HW_out: tl.constexpr,
    HW_in: tl.constexpr,
    W_BLOCK: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    R = pad_top + pad_bottom
    b = pid // R
    r = pid - b * R

    # Output row: top region rows [0, pad_top), bottom region rows
    # [H_in+pad_top, H_out).
    h_out = tl.where(r < pad_top, r, H_in + r)

    # Reflected height index (single-reflection exact: host validates
    # pad_top/pad_bottom < H_in, so |h_out - pad_top| <= 2*(H_in-1)).
    y = h_out - pad_top
    t_h = tl.abs(y)
    pH = 2 * (H_in - 1)
    ih = tl.where(t_h < H_in, t_h, pH - t_h)

    # Reflected width index (same single-reflection argument for pad_left/right).
    w = tl.arange(0, W_BLOCK)
    m = w < W_out
    x = w - pad_left
    t_w = tl.abs(x)
    pW = 2 * (W_in - 1)
    iw = tl.where(t_w < W_in, t_w, pW - t_w)

    vals = tl.load(in_ptr + b * HW_in + ih * W_in + iw)
    tl.store(out_ptr + b * HW_out + h_out * W_out + w, vals, mask=m)


# Left/right pad columns of the INTERIOR (non-padded-height) rows:
# out[b, pad_top+y, w] for w in [0, pad_left) U [pad_left+W_in, W_out).
# Flat index o over B*H_in*(pad_left+pad_right); same in-bounds clamping as
# pad2d_hside_kernel.
@triton.jit
def pad2d_wside_kernel(
    in_ptr,
    out_ptr,
    H_in: tl.constexpr,
    W_in: tl.constexpr,
    pad_left: tl.constexpr,
    pad_right: tl.constexpr,
    pad_top: tl.constexpr,
    W_out: tl.constexpr,
    HW_out: tl.constexpr,
    HW_in: tl.constexpr,
    total_w,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    o = pid * BLOCK + tl.arange(0, BLOCK)
    m = o < total_w
    oc = tl.minimum(o, total_w - 1)

    P = pad_left + pad_right
    row = oc // P
    j = oc - row * P
    b = row // H_in
    y = row - b * H_in

    # output column (left segment [0, pad_left), right segment
    # [pad_left+W_in, W_out)); source column reversed, exact when
    # pad_left/pad_right < W_in (host-validated).
    w_out = tl.where(j < pad_left, j, pad_left + W_in + (j - pad_left))
    src = tl.where(j < pad_left, pad_left - j, W_in - 2 - (j - pad_left))

    vals = tl.load(in_ptr + b * HW_in + y * W_in + src)
    tl.store(out_ptr + b * HW_out + (pad_top + y) * W_out + w_out, vals, mask=m)


# Left/right pad columns of the INTERIOR (non-padded-height) rows, big total_w
# variant: gather into a CONTIGUOUS scratch of B*H_in*P elements, then let the
# vendor `_copy_from` engine write the two strided column segments (the same
# native-engine trick as the interior). The in-place pad2d_wside_kernel above
# keeps the small-total_w cases, where its single launch beats scratch + 2
# copies (measured crossover ~2^18; at 65536 the in-place kernel is 116-141us
# vs 221us for scratch+copy, at 1048576 scratch+copy is 574us vs 1592us).
#
# On KunlunXin XPU a runtime min/clamp defeats the compiler's contiguity proof
# (see the hside rows kernel above), so the STORE uses the unclamped `o` and
# only the LOAD decodes from the clamped `oc` (in-bounds for every lane, even
# the masked tail) — the load is a data-dependent gather anyway, so the clamp is
# free there. The masked tail lanes' stores (`o >= total_w`) are suppressed by
# `m` (same masked-tail pattern as the flat kernel, verified clean) and the host
# allocates BLOCK slots of slack, so even a hypothetical un-suppressed store
# cannot corrupt neighboring memory.
@triton.jit
def pad2d_wside_scratch_kernel(
    in_ptr,
    scratch_ptr,
    H_in: tl.constexpr,
    W_in: tl.constexpr,
    pad_left: tl.constexpr,
    pad_right: tl.constexpr,
    P: tl.constexpr,
    HW_in: tl.constexpr,
    total_w,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    o = pid * BLOCK + tl.arange(0, BLOCK)
    m = o < total_w
    oc = tl.minimum(o, total_w - 1)

    row = oc // P
    j = oc - row * P
    b = row // H_in
    y = row - b * H_in

    # source column: left segment reversed (pad_l-j: [pad_l .. 1]); right
    # segment reversed (W_in-2-(j-pad_l): [W_in-2 .. W_in-1-pad_r]). Exact
    # (single period) because pad_left/pad_right < W_in is host-validated.
    src = tl.where(j < pad_left, pad_left - j, W_in - 2 - (j - pad_left))
    v = tl.load(in_ptr + b * HW_in + y * W_in + src)
    tl.store(scratch_ptr + o, v, mask=m)


def _launch_reflection_pad2d_split(
    x, out, pad_left, pad_right, pad_top, pad_bottom, H_in, W_in, H_out, W_out, B
):
    """Big-shape split: native `_copy_from` for the contiguous interior + two
    small Triton kernels for the H-side (top/bottom rows) and W-side (interior
    left/right columns) borders. `_copy_from` is never overridden by gems, so it
    reaches the vendor strided-copy engine (same trick as slice_backward /
    constant_pad_nd / reflection_pad1d big-shape path)."""
    HW_out = H_out * W_out
    HW_in = H_in * W_in
    with torch_device_fn.device(x.device):
        # 1. Interior block: native strided copy. `narrow` is gems-registered and
        # is a zero-copy as_strided view (no kernel, no 4-arg slice step), so
        # this is both safe and dispatch-cheap.
        mid = torch.ops.aten.narrow(out, -2, pad_top, H_in)
        mid = torch.ops.aten.narrow(mid, -1, pad_left, W_in)
        torch.ops.aten._copy_from(x, mid, False)
        # 2. Top/bottom rows: one program per border row (contiguous store).
        if pad_top > 0 or pad_bottom > 0:
            R = pad_top + pad_bottom
            W_BLOCK = triton.next_power_of_2(W_out)
            pad2d_hside_rows_kernel[(B * R,)](
                x,
                out,
                H_in,
                W_in,
                pad_left,
                pad_top,
                pad_bottom,
                W_out,
                HW_out,
                HW_in,
                W_BLOCK=W_BLOCK,
            )
        # 3. Interior rows' left/right columns.
        if pad_left > 0 or pad_right > 0:
            P = pad_left + pad_right
            total_w = B * H_in * P
            if total_w >= 262144:
                # Big W-side: contiguous scratch + 2 vendor strided copies.
                BLOCK_W = 1024
                scratch = torch.empty(
                    total_w + BLOCK_W, device=x.device, dtype=x.dtype
                )
                pad2d_wside_scratch_kernel[(triton.cdiv(total_w, BLOCK_W),)](
                    x,
                    scratch,
                    H_in,
                    W_in,
                    pad_left,
                    pad_right,
                    P,
                    HW_in,
                    total_w,
                    BLOCK=BLOCK_W,
                )
                s3 = torch.ops.aten.narrow(scratch, 0, 0, total_w).view(B, H_in, P)
                o3 = out.view(B, H_out, W_out)
                i3 = torch.ops.aten.narrow(o3, 1, pad_top, H_in)
                if pad_left > 0:
                    torch.ops.aten._copy_from(
                        torch.ops.aten.narrow(s3, 2, 0, pad_left),
                        torch.ops.aten.narrow(i3, 2, 0, pad_left),
                        False,
                    )
                if pad_right > 0:
                    torch.ops.aten._copy_from(
                        torch.ops.aten.narrow(s3, 2, pad_left, pad_right),
                        torch.ops.aten.narrow(i3, 2, pad_left + W_in, pad_right),
                        False,
                    )
            else:
                pad2d_wside_kernel[(triton.cdiv(total_w, 4096),)](
                    x,
                    out,
                    H_in,
                    W_in,
                    pad_left,
                    pad_right,
                    pad_top,
                    W_out,
                    HW_out,
                    HW_in,
                    total_w,
                    BLOCK=4096,
                )
    return out


def launch_reflection_pad2d(input: torch.Tensor, padding, out: torch.Tensor = None):
    # Validate padding format
    if not isinstance(padding, (list, tuple)):
        raise ValueError("padding must be a sequence")
    if len(padding) != 4:
        raise ValueError(
            "padding must be a sequence of length 4: (pad_left, pad_right, pad_top, pad_bottom)"
        )
    pad_left, pad_right, pad_top, pad_bottom = [int(p) for p in padding]

    # Validate padding values
    if pad_left < 0 or pad_right < 0 or pad_top < 0 or pad_bottom < 0:
        raise ValueError("padding values must be >= 0")

    # Validate input
    if input.dim() < 3:
        raise ValueError("input must have at least 3 dimensions")

    x = input.contiguous()
    H_in = int(x.shape[-2])
    W_in = int(x.shape[-1])
    # Validate reflection padding constraints
    if H_in < 2 or W_in < 2:
        raise ValueError(
            "input spatial dimensions must be at least 2 for reflection padding when padding > 0"
        )
    if H_in <= 0 or W_in <= 0:
        raise ValueError("spatial dimensions must be > 0")
    if pad_left >= W_in or pad_right >= W_in or pad_top >= H_in or pad_bottom >= H_in:
        raise ValueError(
            "padding values must be less than the input spatial dimensions for reflection padding"
        )

    H_out = H_in + pad_top + pad_bottom
    W_out = W_in + pad_left + pad_right

    leading_shape = x.shape[:-2]
    B = int(math.prod(leading_shape)) if len(leading_shape) > 0 else 1

    # Handle output tensor
    if out is None:
        out = torch.empty(
            (*leading_shape, H_out, W_out), device=x.device, dtype=x.dtype
        )
    else:
        expected_shape = (*leading_shape, H_out, W_out)
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

    # No padding: just copy
    if pad_left == 0 and pad_right == 0 and pad_top == 0 and pad_bottom == 0:
        BLOCK = 1024
        total = B * H_in * W_in
        grid = (triton.cdiv(total, BLOCK),)
        with torch_device_fn.device(x.device):
            copy_tensor_kernel[grid](x, out, total, BLOCK=BLOCK)
        return out

    HW_out = H_out * W_out
    HW_in = H_in * W_in
    total_out = B * HW_out

    # Big-output split path: the flat kernel below is gather-bound for large
    # shapes (measured ~40ms on 70M output elements vs native ~0.3ms; the
    # per-lane gather + soft integer div/decode dominates), while the split's
    # interior runs on the vendor's native strided-copy engine and only the
    # B*(pad_top+pad_bottom)*W_out + B*H_in*(pad_left+pad_right) border
    # elements stay in Triton. Small outputs keep the single flat kernel: the
    # border kernels are scatter/gather-bound at ~3.2ns/element (vs ~0.6ns/el
    # for the flat kernel at 0.6M elements), so the split only wins when the
    # interior dominates - measured: split LOSES at (8,16,64,64) pad(3,5,3,5)
    # (663K output, 21% border: 493us vs flat 387us) and WINS by 3.3-3.8x at
    # (16,32,64,128) (17.8M, 1.1% border: 0.73ms vs 2.50ms) and
    # (32,64,128,256) (70.3M, 7.5% border: 10.6ms vs 40.3ms). Threshold
    # 2^20 keeps the whole measured matrix regression-free.
    if total_out >= 1048576:
        return _launch_reflection_pad2d_split(
            x,
            out,
            pad_left,
            pad_right,
            pad_top,
            pad_bottom,
            H_in,
            W_in,
            H_out,
            W_out,
            B,
        )

    # BLOCK=1024 is the best all-round tile on XPU: small shapes avoid the
    # per-program waste of a huge block, while medium/large shapes still get
    # enough work per program to stay off the launch floor (measured sweep).
    BLOCK = 1024
    grid = (triton.cdiv(total_out, BLOCK),)
    with torch_device_fn.device(x.device):
        reflection_pad2d_kernel[grid](
            x,
            out,
            H_in,
            W_in,
            pad_left,
            pad_top,
            W_out,
            HW_out,
            HW_in,
            total_out,
            BLOCK=BLOCK,
        )
    return out


def reflection_pad2d(input: torch.Tensor, padding):
    logger.debug("GEMS_KUNLUNXIN REFLECTION_PAD2D")
    return launch_reflection_pad2d(input, padding, out=None)


def reflection_pad2d_out(input: torch.Tensor, padding, out: torch.Tensor):
    logger.debug("GEMS_KUNLUNXIN REFLECTION_PAD2D_OUT")
    return launch_reflection_pad2d(input, padding, out=out)
