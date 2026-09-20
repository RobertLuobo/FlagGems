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
"""Kunlunxin (XPU) override for ``aten::_upsample_nearest_exact2d`` (+ ``.out``).

Two things are fixed relative to the generic implementation in
``flag_gems/ops/_upsample_nearest_exact2d.py``:

1. **Semantics.**  ``nearest-exact`` reads
   ``src = floor((dst + 0.5) * (in / out))`` (ATen
   ``nearest_neighbor_exact_compute_source_index``), *not* the plain
   ``nearest`` formula ``floor(dst * in / out)``.  The two agree for integer
   up-sampling factors -- which is all the official matrix covers -- and
   diverge for every other ratio, so the generic kernel silently returned a
   *different image* for e.g. ``(1,1,3,5) -> (5,8)`` or ``(1,1,5,5) -> (3,3)``.
2. **Performance.**  The generic kernel iterates ``while nc_iter < NC`` inside
   the program (~0.85 us per loop iteration on this backend, one iteration per
   plane) and expresses the output store as ``base + oh * sH_out + ow * sW_out``
   -- an address the compiler cannot prove stride-1, i.e. the ~85x slower
   per-lane store path.  This kernel keeps the output store affine
   (``ptr_o + idx``) and constexpr-izes the decode dimensions so the div/mod
   becomes a shift/mask.

The one expensive primitive that is left -- the gather load, which lowers to
``llvm.xpu.gm2lm_v3`` plus one ``llvm.xpu.mfence`` per 64 lanes (257 of each
per 16384-lane program in the generated LLIR) -- is spread over the ~12-program
layout the backend itself prefers (``cluster_num`` = 12 in the compiled
metadata).  That gather is clamped to the last plane for the lanes past
``total`` in every launch configuration (see the kernel).
"""

import logging
import math
from typing import Optional, Tuple

import numpy as np
import torch
import triton
import triton.language as tl

from flag_gems.runtime import device, torch_device_fn

logger = logging.getLogger(__name__)
device = device.name

PROGRAMS = 12
MAX_BLOCK = 16384
MIN_BLOCK = 512
NUM_WARPS = 4

# Staging copy (only reached for non-contiguous inputs): aim for ~12 programs
# like the main kernel, but keep the tile at or below 4096 -- 32768 is a
# registered hard-fail on this backend.
STAGE_PROGRAMS = 12
STAGE_MAX_BLOCK = 4096
STAGE_MIN_BLOCK = 512
STAGE_NUM_WARPS = 4


@triton.jit
def _upsample_nearest_exact2d_kernel(
    ptr_o,
    ptr_i,
    total,
    NC: tl.constexpr,
    C: tl.constexpr,
    IH: tl.constexpr,
    IW: tl.constexpr,
    OH: tl.constexpr,
    OW: tl.constexpr,
    rheight,
    rwidth,
    soN,
    soC,
    soH,
    soW,
    OUT_STRIDED: tl.constexpr,
    BLOCK: tl.constexpr,
    NEED_MASK: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    idx = pid * BLOCK + tl.arange(0, BLOCK)

    sp = idx % (OH * OW)
    nc = idx // (OH * OW)
    ow = sp % OW
    oh = sp // OW

    # ATen: (int)floor((dst + 0.5) * scale), scale = in/out (1/scale_factor when
    # the caller passed scales).  The argument is non-negative, so the float->int
    # conversion below is a truncation, i.e. the same floor.
    ih = tl.minimum(((oh + 0.5) * rheight).to(tl.int32), IH - 1)
    iw = tl.minimum(((ow + 0.5) * rwidth).to(tl.int32), IW - 1)

    # A tail lane (idx >= total) decodes nc >= NC.  The gather below has to be
    # clamped in *every* launch configuration: the maskless main entry stores
    # its tail lanes into the over-allocated buffer (so the store side is safe
    # there), but without the clamp those lanes gather
    # `(nc * IH + ih) * IW + iw` past the end of the input -- e.g. the 84 tail
    # lanes of the official (3,7,1023,1025) case read up to 41 elements past
    # the input (offset 22,020,116 vs 22,020,075 in-bounds elements, i.e.
    # entirely mis-indexed data).  The clamp cannot change a valid lane.
    # oh/ow come from a modulo, so they are always valid.
    nc = tl.minimum(nc, NC - 1)
    data = tl.load(ptr_i + (nc * IH + ih) * IW + iw)
    if OUT_STRIDED:
        n = nc // C
        c = nc - n * C
        o_off = n * soN + c * soC + oh * soH + ow * soW
    else:
        o_off = idx
    if NEED_MASK:
        tl.store(ptr_o + o_off, data, mask=idx < total)
    else:
        tl.store(ptr_o + o_off, data)


def _f32(value):
    """Round a Python float to the nearest float32 (bit-exact fp32 constant)."""
    return float(np.float32(value))


def _reciprocal_scale(in_size, out_size, scale):
    """Match ATen ``compute_scales_value<float>`` exactly."""
    if scale is not None and scale > 0:
        # static_cast<float>(1.0 / scale)
        return _f32(1.0 / scale)
    # static_cast<float>(in) / out  (a genuine fp32 division)
    return float(np.float32(in_size) / np.float32(out_size))


def _pick_block(total):
    return min(
        MAX_BLOCK, max(MIN_BLOCK, triton.next_power_of_2(triton.cdiv(total, PROGRAMS)))
    )


def _shape_args(input, output_size, scales_h, scales_w):
    N, C, IH, IW = input.shape
    if output_size is not None:
        OH, OW = int(output_size[-2]), int(output_size[-1])
    else:
        scale_h = scales_h if scales_h is not None else 1.0
        scale_w = scales_w if scales_w is not None else 1.0
        OH, OW = int(math.floor(IH * scale_h)), int(math.floor(IW * scale_w))
    if OH < 0 or OW < 0:
        raise ValueError("Output size must be non-negative.")
    if OH * OW > 0 and (IH == 0 or IW == 0):
        # Same rejection as ATen; the generic implementation silently loads out
        # of bounds here.
        raise RuntimeError(
            "Input and output sizes should be greater than 0, but got input "
            f"(H: {IH}, W: {IW}) output (H: {OH}, W: {OW})"
        )
    return N, C, IH, IW, OH, OW


@triton.jit
def _stage_contiguous_kernel(
    ptr_dst,
    ptr_src,
    N: tl.constexpr,
    C: tl.constexpr,
    IH: tl.constexpr,
    IW: tl.constexpr,
    sN,
    sC,
    sH,
    sW,
    BLOCK: tl.constexpr,
):
    """Copy an arbitrarily strided ``(N, C, IH, IW)`` source into contiguous dst.

    ``dst`` is contiguous, so ``ptr_dst + flat`` is provably stride-1 and the
    store lowers to a block DMA.  The source is addressed through its *real*
    strides (``n * sN + c * sC + ih * sH + iw * sW``), which also covers zero
    (broadcast) and negative strides.  The launch covers
    ``cdiv(numel, BLOCK) * BLOCK`` lanes and ``dst`` is over-allocated to that
    same multiple, so the main body needs no mask; the tail lanes decode
    ``n >= N`` and are clamped to the last plane, which keeps the gather (and
    the store) in bounds -- masks are not honoured on this backend, so a
    guarded store cannot be relied on (see HARNESS_SUMMARY 8.4).
    """
    pid = tl.program_id(axis=0)
    flat = pid * BLOCK + tl.arange(0, BLOCK)
    iw = flat % IW
    ih = (flat // IW) % IH
    c = (flat // (IH * IW)) % C
    n = tl.minimum(flat // (C * IH * IW), N - 1)
    off = n * sN + c * sC + ih * sH + iw * sW
    tl.store(ptr_dst + flat, tl.load(ptr_src + off))


def _stage_contiguous(input):
    """Return ``input`` itself, or a contiguous copy of it.

    Contiguous inputs are returned unchanged (no launch, no allocation).  A
    strided input is copied with this file's own ``_stage_contiguous_kernel``:
    the destination index is flat and stride-1, so the store is a block DMA,
    and the source is addressed with its real strides.  The other candidate
    primitives are all unsafe or forbidden here -- ``Tensor.contiguous`` / the
    vendor ``copy_`` wedge the card on strided 2-byte sources (``copy_slice``),
    and ``aten::_copy_from`` is a native fallback that a gems operator must not
    use (user red line, 2026-09-20).  ``empty`` (not ``empty_like``) so the
    staging buffer is genuinely contiguous.
    """
    N, C, IH, IW = input.shape
    if input.stride() == (C * IH * IW, IH * IW, IW, 1):
        return input
    total = N * C * IH * IW
    sN, sC, sH, sW = input.stride()
    # Same shape of choice as _pick_block: ~12 programs for the big copies, but
    # a small tile for small tensors (a 4096-lane tile on a 768-element tensor
    # costs 29.7us instead of 8.4us, measured).
    block = min(
        STAGE_MAX_BLOCK,
        max(
            STAGE_MIN_BLOCK,
            triton.next_power_of_2(triton.cdiv(total, STAGE_PROGRAMS)),
        ),
    )
    grid = (triton.cdiv(total, block),)
    # Over-allocate to the tile multiple so the store body stays maskless; the
    # returned view carries the requested shape/strides.
    buf = torch.empty(grid[0] * block, device=input.device, dtype=input.dtype)
    staged = buf[:total].view(N, C, IH, IW)
    with torch_device_fn.device(input.device):
        _stage_contiguous_kernel[grid](
            staged,
            input,
            N=N,
            C=C,
            IH=IH,
            IW=IW,
            sN=sN,
            sC=sC,
            sH=sH,
            sW=sW,
            BLOCK=block,
            num_warps=STAGE_NUM_WARPS,
        )
    return staged


def _launch(input, out, total, rheight, rwidth, out_strided, need_mask):
    N, C, IH, IW = input.shape
    OH, OW = out.shape[-2:]
    block = _pick_block(total)
    grid = (triton.cdiv(total, block),)
    soN, soC, soH, soW = out.stride()
    with torch_device_fn.device(input.device):
        _upsample_nearest_exact2d_kernel[grid](
            out,
            input,
            total,
            NC=N * C,
            C=C,
            IH=IH,
            IW=IW,
            OH=OH,
            OW=OW,
            rheight=rheight,
            rwidth=rwidth,
            soN=soN,
            soC=soC,
            soH=soH,
            soW=soW,
            OUT_STRIDED=out_strided,
            BLOCK=block,
            NEED_MASK=need_mask,
            num_warps=NUM_WARPS,
        )
    return out


def _upsample_nearest_exact2d(
    input: torch.Tensor,
    output_size: Optional[Tuple[int, int]] = None,
    scales_h: Optional[float] = None,
    scales_w: Optional[float] = None,
) -> torch.Tensor:
    logger.debug("GEMS_KUNLUNXIN _UPSAMPLE_NEAREST_EXACT2D")
    assert input.device.type == device
    if input.ndim != 4:
        raise ValueError(
            "_upsample_nearest_exact2d expects a 4D tensor (N, C, H, W); "
            f"got shape {tuple(input.shape)}"
        )
    N, C, IH, IW, OH, OW = _shape_args(input, output_size, scales_h, scales_w)
    total = N * C * OH * OW
    if total == 0:
        return torch.empty((N, C, OH, OW), device=input.device, dtype=input.dtype)
    rheight = _reciprocal_scale(IH, OH, scales_h)
    rwidth = _reciprocal_scale(IW, OW, scales_w)
    input = _stage_contiguous(input)

    # Over-allocate to the tile multiple so the bulk store stays maskless, then
    # hand back a view with the requested shape (strides/contiguity unchanged).
    block = _pick_block(total)
    buf = torch.empty(
        triton.cdiv(total, block) * block, device=input.device, dtype=input.dtype
    )
    output = buf[:total].view(N, C, OH, OW)
    return _launch(input, output, total, rheight, rwidth, False, False)


def _upsample_nearest_exact2d_out(
    input: torch.Tensor,
    output_size: Optional[Tuple[int, int]] = None,
    scales_h: Optional[float] = None,
    scales_w: Optional[float] = None,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    logger.debug("GEMS_KUNLUNXIN _UPSAMPLE_NEAREST_EXACT2D_OUT")
    assert input.device.type == device
    if input.ndim != 4:
        raise ValueError(
            "_upsample_nearest_exact2d expects a 4D tensor (N, C, H, W); "
            f"got shape {tuple(input.shape)}"
        )
    N, C, IH, IW, OH, OW = _shape_args(input, output_size, scales_h, scales_w)
    if tuple(out.shape) != (N, C, OH, OW):
        raise ValueError(
            "Provided out tensor has shape "
            f"{tuple(out.shape)} but expected {(N, C, OH, OW)}."
        )
    if out.dtype != input.dtype:
        raise ValueError(
            f"Provided out tensor has dtype {out.dtype} but expected {input.dtype}."
        )
    total = out.numel()
    if total == 0:
        return out
    rheight = _reciprocal_scale(IH, OH, scales_h)
    rwidth = _reciprocal_scale(IW, OW, scales_w)
    input = _stage_contiguous(input)
    out_strided = not out.is_contiguous()
    # The caller's buffer cannot be over-allocated, so a partial tail tile has
    # to fall back to a guarded store.
    need_mask = out_strided or total % _pick_block(total) != 0
    return _launch(input, out, total, rheight, rwidth, out_strided, need_mask)


_aten_out_lib = None


def _register_upsample_nearest_exact2d_out():
    """Bind the CUDA device key of ``aten::_upsample_nearest_exact2d.out``.

    ``_FULL_CONFIG`` (``src/flag_gems/__init__.py``) carries only the default
    overload, so ``torch.ops.aten._upsample_nearest_exact2d.out`` kept running
    the native torch_xmlir kernel while this module was dead code for it: the
    ``*_out`` cases of ``tests/test_upsample_nearest_exact2d.py`` passed without
    ever reaching Gems.  Registering the device key here is the same shape as
    the ``hypot.out`` / ``adaptive_max_pool2d/3d`` / ``range.step``
    registrations and keeps the ``_FULL_CONFIG`` red line intact.  The
    registration is process global; ``use_gems`` never registers this overload,
    so there is no conflict.
    """
    global _aten_out_lib
    if _aten_out_lib is not None:
        return
    _aten_out_lib = torch.library.Library("aten", "IMPL")
    _aten_out_lib.impl(
        "_upsample_nearest_exact2d.out",
        _upsample_nearest_exact2d_out,
        "CUDA",
        allow_override=True,
    )


_register_upsample_nearest_exact2d_out()


__all__ = ["_upsample_nearest_exact2d", "_upsample_nearest_exact2d_out"]
