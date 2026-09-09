"""Backward of reflection_pad2d WITHOUT atomics (Kunlunxin XPU).

Performance reconstruction (2026-09-08):
- Flat-HW input-parallel decomposition, grid=(cdiv(H*W,BLOCK), N*C): no atomics,
  each input element accumulates the reflected grad_output contributions in fp32.
  (Replaces the previous narrow/flip/add_/clone torch-op composition which was
  ~20-60x slower than native: measured baseline 3.8-36ms vs torch 0.01-1.4ms.)
- All shape dims are tl.constexpr (compile-time magic-number division) and every
  index/address vector stays int32, mirroring the proven reflection_pad3d_backward
  kernel.
- Reflected h/w coordinates are CLAMPED with tl.where to a valid center value, so
  every load is unconditional and in-bounds; per-contribution validity is
  reapplied with value-level tl.where(mask, v, 0.0). This also guards against the
  XPU backend's known masked-load `other` quirk (masked lanes may still perform
  OOB accesses -- the select zeroes the value afterwards).
- NOTE (XPU hazard, same as pad2d_hside_kernel in reflection_pad2d): the flat
  index is clamped (oc = min(offs, H*W-1)) BEFORE decoding, so every lane decodes
  a valid (h, w) and both load and store addresses stay in-bounds by
  construction; masked-off lanes re-read/re-write the last valid element
  (idempotent), preventing the XPU masked-tail corruption of neighboring memory.
- fp32 accumulation, explicit destination-dtype store.

Semantics identical to the previous implementation (84P/9S CPU-ref test).
"""

import logging

import torch
import triton
import triton.language as tl

from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as tle

logger = logging.getLogger(__name__)


@triton.jit
def _load_grad(grad_ptr, base, off, mask):
    value = tl.load(grad_ptr + base + off).to(tl.float32)
    return tl.where(mask, value, 0.0)


@libentry()
@triton.jit
def _reflection_pad2d_backward_kernel(
    grad_ptr,
    out_ptr,
    N,
    C,
    pad_left,
    pad_right,
    pad_top,
    pad_bottom,
    OUT_DTYPE: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    HO: tl.constexpr,
    WO: tl.constexpr,
    BLOCK_HW: tl.constexpr,
):
    pid = tle.program_id(0).to(tl.int32)
    bc = tle.program_id(1).to(tl.int32)
    offs = pid * BLOCK_HW + tl.arange(0, BLOCK_HW)
    mask = offs < H * W
    n = bc % N
    c = bc // N

    # Clamp BEFORE decoding: every lane's (h, w) stays in-bounds so both load and
    # store addresses are in-bounds by construction (idempotent re-access of the
    # last valid element for masked-off lanes; see reflection_pad2d pad2d_hside).
    oc = tl.minimum(offs, H * W - 1)
    h = oc // W
    w = oc - h * W

    hlm = (h > 0) & (h <= pad_top)
    hrm = (h >= H - 1 - pad_bottom) & (h < H - 1)
    wlm = (w > 0) & (w <= pad_left)
    wrm = (w >= W - 1 - pad_right) & (w < W - 1)

    hw_out = HO * WO
    gb = (n * C + c) * hw_out
    ob = (n * C + c) * (H * W)

    # ---- clamped reflected coordinates (keeps every load in-bounds) ----
    h0 = pad_top + h
    hl = tl.where(hlm, pad_top - h, h0)
    hr = tl.where(hrm, pad_top + 2 * H - 2 - h, h0)
    w0 = pad_left + w
    wl = tl.where(wlm, pad_left - w, w0)
    wr = tl.where(wrm, pad_left + 2 * W - 2 - w, w0)

    # ---- 3 shared row bases (h-mode) x 3 shared columns (w-mode) ----
    r0 = gb + h0 * WO
    rl = gb + hl * WO
    rr = gb + hr * WO

    acc = tl.zeros((BLOCK_HW,), dtype=tl.float32)
    acc += _load_grad(grad_ptr, r0, w0, mask)
    acc += _load_grad(grad_ptr, r0, wl, mask & wlm)
    acc += _load_grad(grad_ptr, r0, wr, mask & wrm)
    acc += _load_grad(grad_ptr, rl, w0, mask & hlm)
    acc += _load_grad(grad_ptr, rl, wl, mask & hlm & wlm)
    acc += _load_grad(grad_ptr, rl, wr, mask & hlm & wrm)
    acc += _load_grad(grad_ptr, rr, w0, mask & hrm)
    acc += _load_grad(grad_ptr, rr, wl, mask & hrm & wlm)
    acc += _load_grad(grad_ptr, rr, wr, mask & hrm & wrm)

    dst = ob + h * W + w
    tl.store(out_ptr + dst, acc.to(OUT_DTYPE), mask=mask)


def reflection_pad2d_backward(grad_output, self, padding):
    logger.debug("GEMS_KUNLUNXIN REFLECTION_PAD2D_BACKWARD")

    if len(padding) != 4:
        raise ValueError("padding must be a sequence of 4 elements")
    pad_left, pad_right, pad_top, pad_bottom = (int(pad) for pad in padding)

    if self.dim() not in (3, 4):
        raise ValueError("input must be a 3D or 4D tensor")

    input_height, input_width = self.shape[-2:]
    output_height = input_height + pad_top + pad_bottom
    output_width = input_width + pad_left + pad_right
    if tuple(grad_output.shape[-2:]) != (output_height, output_width):
        raise ValueError(
            "grad_output spatial shape "
            f"{tuple(grad_output.shape[-2:])}, expected {(output_height, output_width)}"
        )

    if not any((pad_left, pad_right, pad_top, pad_bottom)):
        return grad_output.clone()

    g = grad_output.contiguous()
    if g.dim() == 3:
        g = g.unsqueeze(0)
        squeeze_out = True
    else:
        squeeze_out = False

    N, C, HO, WO = g.shape
    H, W = input_height, input_width

    if self.dtype == torch.float16:
        out_dtype = tl.float16
    elif self.dtype == torch.bfloat16:
        out_dtype = tl.bfloat16
    else:
        out_dtype = tl.float32

    out = torch.empty((N, C, H, W), device=self.device, dtype=self.dtype)
    if out.numel() == 0:
        return out.squeeze(0) if squeeze_out else out

    BLOCK_HW = 256 if H * W <= 4096 else 512
    grid = (triton.cdiv(H * W, BLOCK_HW), N * C)
    _reflection_pad2d_backward_kernel[grid](
        g,
        out,
        N,
        C,
        pad_left,
        pad_right,
        pad_top,
        pad_bottom,
        OUT_DTYPE=out_dtype,
        H=H,
        W=W,
        HO=HO,
        WO=WO,
        BLOCK_HW=BLOCK_HW,
    )
    return out.squeeze(0) if squeeze_out else out