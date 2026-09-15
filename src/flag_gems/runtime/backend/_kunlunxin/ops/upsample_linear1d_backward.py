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

"""KunlunXin (XPU) implementation of `upsample_linear1d_backward`.

Gather-style backward: every program owns a contiguous chunk of the *input*
gradient, so each output element is written exactly once (no `atomic_add`, no
ordering non-determinism).

Why this differs from the generic implementation:

* The generic kernel stores at ``base + (offs % in_w) * stride_w``. The runtime
  modulo in the store index defeats KunlunXin OffsetAnalysis and the store is
  lowered as a discrete scatter. Here `grad_in` is freshly allocated and
  contiguous, so the store index is the raw flat ``pid * BLOCK + arange``
  (provably stride-1 -> contiguous block DMA).
* The generic kernel uses ``BLOCK = 512``; on a 512M-element benchmark case
  that is ~1e6 programs, and each program repeats the per-iteration float
  divisions. Here the block is large and the affine coefficients are computed
  once on the host, so the inner loop is add/mul only.
* The scan window is the exact analytic window (see `_window` below) instead of
  a fixed ``WINDOW = 2`` / ``cdiv(out_w, in_w) + 2``.
* `grad_in` is `torch.empty` (every element is written unconditionally) instead
  of `torch.zeros`, which removes a full-size memset launch.
"""

import logging
import math

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn

logger = logging.getLogger(__name__)


@triton.jit
def upsample_linear1d_backward_kernel(
    grad_out_ptr,
    grad_in_ptr,
    total,
    in_w,
    out_w,
    c_scale,
    c_in_off,
    c_out_off,
    r_scale,
    r_in_off,
    r_out_off,
    in_w_m1_f,
    WL: tl.constexpr,
    WR: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    o = pid * BLOCK + tl.arange(0, BLOCK)
    mask = o < total

    # The tail lanes must not produce an out-of-range *load* address. On
    # KunlunXin a masked `tl.load` still issues the access for masked-off lanes
    # (its `other` value is not applied), so an unclamped `nc = o // in_w` can
    # address far past the end of `grad_output` and hang the device (observed:
    # n=2, c=3, in_w=33, out_w=16 wedged the card until SIGKILL). Clamping `o`
    # only for the gather index keeps the store index the raw stride-1 `o`.
    o_c = tl.minimum(o, total - 1)
    nc = o_c // in_w
    x_in = o_c - nc * in_w

    x_in_f = x_in.to(tl.float32)
    # center(x_in): continuous output coordinate that maps back to x_in.
    center = (x_in_f + c_in_off) * c_scale + c_out_off
    base = tl.floor(center).to(tl.int32)
    base_f = base.to(tl.float32)
    # x_real(x_out) = (x_out + r_in_off) * r_scale + r_out_off is affine in x_out,
    # so the per-iteration value is just xr_base + i * r_scale (no division).
    xr_base = (base_f + r_in_off) * r_scale + r_out_off

    go_base = nc * out_w
    acc = tl.zeros([BLOCK], dtype=tl.float32)

    for i in tl.static_range(-WL, WR + 1):
        x_out = base + i
        valid = (x_out >= 0) & (x_out < out_w)
        # The address is clamped in range so out-of-window lanes never touch
        # memory outside `grad_output`; `valid` is applied to the *weight*
        # instead of relying on the masked-load `other` value (the generic
        # kernel relies on `other=0.0`, which is why an out-of-range x_out with
        # r_scale == 0 could accumulate garbage on XPU).
        x_out_c = tl.minimum(tl.maximum(x_out, 0), out_w - 1)
        x_real = xr_base + i * r_scale
        x_real = tl.maximum(x_real, 0.0)
        x0_f = tl.floor(x_real)
        w1 = x_real - x0_f
        w0 = 1.0 - w1
        x0_i = x0_f.to(tl.int32)
        x1_i = tl.minimum(x0_f + 1.0, in_w_m1_f).to(tl.int32)

        g = tl.load(grad_out_ptr + go_base + x_out_c, mask=mask & valid, other=0.0).to(
            tl.float32
        )

        same = x0_i == x1_i
        weight = tl.where(x_in == x0_i, tl.where(same, w0 + w1, w0), 0.0)
        weight += tl.where((~same) & (x_in == x1_i), w1, 0.0)
        acc += tl.where(valid, g * weight, 0.0)

    tl.store(grad_in_ptr + o, acc.to(grad_in_ptr.dtype.element_ty), mask=mask)


@triton.jit
def upsample_linear1d_backward_iw1_kernel(
    grad_out_ptr,
    grad_in_ptr,
    nc,
    out_w,
    BLOCK: tl.constexpr,
):
    # in_w == 1 is degenerate: every output position maps to input position 0
    # (align_corners makes the scale 0/0), so grad_in[nc, 0] is a plain row sum.
    # Handled by a dedicated reduction kernel instead of widening the scan
    # window of the main kernel to `out_w` (which would explode the unrolled IR).
    row = tl.program_id(0)
    base = row * out_w
    acc = tl.zeros([BLOCK], dtype=tl.float32)
    for start in range(0, out_w, BLOCK):
        offs = start + tl.arange(0, BLOCK)
        keep = offs < out_w
        # Same masked-load caveat as the main kernel: clamp the address and
        # zero the contribution explicitly instead of trusting `other`.
        offs_c = tl.minimum(offs, out_w - 1)
        g = tl.load(grad_out_ptr + base + offs_c, mask=keep, other=0.0).to(tl.float32)
        acc += tl.where(keep, g, 0.0)
    row_sum = tl.sum(acc, axis=0)
    tl.store(grad_in_ptr + row, row_sum.to(grad_in_ptr.dtype.element_ty))


def _window(c_scale):
    # Contributors to input position x are the x_out with
    # x_real(x_out) in [x-1, x+1), i.e. x_out in [center - s, center + s)
    # with s = c_scale. Since base = floor(center), i = x_out - base satisfies
    # frac(center) - s <= i < 1 + s, hence |i| <= ceil(s).
    return max(1, int(math.ceil(c_scale - 1e-6)))


def upsample_linear1d_backward(
    grad_output: torch.Tensor,
    output_size,
    input_size,
    align_corners: bool,
    scale_factors=None,
) -> torch.Tensor:
    logger.debug("GEMS_KUNLUNXIN UPSAMPLE_LINEAR1D_BACKWARD")

    if len(input_size) == 3:
        n, c, in_w = input_size
    elif len(input_size) == 2:
        n, c, in_w = input_size[0], 1, input_size[1]
    elif len(input_size) == 1:
        n, c, in_w = 1, 1, input_size[0]
    else:
        raise ValueError

    if output_size is not None:
        out_w = output_size[0]
    else:
        assert scale_factors is not None
        out_w = int(in_w * scale_factors[0])

    assert grad_output.shape[-1] == out_w

    grad_out_flat = grad_output.contiguous().view(n * c, out_w)
    grad_in = torch.empty(
        (n, c, in_w), device=grad_output.device, dtype=grad_output.dtype
    )

    if in_w == 1:
        with torch_device_fn.device(grad_output.device):
            upsample_linear1d_backward_iw1_kernel[(n * c,)](
                grad_out_flat, grad_in, n * c, out_w, BLOCK=1024
            )
        return grad_in

    if align_corners:
        c_scale = (out_w - 1.0) / (in_w - 1.0) if in_w > 1 else 0.0
        c_in_off, c_out_off = 0.0, 0.0
        r_scale = (in_w - 1.0) / (out_w - 1.0) if out_w > 1 else 0.0
        r_in_off, r_out_off = 0.0, 0.0
    else:
        c_scale = out_w / in_w
        c_in_off, c_out_off = 0.5, -0.5
        r_scale = in_w / out_w
        r_in_off, r_out_off = 0.5, -0.5

    w = _window(c_scale)
    total = n * c * in_w
    BLOCK = 2048
    grid = (triton.cdiv(total, BLOCK),)

    with torch_device_fn.device(grad_output.device):
        upsample_linear1d_backward_kernel[grid](
            grad_out_flat,
            grad_in,
            total,
            in_w,
            out_w,
            c_scale,
            c_in_off,
            c_out_off,
            r_scale,
            r_in_off,
            r_out_off,
            float(in_w - 1),
            WL=w,
            WR=w,
            BLOCK=BLOCK,
        )

    return grad_in
