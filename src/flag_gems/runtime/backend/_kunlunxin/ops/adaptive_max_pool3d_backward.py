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
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)


@libentry()
@triton.jit
def _adaptive_max_pool3d_backward_scatter_kernel(
    grad_output_ptr,
    indices_ptr,
    grad_input_ptr,
    n_out,
    out_per_nc,
    in_spatial,
    BLOCK: tl.constexpr,
):
    """Scatter-based adaptive max pool 3d backward (Kunlunxin/XPU).

    One lane per output position; valid only when ``in % out == 0`` on all
    three spatial dims (each input position belongs to exactly one adaptive
    window, hence the argmax positions of the disjoint windows are distinct
    input positions and the plain non-atomic stores can never race).  The
    Gems forward (see ``_patch_adaptive_max_pool3d_aten``) produces ATen-exact
    per-plane flat indices``d * in_hw + h * in_w + w``, so the upstream
    gradient is stored directly at ``nc * in_spatial + idx``.
    """
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_out
    idx = tl.load(indices_ptr + offsets).to(tl.int32)
    val = tl.load(grad_output_ptr + offsets).to(tl.float32)
    nc = offsets // out_per_nc
    tl.store(
        grad_input_ptr + nc * in_spatial + idx,
        val.to(grad_input_ptr.dtype.element_ty),
        mask=mask,
    )


@libentry()
@triton.jit
def _adaptive_max_pool3d_backward_gather_kernel(
    grad_output_ptr,
    indices_ptr,  # Gems-forward int64 indices, layout (n, c, out_d, out_h, out_w)
    grad_input_ptr,
    n_elems,  # in_n * in_c * in_d * in_h * in_w
    in_d,
    in_h,
    in_w,
    out_d,
    out_h,
    out_w,
    MAX_D: tl.constexpr,
    MAX_H: tl.constexpr,
    MAX_W: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Gather-based adaptive max pool 3d backward (Kunlunxin/XPU).

    One lane per input position.  The output positions whose adaptive window
    may contain this input element form the small box
    ``[d_min, d_max) x [h_min, h_max) x [w_min, w_max)`` with
    ``d_min = floor(d * out / in)``, ``d_max = ceil((d + 1) * out / in)``.
    For each candidate we load the argmax index produced by the Gems forward
    and the upstream gradient, and accumulate the gradient whose index equals
    this position.

    This is the same exact, deterministic, race-free pattern as the
    Kunlunxin ``max_pool3d_backward``: ``tl.atomic_add`` scatter loses updates
    on this backend (~1e-5 per op, seed-dependent), so every output element's
    contribution is accumulated in registers and written with a single
    masked store.  The candidate box is bounded by the constexpr
    ``MAX_* = (out + in - 1) // in + 1``-style bound (1 when in is a multiple
    of out, ``out // in + 2`` otherwise), and candidate addresses are clamped
    so the (unmasked) loads can never go out of the (n, c) plane.
    """
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elems
    safe_offsets = tl.where(mask, offsets, 0)

    in_hw = in_h * in_w
    in_spatial = in_d * in_hw
    nc = safe_offsets // in_spatial
    rem = safe_offsets % in_spatial
    d = rem // in_hw
    rem2 = rem % in_hw
    h = rem2 // in_w
    w = rem2 % in_w

    my_flat = d * in_hw + h * in_w + w

    d_min = (d * out_d) // in_d
    d_max = tl.minimum(((d + 1) * out_d + in_d - 1) // in_d, out_d)
    h_min = (h * out_h) // in_h
    h_max = tl.minimum(((h + 1) * out_h + in_h - 1) // in_h, out_h)
    w_min = (w * out_w) // in_w
    w_max = tl.minimum(((w + 1) * out_w + in_w - 1) // in_w, out_w)

    out_per_nc = out_d * out_h * out_w
    gop = grad_output_ptr + nc * out_per_nc
    iop = indices_ptr + nc * out_per_nc

    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    # Small static bounds (MAX_* <= 2 whenever out <= in, the only legal
    # adaptive-pool configuration): fully unrolled bodies let the compiler
    # issue all candidate loads up front (ILP), matching the proven
    # Kunlunxin ``max_pool3d_backward_flat_kernel`` pattern.
    for od in tl.static_range(0, MAX_D):
        o_d = d_min + od
        d_ok = o_d < d_max
        c_d = tl.minimum(o_d, out_d - 1)
        for oh in tl.static_range(0, MAX_H):
            o_h = h_min + oh
            h_ok = o_h < h_max
            c_h = tl.minimum(o_h, out_h - 1)
            for ow in tl.static_range(0, MAX_W):
                o_w = w_min + ow
                w_ok = o_w < w_max
                active = mask & d_ok & h_ok & w_ok
                c_w = tl.minimum(o_w, out_w - 1)
                o_off = c_d * (out_h * out_w) + c_h * out_w + c_w
                idx = tl.load(iop + o_off).to(tl.int32)
                val = tl.load(gop + o_off).to(tl.float32)
                acc += tl.where(active & (idx == my_flat), val, 0.0)

    tl.store(
        grad_input_ptr + offsets,
        acc.to(grad_input_ptr.dtype.element_ty),
        mask=mask,
    )


def adaptive_max_pool3d_backward(
    grad_output: torch.Tensor,
    self: torch.Tensor,
    indices: torch.Tensor,
):
    """Gradient of adaptive_max_pool3d (Kunlunxin/XPU implementation).

    Built on the ATen-exact indices produced by the Gems ``adaptive_max_pool3d``
    forward (see ``_patch_adaptive_max_pool3d_aten`` and
    ``_patch_adaptive_max_pool3d_functional``; on this XPU stack the vendor
    XDNN wrapper rejects bfloat16 and returns uninitialized index memory for
    float16/float32, so a backward consuming XDNN indices would either fault
    or disagree with ATen).  Two exact, deterministic, atomics-free paths:

    * exact division (``in % out == 0`` on all three spatial dims): one lane
      per output position, non-atomic scatter (the argmax positions of
      disjoint adaptive windows are distinct input positions, so stores never
      race) -- O(n_out);
    * otherwise: one lane per input position, the gradient is accumulated in
      registers from the candidate output box and written with a single
      masked store -- O(n_in).
    """
    logger.debug("GEMS_KUNLUNXIN ADAPTIVE_MAX_POOL3D_BACKWARD")

    grad_output = grad_output.contiguous()
    self = self.contiguous()
    indices = indices.contiguous()
    in_n, in_c, in_d, in_h, in_w = self.shape
    out_d, out_h, out_w = grad_output.shape[-3:]

    # ATen semantics: grad_input is zero everywhere except at the argmax
    # positions of each output (unwritten positions must be 0, never garbage).
    grad_input = torch.zeros_like(self)

    n_in = grad_input.numel()
    if n_in == 0 or grad_output.numel() == 0:
        return grad_input

    # Exact division on all three dims: each input position belongs to exactly
    # one adaptive window, so the scatter fast path below is race-free.
    exact = (
        (in_d % out_d == 0) and (in_h % out_h == 0) and (in_w % out_w == 0)
    )

    with torch_device_fn.device(self.device):
        if exact:
            # Fast path: exact division -> each output's argmax is a distinct
            # input position, one lane per output, no atomics, no races.
            n_out = grad_output.numel()
            _adaptive_max_pool3d_backward_scatter_kernel[
                (triton.cdiv(n_out, 256),)
            ](
                grad_output,
                indices,
                grad_input,
                n_out,
                out_d * out_h * out_w,
                in_d * in_h * in_w,
                BLOCK=256,
                num_warps=4,
                buffer_size_limit=2048,
                isCloseVectorization=True,
            )
        else:
            # Exact upper bounds for the per-dim candidate counts (see gather
            # kernel comment): at most 1 when in is a multiple of out, at most
            # floor(out / in) + 2 otherwise.
            max_d = 1 if in_d % out_d == 0 else (out_d // in_d + 2)
            max_h = 1 if in_h % out_h == 0 else (out_h // in_h + 2)
            max_w = 1 if in_w % out_w == 0 else (out_w // in_w + 2)
            # 128 lanes / num_warps=2: the candidate box is at most 2 elements
            # per dim and fully statically unrolled; larger tiles overrun
            # uni_sram on this backend, do not raise without re-measuring.
            _adaptive_max_pool3d_backward_gather_kernel[(triton.cdiv(n_in, 128),)](
                grad_output,
                indices,
                grad_input,
                n_in,
                in_d,
                in_h,
                in_w,
                out_d,
                out_h,
                out_w,
                MAX_D=max_d,
                MAX_H=max_h,
                MAX_W=max_w,
                BLOCK=128,
                num_warps=2,
                buffer_size_limit=2048,
                isCloseVectorization=True,
            )

    return grad_input