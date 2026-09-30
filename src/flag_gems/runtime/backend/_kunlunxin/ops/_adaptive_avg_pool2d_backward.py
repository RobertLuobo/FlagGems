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
def _adaptive_avg_pool2d_backward_exact_kernel(
    grad_output_ptr,
    grad_input_ptr,
    in_h,
    in_w,
    out_h,
    out_w,
    KH: tl.constexpr,
    KW: tl.constexpr,
    AREA: tl.constexpr,
    n_elems: tl.constexpr,
    BLOCK: tl.constexpr,
):
    # Fast path: exact integer ratio (in % out == 0 on every dim).
    # Each input element belongs to exactly one output's pooling region, so
    # grad_input[i] = grad_output[i // K] / (KH*KW).  One load, one store.
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elems

    h = (offsets // in_w) % in_h
    w = offsets % in_w
    nc = offsets // (in_h * in_w)

    o_off = nc * (out_h * out_w) + (h // KH) * out_w + (w // KW)
    val = tl.load(grad_output_ptr + o_off, mask=mask)
    tl.store(grad_input_ptr + offsets, val / AREA, mask=mask)


@libentry()
@triton.jit
def _adaptive_avg_pool2d_backward_general_kernel(
    grad_output_ptr,
    grad_input_ptr,
    numel,
    in_h,
    in_w,
    out_h,
    out_w,
    MAX_H: tl.constexpr,
    MAX_W: tl.constexpr,
    BLOCK: tl.constexpr,
):
    # General path (non-integer ratios, handles upsampling).
    # Flat 1D grid over grad_input (same shape as the proven avg_pool2d
    # backward): one program handles BLOCK flat input elements.  For each
    # input element (h, w) we enumerate the (few) output positions whose
    # pooling region may contain it: o in [o_min, o_max) per dim, with
    # o_max - o_min <= ceil(out/in) + 1.  Both tap loops are statically bounded
    # (MAX_H, MAX_W constexpr) so the XPU unroller stays small.  Loads clamp
    # the candidate to [0, out-1] and the flat index to a valid lane, so the
    # address is always in-bounds; the contribution is zeroed with tl.where.
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < numel
    safe = tl.where(mask, offsets, 0)

    w = safe % in_w
    h = (safe // in_w) % in_h
    nc = safe // (in_h * in_w)

    h_min = (h * out_h) // in_h
    h_max = tl.minimum(((h + 1) * out_h + in_h - 1) // in_h, out_h)
    w_min = (w * out_w) // in_w
    w_max = tl.minimum(((w + 1) * out_w + in_w - 1) // in_w, out_w)

    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    gop = grad_output_ptr + nc * (out_h * out_w)
    for oh in tl.static_range(0, MAX_H):
        o_h = h_min + oh
        h_ok = o_h < h_max
        c_h = tl.minimum(o_h, out_h - 1)
        hs = (c_h * in_h) // out_h
        he = tl.minimum(((c_h + 1) * in_h + out_h - 1) // out_h, in_h)
        h_in_region = (h >= hs) & (h < he)
        for ow in tl.static_range(0, MAX_W):
            o_w = w_min + ow
            w_ok = o_w < w_max
            c_w = tl.minimum(o_w, out_w - 1)
            ws = (c_w * in_w) // out_w
            we = tl.minimum(((c_w + 1) * in_w + out_w - 1) // out_w, in_w)
            in_region = h_in_region & (w >= ws) & (w < we)
            active = mask & h_ok & w_ok
            area = (he - hs) * (we - ws)
            val = tl.load(gop + c_h * out_w + c_w, mask=mask, other=0.0)
            contrib = tl.where(in_region, val, 0.0) / tl.cast(area, tl.float32)
            acc += tl.where(active, contrib, 0.0)

    tl.store(
        grad_input_ptr + offsets,
        acc.to(grad_input_ptr.type.element_ty),
        mask=mask,
    )


def _adaptive_avg_pool2d_backward(grad_output, self):
    """Gradient of _adaptive_avg_pool2d (Kunlunxin/XPU implementation).

    The generic pointwise kernel builds 2D (BLOCK_H, BLOCK_W) tiles and a
    dynamic (out_h, out_w) nested loop; that lowering fails TritonXPULegalize
    on XPU3.  This override mirrors the proven 3D backward: a 1D exact fast
    path for integer ratios and a 1D general path (one program per input row,
    lanes over w) that keeps the innermost vector loop statically bounded.
    """
    logger.debug("GEMS_KUNLUNXIN _ADAPTIVE_AVG_POOL2D_BACKWARD")

    input_is_3d = self.dim() == 3
    if input_is_3d:
        grad_output = grad_output.unsqueeze(0)
        self = self.unsqueeze(0)

    grad_output = grad_output.contiguous()
    self = self.contiguous()
    in_n, in_c, in_h, in_w = self.shape
    out_h, out_w = grad_output.shape[-2:]

    grad_input = torch.empty_like(self)
    if grad_output.numel() == 0 or self.numel() == 0:
        if input_is_3d:
            return grad_input.squeeze(0)
        return grad_input

    with torch_device_fn.device(self.device):
        if in_h % out_h == 0 and in_w % out_w == 0:
            kh, kw = in_h // out_h, in_w // out_w
            n_elems = in_n * in_c * in_h * in_w
            grid = (triton.cdiv(n_elems, 1024),)
            _adaptive_avg_pool2d_backward_exact_kernel[grid](
                grad_output,
                grad_input,
                in_h,
                in_w,
                out_h,
                out_w,
                KH=kh,
                KW=kw,
                AREA=kh * kw,
                n_elems=n_elems,
                BLOCK=1024,
                num_warps=4,
            )
        else:
            n_elems = in_n * in_c * in_h * in_w
            grid = (triton.cdiv(n_elems, 1024),)
            _adaptive_avg_pool2d_backward_general_kernel[grid](
                grad_output,
                grad_input,
                n_elems,
                in_h,
                in_w,
                out_h,
                out_w,
                MAX_H=(out_h + in_h - 1) // in_h + 1,
                MAX_W=(out_w + in_w - 1) // in_w + 1,
                BLOCK=1024,
                num_warps=4,
            )

    if input_is_3d:
        return grad_input.squeeze(0)
    return grad_input
