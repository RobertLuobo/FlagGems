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
from flag_gems.utils import triton_lang_extension as ext

logger = logging.getLogger(__name__)


@libentry()
@triton.jit
def upsample_linear1d_backward_window_kernel(
    grad_output,
    grad_input,
    total,
    in_w: tl.constexpr,
    out_w: tl.constexpr,
    ALIGN_CORNERS: tl.constexpr,
    WINDOW: tl.constexpr,
    BLOCK: tl.constexpr,
):
    # Unique-writer windowed kernel: one program handles BLOCK input
    # positions; each input only scans the ~2*out_w/in_w outputs that can
    # contribute to it around its mapped center, i.e. O(in_w * WINDOW) work
    # instead of the O(in_w * out_w) dense scan.
    #
    # XPU constraints honored here (see harness/HARNESS_SUMMARY.md):
    # - no tl.sum / no tl.where inside the accumulation; contribution masks
    #   are applied by float multiply on pre-computed masks;
    # - masked loads with other=0.0 are not reliable on the XPU backend
    #   (the o-window guard would be compiled away), so every load address
    #   is clamped into [0, n*c*out_w) and the window guard is enforced
    #   purely by float masking of the weight.
    pid = ext.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < total

    x_in = offs % in_w
    row = offs // in_w
    row_c = tl.minimum(row, total // in_w - 1)

    x_in_f = x_in.to(tl.float32)
    if ALIGN_CORNERS:
        if in_w > 1:
            center = x_in_f * (out_w - 1) / (in_w - 1)
        else:
            center = tl.zeros([BLOCK], dtype=tl.float32)
    else:
        center = (x_in_f + 0.5) * out_w / in_w - 0.5
    base = tl.floor(center).to(tl.int32)

    acc = tl.zeros([BLOCK], dtype=tl.float32)
    go = grad_output + row_c * out_w

    for i in tl.static_range(-WINDOW, WINDOW + 1):
        o = base + i
        if ALIGN_CORNERS:
            if out_w > 1:
                s = o.to(tl.float32) * (in_w - 1) / (out_w - 1)
            else:
                s = tl.zeros([BLOCK], dtype=tl.float32)
        else:
            s = (o.to(tl.float32) + 0.5) * in_w / out_w - 0.5
        # ATen clamps the source index to >= 0 for linear (non-cubic)
        real = tl.maximum(s, 0.0)
        x0 = tl.floor(real).to(tl.int32)
        x1 = tl.minimum(x0 + 1, in_w - 1)
        w1 = real - x0.to(tl.float32)
        w0 = 1.0 - w1
        o_c = tl.minimum(tl.maximum(o, 0), out_w - 1)
        g = tl.load(go + o_c).to(tl.float32)
        v = (o >= 0).to(tl.float32) * (o < out_w).to(tl.float32)
        m0 = (x0 == x_in).to(tl.float32)
        m1 = (x1 == x_in).to(tl.float32)
        # split the two weight terms to mirror the ATen accumulation
        # (grad[x0] += w0*g ; grad[x1] += w1*g) exactly
        acc += g * v * (w0 * m0) + g * v * (w1 * m1)

    tl.store(grad_input + offs, acc, mask=mask)


def upsample_linear1d_backward(
    grad_output,
    output_size,
    input_size,
    align_corners,
    scale_factors=None,
):
    logger.debug("GEMS_KUNLUNXIN UPSAMPLE_LINEAR1D_BACKWARD")
    if len(input_size) == 3:
        n, c, in_w = input_size
    elif len(input_size) == 2:
        n, c, in_w = input_size[0], 1, input_size[1]
    elif len(input_size) == 1:
        n, c, in_w = 1, 1, input_size[0]
    else:
        raise ValueError("input_size must have one to three dimensions")

    if output_size is not None:
        out_w = output_size[0]
    else:
        assert scale_factors is not None
        out_w = int(in_w * scale_factors[0])
    assert grad_output.shape[-1] == out_w

    grad_output = grad_output.contiguous().view(n, c, out_w)
    grad_input = torch.empty(
        (n, c, in_w), dtype=grad_output.dtype, device=grad_output.device
    )
    total = n * c * in_w
    if total == 0:
        return grad_input.view(input_size)
    window = triton.cdiv(out_w, in_w) + 2
    block = 512
    with torch_device_fn.device(grad_output.device):
        upsample_linear1d_backward_window_kernel[(triton.cdiv(total, block), 1, 1)](
            grad_output,
            grad_input,
            total,
            in_w,
            out_w,
            ALIGN_CORNERS=align_corners,
            WINDOW=window,
            BLOCK=block,
            isCloseVectorization=True,
        )
    return grad_input.view(input_size)