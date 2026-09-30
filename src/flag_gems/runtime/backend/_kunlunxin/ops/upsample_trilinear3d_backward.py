# Copyright 2026, The FlagOS Contributors.
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
from typing import Optional, Sequence

import torch
import triton
import triton.language as tl

import flag_gems

logger = logging.getLogger(__name__)


@triton.jit
def _wmat_kernel(
    w_ptr,
    OUT_SZ,
    IN_SZ,
    scale,
    bias,
    SAME: tl.constexpr,
    BLOCK_I: tl.constexpr,
):
    o = tl.program_id(0)
    if SAME:
        src = o.to(tl.float32)
    else:
        src = o.to(tl.float32) * scale + bias
    src = tl.maximum(0.0, tl.minimum(src, IN_SZ - 1.0))
    i0 = tl.floor(src).to(tl.int32)
    i1 = tl.minimum(i0 + 1, IN_SZ - 1)
    t = src - i0.to(tl.float32)
    cols = tl.arange(0, BLOCK_I)
    w = tl.where(cols == i0, 1.0 - t, 0.0) + tl.where(cols == i1, t, 0.0)
    tl.store(w_ptr + o * IN_SZ + cols, w, mask=cols < IN_SZ)


def _calc_scale_and_bias(in_sz, out_sz, align_corners, scale):
    if align_corners:
        if out_sz > 1:
            scale_val = (in_sz - 1.0) / (out_sz - 1.0)
        else:
            scale_val = 0.0
        bias_val = 0.0
    else:
        if scale is not None and scale > 0:
            real_scale = 1.0 / scale
        else:
            real_scale = in_sz / out_sz
        scale_val = real_scale
        bias_val = 0.5 * real_scale - 0.5
    return scale_val, bias_val


def _build_wmat(out_sz, in_sz, align_corners, scale, device):
    w = torch.empty((out_sz, in_sz), device=device, dtype=torch.float32)
    scale_val, bias_val = _calc_scale_and_bias(in_sz, out_sz, align_corners, scale)
    BLOCK_I = triton.next_power_of_2(in_sz)
    _wmat_kernel[(out_sz,)](
        w,
        out_sz,
        in_sz,
        scale_val,
        bias_val,
        SAME=(in_sz == out_sz),
        BLOCK_I=BLOCK_I,
    )
    return w


def upsample_trilinear3d_backward(
    grad_output: torch.Tensor,
    output_size: Sequence[int],
    input_size: Sequence[int],
    align_corners: bool,
    scales_d: Optional[float] = None,
    scales_h: Optional[float] = None,
    scales_w: Optional[float] = None,
) -> torch.Tensor:
    logger.debug("GEMS UPSAMPLE_TRILINEAR3D_BACKWARD")

    N, C, ID, IH, IW = input_size
    OD, OH, OW = output_size
    NC = N * C
    device = grad_output.device

    grad_out = grad_output.contiguous().view(NC, OD, OH, OW).float()

    if grad_out.numel() == 0 or ID * IH * IW == 0:
        return torch.zeros(
            (N, C, ID, IH, IW), device=device, dtype=grad_output.dtype
        )

    # The trilinear backward is separable across (d, h, w): grad_input is the
    # transposed interpolation, i.e. three sequential contractions with the
    # per-axis weight matrices Wd/Wh/Ww. This avoids tl.atomic_add scatter,
    # whose updates are silently dropped on this backend. The h/d contractions
    # use bmm with a shared (broadcast) weight so the large activation tensor is
    # never permute-copied; only the tiny weight is materialized per batch.
    Wd = _build_wmat(OD, ID, align_corners, scales_d, device)
    Wh = _build_wmat(OH, IH, align_corners, scales_h, device)
    Ww = _build_wmat(OW, IW, align_corners, scales_w, device)

    x = flag_gems.mm(grad_out.reshape(NC * OD * OH, OW), Ww)
    WhT = Wh.t().unsqueeze(0).expand(NC * OD, IH, OH).contiguous()
    x = flag_gems.bmm(WhT, x.reshape(NC * OD, OH, IW))
    WdT = Wd.t().unsqueeze(0).expand(NC, ID, OD).contiguous()
    x = flag_gems.bmm(WdT, x.reshape(NC, OD, IH * IW))

    return x.reshape(N, C, ID, IH, IW).to(grad_output.dtype)
