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
import triton.language.extra.libdevice as libdevice

logger = logging.getLogger(__name__)


@triton.jit
def _learnable_per_channel_fq_kernel(
    x_ptr,
    out_ptr,
    scale_ptr,
    zp_ptr,
    N,
    R,
    qmin,
    qmax,
    BLOCK: tl.constexpr,
):
    off = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = off < N
    x = tl.load(x_ptr + off, mask=m, other=0.0).to(tl.float32)
    c = off // R
    s = tl.load(scale_ptr + c, mask=m, other=1.0).to(tl.float32)
    z = tl.load(zp_ptr + c, mask=m, other=0).to(tl.float32)

    qmnf = qmin.to(tl.float32)
    qmxf = qmax.to(tl.float32)
    # Native computes qval = nearbyint(x * (1.0f / scale)) + zp using the
    # correctly-rounded fp32 reciprocal.  Using a true fp32 division
    # (div_rn(x, s)) instead flips round-half-to-even at exact half-way
    # quotients and diverges from the native output by one quant level
    # (observed on the large 4-D shapes).  Emit div_rn(1.0, s) to get the
    # fp32 reciprocal, then a plain fp32 multiply, matching C's 1.0f/scale.
    inv_s = libdevice.div_rn(1.0, s)
    q = libdevice.rint(x * inv_s) + z
    qc = tl.minimum(tl.maximum(q, qmnf), qmxf)
    out = (qc - z) * s
    tl.store(out_ptr + off, out.to(out_ptr.dtype.element_ty), mask=m)


def _prepare_zero_point(zero_point, quant_min, quant_max):
    """Round, clamp and cast the learnable zero point to int, mirroring
    ``_get_rounded_zero_point`` followed by ``.to(kInt)`` in the PyTorch
    native implementation."""
    return zero_point.round().clamp_(quant_min, quant_max).to(torch.int32)


def _run_forward(self, scale, zero_point_int, axis, quant_min, quant_max, out=None):
    if axis < 0:
        axis = self.dim() + axis

    x = self.contiguous()
    N = x.numel()
    C = x.shape[axis]

    # Move the channel axis to the front so the flat element index maps to a
    # channel via ``idx // R`` with R = elements-per-channel.  This matches the
    # native per-channel broadcast without needing a scatter/modulo-index pass
    # (which fails XPU TritonXPUUnrollControl).
    xp = x.movedim(axis, 0).contiguous()
    R = N // C if C > 0 else N

    xf = xp.view(-1)
    out_buf = torch.empty_like(xf)
    scale_f = scale.contiguous().to(torch.float32)
    zp_f = zero_point_int.contiguous().to(torch.int32)

    BLK = 1024
    _learnable_per_channel_fq_kernel[(triton.cdiv(N, BLK),)](
        xf,
        out_buf,
        scale_f,
        zp_f,
        N,
        R,
        int(quant_min),
        int(quant_max),
        BLOCK=BLK,
    )

    result = out_buf.view(xp.shape).movedim(0, axis).contiguous()
    if out is not None:
        out.copy_(result)
        return out
    return result


def _fake_quantize_learnable_per_channel_affine(
    self, scale, zero_point, axis, quant_min, quant_max, grad_factor=1.0
):
    logger.debug("GEMS_KUNLUNXIN _FAKE_QUANTIZE_LEARNABLE_PER_CHANNEL_AFFINE")
    if self.dtype not in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
        raise TypeError(
            f"unsupported dtype {self.dtype}; only floating-point dtypes are supported"
        )
    zero_point_rounded = _prepare_zero_point(zero_point, quant_min, quant_max)
    return _run_forward(self, scale, zero_point_rounded, axis, quant_min, quant_max)


def _fake_quantize_learnable_per_channel_affine_out_impl(
    self,
    scale,
    zero_point,
    axis,
    quant_min,
    quant_max,
    grad_factor=1.0,
    *,
    out=None,
):
    logger.debug("GEMS_KUNLUNXIN _FAKE_QUANTIZE_LEARNABLE_PER_CHANNEL_AFFINE_OUT")
    zero_point_rounded = _prepare_zero_point(zero_point, quant_min, quant_max)
    return _run_forward(
        self, scale, zero_point_rounded, axis, quant_min, quant_max, out=out
    )
