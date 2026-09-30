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

from flag_gems.ops._fused_moving_avg_obs_fq_helper import (
    _SCALAR_TYPE_NAMES,
    _fake_quant,
    _qparams,
)

logger = logging.getLogger(__name__)


@triton.jit
def _reduce_minmax_row(x_ptr, cmin_ptr, cmax_ptr, R, BLOCK: tl.constexpr):
    # One program per channel reduces channel c's R-length row with a dynamic
    # scf.for loop over BLOCK-sized chunks. This avoids cross-block
    # tl.atomic_min/tl.atomic_max into a shared cmin/cmax slot, which loses
    # updates under contention on XPU3 and yields wrong per-channel min/max.
    c = tl.program_id(0)
    base = c * R
    vmin = float("inf")
    vmax = float("-inf")
    for start in range(0, R, BLOCK):
        j = start + tl.arange(0, BLOCK)
        m = j < R
        x = tl.load(x_ptr + base + j, mask=m, other=0.0).to(tl.float32)
        xmin = tl.where(m, x, float("inf"))
        xmax = tl.where(m, x, float("-inf"))
        vmin = tl.minimum(vmin, tl.min(xmin, axis=0))
        vmax = tl.maximum(vmax, tl.max(xmax, axis=0))
    tl.store(cmin_ptr + c, vmin)
    tl.store(cmax_ptr + c, vmax)


def _fused_moving_avg_obs_fq_helper(
    self,
    observer_on,
    fake_quant_on,
    running_min,
    running_max,
    scale,
    zero_point,
    averaging_const,
    quant_min,
    quant_max,
    ch_axis,
    per_row_fake_quant=False,
    symmetric_quant=False,
):
    logger.debug("GEMS_KUNLUNXIN _FUSED_MOVING_AVG_OBS_FQ_HELPER")
    if self.dtype is not torch.float32:
        scalar_type = _SCALAR_TYPE_NAMES.get(self.dtype, str(self.dtype))
        raise RuntimeError(f"expected scalar type Float but found {scalar_type}")

    x = self
    dev = x.device
    obs = int(observer_on)
    fq = int(fake_quant_on)
    sym = bool(symmetric_quant)
    pc = bool(per_row_fake_quant)
    qmin = int(quant_min)
    qmax = int(quant_max)
    ac = float(averaging_const)

    N = x.numel()
    xf = x.contiguous().view(-1)

    if pc:
        C = x.shape[int(ch_axis)]
        R = N // C
    else:
        C = 1
        R = N

    out = torch.empty_like(xf, dtype=torch.float32)
    mask_b = torch.empty(N, dtype=torch.bool, device=dev)

    # fake-quant-on path. The generic backend fuses qparam compute + fake-quant
    # into a single monolithic kernel (_fused_fq / _pt_qparam_fq). On XPU3 the
    # per-channel _fused_fq blows up TritonXPUUnrollControl ("operand does not
    # dominate" / uni_sram OOR). Decompose into simple, separately-compilable
    # kernels: per-channel min/max reduction (non-atomic) -> _qparams (updates
    # running stats in place + computes scale/zp) -> _fake_quant (per-element).
    if fq:
        if obs:
            cmin = torch.empty(C, dtype=torch.float32, device=dev)
            cmax = torch.empty(C, dtype=torch.float32, device=dev)
            _reduce_minmax_row[(C,)](xf, cmin, cmax, R, BLOCK=1024)
        else:
            cmin = running_min
            cmax = running_max

        BLK_Q = 128
        _qparams[(triton.cdiv(C, BLK_Q),)](
            cmin,
            cmax,
            running_min,
            running_max,
            scale,
            zero_point,
            C,
            ac,
            qmin,
            qmax,
            OBS=obs,
            FQ=fq,
            SYM=sym,
            BLOCK=BLK_Q,
        )

        BLK = 1024
        _fake_quant[(triton.cdiv(N, BLK),)](
            xf,
            out,
            mask_b,
            scale,
            zero_point,
            N,
            R,
            qmin,
            qmax,
            PER_CHANNEL=pc,
            BLOCK=BLK,
        )
        out_t = out.view(x.shape)
        mask_t = mask_b.view(x.shape)
        return (out_t, mask_t)

    if obs:
        cmin = torch.empty(C, dtype=torch.float32, device=dev)
        cmax = torch.empty(C, dtype=torch.float32, device=dev)
        _reduce_minmax_row[(C,)](xf, cmin, cmax, R, BLOCK=1024)
    else:
        cmin = running_min
        cmax = running_max

    BLK_Q = 128
    _qparams[(triton.cdiv(C, BLK_Q),)](
        cmin,
        cmax,
        running_min,
        running_max,
        scale,
        zero_point,
        C,
        ac,
        qmin,
        qmax,
        OBS=obs,
        FQ=fq,
        SYM=sym,
        BLOCK=BLK_Q,
    )

    out_t = xf.to(torch.float32).view(x.shape)
    mask_t = torch.ones(x.shape, dtype=torch.bool, device=dev)
    return (out_t, mask_t)
