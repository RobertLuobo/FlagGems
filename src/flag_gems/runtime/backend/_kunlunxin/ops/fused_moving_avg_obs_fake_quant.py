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

from flag_gems.ops.fused_moving_avg_obs_fake_quant import _fake_quant_kernel
from flag_gems.runtime import torch_device_fn

logger = logging.getLogger(__name__)


@triton.jit
def _partial_minmax_kernel(
    x_ptr,
    pmin_ptr,
    pmax_ptr,
    N,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    off = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = off < N
    x = tl.load(x_ptr + off, mask=mask, other=0.0).to(tl.float32)
    x_for_min = tl.where(mask, x, float("inf"))
    x_for_max = tl.where(mask, x, float("-inf"))
    tl.store(pmin_ptr + pid, tl.min(x_for_min, axis=0))
    tl.store(pmax_ptr + pid, tl.max(x_for_max, axis=0))


@triton.jit
def _final_minmax_kernel(
    pmin_ptr,
    pmax_ptr,
    gmin_ptr,
    gmax_ptr,
    M,
    BLOCK_SIZE: tl.constexpr,
):
    acc_min = float("inf")
    acc_max = float("-inf")
    for start in range(0, M, BLOCK_SIZE):
        off = start + tl.arange(0, BLOCK_SIZE)
        mask = off < M
        vmn = tl.load(pmin_ptr + off, mask=mask, other=float("inf"))
        vmx = tl.load(pmax_ptr + off, mask=mask, other=float("-inf"))
        acc_min = tl.minimum(acc_min, tl.min(vmn, axis=0))
        acc_max = tl.maximum(acc_max, tl.max(vmx, axis=0))
    tl.store(gmin_ptr, acc_min)
    tl.store(gmax_ptr, acc_max)


def fused_moving_avg_obs_fake_quant(
    input,
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
    logger.debug("GEMS_KUNLUNXIN FUSED_MOVING_AVG_OBS_FAKE_QUANT")

    obs = (
        bool(observer_on.item()) if torch.is_tensor(observer_on) else bool(observer_on)
    )
    fq = (
        bool(fake_quant_on.item())
        if torch.is_tensor(fake_quant_on)
        else bool(fake_quant_on)
    )
    sym = bool(symmetric_quant)

    x = input.contiguous().view(-1).float()
    N = x.numel()
    dev = x.device

    out = torch.empty_like(x)

    BLOCK_SIZE = 1024
    num_blocks = triton.cdiv(N, BLOCK_SIZE)

    qmin_f = float(quant_min)
    qmax_f = float(quant_max)

    if obs:
        # Phase 1: two-stage parallel min/max reduction.
        # Cross-block tl.atomic_min/tl.atomic_max do not merge correctly on
        # XPU3, so each block writes its partial into a dedicated slot and a
        # single final program reduces the partials with a bounded loop.
        partial_min = torch.empty(num_blocks, dtype=torch.float32, device=dev)
        partial_max = torch.empty(num_blocks, dtype=torch.float32, device=dev)
        global_min = torch.empty(1, dtype=torch.float32, device=dev)
        global_max = torch.empty(1, dtype=torch.float32, device=dev)

        with torch_device_fn.device(dev):
            _partial_minmax_kernel[(num_blocks,)](
                x,
                partial_min,
                partial_max,
                N,
                BLOCK_SIZE=BLOCK_SIZE,
            )
            _final_minmax_kernel[(1,)](
                partial_min,
                partial_max,
                global_min,
                global_max,
                num_blocks,
                BLOCK_SIZE=1024,
            )

        cur_min = global_min.item()
        cur_max = global_max.item()
        old_min = running_min.item()
        old_max = running_max.item()
        new_min = old_min + averaging_const * (cur_min - old_min)
        new_max = old_max + averaging_const * (cur_max - old_max)
        running_min.fill_(new_min)
        running_max.fill_(new_max)

    rmin = min(running_min.item(), 0.0)
    rmax = max(running_max.item(), 0.0)
    sc = (rmax - rmin) / (qmax_f - qmin_f)
    if sc == 0.0:
        sc = 0.1
    zp = round(qmin_f - rmin / sc)
    zp = max(qmin_f, min(qmax_f, zp))

    if sym and rmin < 0.0 and rmax > 0.0:
        sc = max(-rmin / (-qmin_f), rmax / qmax_f)
        if sc == 0.0:
            sc = 0.1
        zp = round((qmin_f + qmax_f) / 2.0)

    scale.fill_(sc)
    zero_point.fill_(int(zp))

    sc_val = sc
    zp_val = float(zp)

    if fq:
        with torch_device_fn.device(dev):
            _fake_quant_kernel[(num_blocks,)](
                x,
                out,
                sc_val,
                zp_val,
                quant_min,
                quant_max,
                N,
                BLOCK_SIZE=BLOCK_SIZE,
            )
    else:
        out.copy_(x)

    return out.view(input.shape).to(input.dtype)
