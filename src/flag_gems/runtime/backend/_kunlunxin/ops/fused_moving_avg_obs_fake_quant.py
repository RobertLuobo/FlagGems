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
def _reduce_minmax_sp(x_ptr, cmin_ptr, cmax_ptr, N, BLOCK: tl.constexpr):
    cur_min = float("inf")
    cur_max = float("-inf")
    for start in range(0, N, BLOCK):
        off = start + tl.arange(0, BLOCK)
        m = off < N
        x = tl.load(x_ptr + off, mask=m, other=0.0).to(tl.float32)
        xmin = tl.where(m, x, float("inf"))
        xmax = tl.where(m, x, float("-inf"))
        cur_min = tl.minimum(cur_min, tl.min(xmin, axis=0))
        cur_max = tl.maximum(cur_max, tl.max(xmax, axis=0))
    tl.store(cmin_ptr, cur_min)
    tl.store(cmax_ptr, cur_max)


@triton.jit
def _fake_quant_kernel(
    x_ptr,
    out_ptr,
    scale,
    zero_point,
    quant_min,
    quant_max,
    N,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    off = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = off < N
    x = tl.load(x_ptr + off, mask=mask, other=0.0).to(tl.float32)

    inv_scale = libdevice.div_rn(1.0, scale)
    qmin_f = quant_min + 0.0
    qmax_f = quant_max + 0.0

    scaled = x * inv_scale
    r = (scaled + 12582912.0) - 12582912.0
    q = r + zero_point
    q = tl.where(q < qmin_f, qmin_f, q)
    q = tl.where(q > qmax_f, qmax_f, q)
    out = (q - zero_point) * scale

    tl.store(out_ptr + off, out, mask=mask)


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
    """Fused moving average observer + fake quantization (Kunlunxin XPU overlay).

    Identical semantics to the generic implementation, but the global min/max
    reduction uses a single-program scan instead of cross-program
    tl.atomic_min/tl.atomic_max. On XPU3 those cross-program atomics silently
    drop concurrent updates (observed: multi-block reductions returned a min of
    -1.0213 vs the true -1.0359), corrupting the observer min/max -> running
    min/max EMA. Fake-quant rounding uses div_rn reciprocal + an explicit
    round-half-to-even (magic constant) to avoid the tl_extra_shim.rint
    libdevice symbol, which miscompiles on XPU3.
    """
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
        global_min = torch.empty(1, dtype=torch.float32, device=dev)
        global_max = torch.empty(1, dtype=torch.float32, device=dev)
        _reduce_minmax_sp[(1,)](x, global_min, global_max, N, BLOCK=BLOCK_SIZE)

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


import sys as _sys  # noqa: E402

# The accuracy test calls flag_gems.ops.fused_moving_avg_obs_fake_quant
# directly (the generic package attribute), which SpecOpRegistrar does not
# override. Patch the generic flag_gems.ops module so the Kunlunxin overlay is
# the one exercised, mirroring the mechanism used by multi_margin_loss.
_generic_ops_module = _sys.modules.get("flag_gems.ops")
if _generic_ops_module is not None:
    setattr(
        _generic_ops_module,
        "fused_moving_avg_obs_fake_quant",
        fused_moving_avg_obs_fake_quant,
    )
