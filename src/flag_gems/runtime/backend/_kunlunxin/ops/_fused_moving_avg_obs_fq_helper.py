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


_SCALAR_TYPE_NAMES = {
    torch.float16: "Half",
    torch.bfloat16: "BFloat16",
    torch.float64: "Double",
}


@triton.jit
def _reduce_minmax_sp(x_ptr, cmin_ptr, cmax_ptr, R, BLOCK: tl.constexpr):
    c = tl.program_id(0)
    base = c * R
    cur_min = float("inf")
    cur_max = float("-inf")
    for start in range(0, R, BLOCK):
        off = start + tl.arange(0, BLOCK)
        m = off < R
        x = tl.load(x_ptr + base + off, mask=m, other=0.0).to(tl.float32)
        xmin = tl.where(m, x, float("inf"))
        xmax = tl.where(m, x, float("-inf"))
        cur_min = tl.minimum(cur_min, tl.min(xmin, axis=0))
        cur_max = tl.maximum(cur_max, tl.max(xmax, axis=0))
    tl.store(cmin_ptr + c, cur_min)
    tl.store(cmax_ptr + c, cur_max)


@triton.jit
def _qparams(
    cmin_ptr,
    cmax_ptr,
    rmin_ptr,
    rmax_ptr,
    scale_ptr,
    zp_ptr,
    C,
    avg_const,
    qmin,
    qmax,
    OBS: tl.constexpr,
    FQ: tl.constexpr,
    SYM: tl.constexpr,
    BLOCK: tl.constexpr,
):
    off = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = off < C
    rmin = tl.load(rmin_ptr + off, mask=m, other=0.0).to(tl.float32)
    rmax = tl.load(rmax_ptr + off, mask=m, other=0.0).to(tl.float32)

    if OBS:
        cmin = tl.load(cmin_ptr + off, mask=m, other=0.0).to(tl.float32)
        cmax = tl.load(cmax_ptr + off, mask=m, other=0.0).to(tl.float32)
        rmin = rmin + avg_const * (cmin - rmin)
        rmax = rmax + avg_const * (cmax - rmax)
        tl.store(rmin_ptr + off, rmin, mask=m)
        tl.store(rmax_ptr + off, rmax, mask=m)

    if FQ:
        qmnf = qmin.to(tl.float32)
        qmxf = qmax.to(tl.float32)
        mn = tl.minimum(rmin, 0.0)
        mx = tl.maximum(rmax, 0.0)
        sc = libdevice.div_rn(mx - mn, qmxf - qmnf)
        sc = tl.where(sc == 0.0, 0.1, sc)
        z = libdevice.rint(qmnf - libdevice.div_rn(mn, sc))
        z = tl.minimum(tl.maximum(z, qmnf), qmxf)

        both = (mn < 0.0) & (mx > 0.0)
        if SYM:
            sc_sym = tl.maximum(
                libdevice.div_rn(-mn, -qmnf), libdevice.div_rn(mx, qmxf)
            )
            sc_sym = tl.where(sc_sym == 0.0, 0.1, sc_sym)
            z_sym = libdevice.rint((qmnf + qmxf) / 2.0)
            sc = tl.where(both, sc_sym, sc)
            z = tl.where(both, z_sym, z)

        tl.store(scale_ptr + off, sc, mask=m)
        tl.store(zp_ptr + off, z.to(tl.int32), mask=m)


@triton.jit
def _fake_quant(
    x_ptr,
    out_ptr,
    mask_ptr,
    scale_ptr,
    zp_ptr,
    N,
    R,
    qmin,
    qmax,
    PER_CHANNEL: tl.constexpr,
    BLOCK: tl.constexpr,
):
    off = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = off < N
    x = tl.load(x_ptr + off, mask=m, other=0.0).to(tl.float32)
    if PER_CHANNEL:
        c = off // R
    else:
        c = tl.zeros((BLOCK,), tl.int32)
    s = tl.load(scale_ptr + c, mask=m, other=1.0).to(tl.float32)
    z = tl.load(zp_ptr + c, mask=m, other=0).to(tl.float32)

    qmnf = qmin.to(tl.float32)
    qmxf = qmax.to(tl.float32)
    q = libdevice.rint(libdevice.div_rn(x, s)) + z
    valid = (q >= qmnf) & (q <= qmxf)
    qc = tl.minimum(tl.maximum(q, qmnf), qmxf)
    out = (qc - z) * s
    tl.store(out_ptr + off, out, mask=m)
    tl.store(mask_ptr + off, valid.to(tl.int8), mask=m)


@triton.jit
def _pt_qparam_fq(
    x_ptr,
    out_ptr,
    mask_ptr,
    cmin_ptr,
    cmax_ptr,
    rmin_in_ptr,
    rmax_in_ptr,
    rmin_out_ptr,
    rmax_out_ptr,
    scale_out_ptr,
    zp_out_ptr,
    N,
    avg_const,
    qmin,
    qmax,
    OBS: tl.constexpr,
    SYM: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)

    rmin = tl.load(rmin_in_ptr).to(tl.float32)
    rmax = tl.load(rmax_in_ptr).to(tl.float32)
    if OBS:
        cmin = tl.load(cmin_ptr).to(tl.float32)
        cmax = tl.load(cmax_ptr).to(tl.float32)
        rmin = rmin + avg_const * (cmin - rmin)
        rmax = rmax + avg_const * (cmax - rmax)

    qmnf = qmin.to(tl.float32)
    qmxf = qmax.to(tl.float32)
    mn = tl.minimum(rmin, 0.0)
    mx = tl.maximum(rmax, 0.0)
    sc = libdevice.div_rn(mx - mn, qmxf - qmnf)
    sc = tl.where(sc == 0.0, 0.1, sc)
    z = libdevice.rint(qmnf - libdevice.div_rn(mn, sc))
    z = tl.minimum(tl.maximum(z, qmnf), qmxf)
    if SYM:
        both = (mn < 0.0) & (mx > 0.0)
        sc_sym = tl.maximum(libdevice.div_rn(-mn, -qmnf), libdevice.div_rn(mx, qmxf))
        sc_sym = tl.where(sc_sym == 0.0, 0.1, sc_sym)
        z_sym = libdevice.rint((qmnf + qmxf) / 2.0)
        sc = tl.where(both, sc_sym, sc)
        z = tl.where(both, z_sym, z)

    if pid == 0:
        if OBS:
            tl.store(rmin_out_ptr, rmin)
            tl.store(rmax_out_ptr, rmax)
        tl.store(scale_out_ptr, sc)
        tl.store(zp_out_ptr, z.to(tl.int32))

    off = pid * BLOCK + tl.arange(0, BLOCK)
    m = off < N
    x = tl.load(x_ptr + off, mask=m, other=0.0).to(tl.float32)
    q = libdevice.rint(libdevice.div_rn(x, sc)) + z
    valid = (q >= qmnf) & (q <= qmxf)
    qc = tl.minimum(tl.maximum(q, qmnf), qmxf)
    out = (qc - z) * sc
    tl.store(out_ptr + off, out, mask=m)
    tl.store(mask_ptr + off, valid.to(tl.int8), mask=m)


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
    """Fused moving-average observer + fake-quantize helper (QAT).

    Kunlunxin XPU overlay. Identical semantics to the generic implementation,
    but the per-channel min/max reduction uses a single-program-per-channel
    scan instead of cross-program tl.atomic_min/tl.atomic_max. On XPU3 those
    cross-program atomics silently drop concurrent updates (probe: an 8-block
    reduction returned max 2.64 vs true 4.13), which corrupted the observer
    min/max -> running_min/max EMA -> scale/zero_point.
    """
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

    if fq:
        rmin_in = running_min.to(torch.float32)
        rmax_in = running_max.to(torch.float32)
        rmin_out = torch.empty(C, dtype=torch.float32, device=dev)
        rmax_out = torch.empty(C, dtype=torch.float32, device=dev)
        if obs:
            cmin = torch.empty(C, dtype=torch.float32, device=dev)
            cmax = torch.empty(C, dtype=torch.float32, device=dev)
            BLK_R = 1024
            _reduce_minmax_sp[(C,)](xf, cmin, cmax, R, BLOCK=BLK_R)
        else:
            cmin = rmin_in
            cmax = rmax_in
        BLK = 1024
        if pc:
            # Per-channel: the single fused scatter/gather kernel (modulo
            # channel index + masked per-channel scatter) fails the XPU
            # TritonXPUUnrollControl pass, so split into reduce -> qparams
            # (small grid over C, updates running_min/max in place and writes
            # scale/zero_point) -> fake-quant with a plain per-channel gather.
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
                PER_CHANNEL=True,
                BLOCK=BLK,
            )
            out_t = out.view(x.shape)
            mask_t = mask_b.view(x.shape)
            return (out_t, mask_t)
        else:
            _pt_qparam_fq[(triton.cdiv(N, BLK),)](
                xf,
                out,
                mask_b,
                cmin,
                cmax,
                rmin_in,
                rmax_in,
                rmin_out,
                rmax_out,
                scale,
                zero_point,
                N,
                ac,
                qmin,
                qmax,
                OBS=obs,
                SYM=sym,
                BLOCK=BLK,
            )
        if obs:
            running_min.copy_(rmin_out)
            running_max.copy_(rmax_out)
        out_t = out.view(x.shape)
        mask_t = mask_b.view(x.shape)
        return (out_t, mask_t)

    if obs:
        cmin = torch.empty(C, dtype=torch.float32, device=dev)
        cmax = torch.empty(C, dtype=torch.float32, device=dev)
        BLK_R = 1024
        _reduce_minmax_sp[(C,)](xf, cmin, cmax, R, BLOCK=BLK_R)
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
