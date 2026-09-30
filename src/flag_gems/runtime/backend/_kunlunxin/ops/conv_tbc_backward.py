# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)


@libentry()
@triton.jit
def _bias_grad_kernel(
    dy,
    gb,
    M,
    N,
    s_m,
    s_n,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    o = pid * BLOCK + tl.arange(0, BLOCK)
    omask = o < N
    acc = tl.zeros((BLOCK,), tl.float32)
    for m in range(0, M):
        v = tl.load(dy + m * s_m + o * s_n, mask=omask, other=0.0)
        acc += v.to(tl.float32)
    tl.store(gb + o, acc.to(gb.dtype.element_ty), mask=omask)


@libentry()
@triton.jit
def _weight_grad_kernel(
    inp,
    dy,
    gw,
    ilen,
    batch,
    in_c,
    out_c,
    olen,
    kw,
    pad,
    i_s0,
    i_s1,
    i_s2,
    d_s0,
    d_s1,
    d_s2,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    idx = pid * BLOCK + tl.arange(0, BLOCK)
    total = kw * in_c * out_c
    kmask = idx < total
    kk = idx // (in_c * out_c)
    rem = idx % (in_c * out_c)
    ic = rem // out_c
    oc = rem % out_c
    acc = tl.zeros((BLOCK,), tl.float32)
    for o in range(0, olen):
        it = o + kk - pad
        valid = kmask & (it >= 0) & (it < ilen)
        safe_it = tl.where(valid, it, 0)
        for b in range(0, batch):
            xv = tl.load(
                inp + safe_it * i_s0 + b * i_s1 + ic * i_s2,
                mask=valid,
                other=0.0,
            )
            xv = tl.where(valid, xv.to(tl.float32), 0.0)
            gy = tl.load(
                dy + o * d_s0 + b * d_s1 + oc * d_s2,
                mask=kmask,
                other=0.0,
            )
            acc += xv * gy.to(tl.float32)
    tl.store(gw + idx, acc.to(gw.dtype.element_ty), mask=kmask)


@libentry()
@triton.jit
def _input_grad_kernel(
    dy,
    w,
    gi,
    ilen,
    batch,
    in_c,
    out_c,
    olen,
    kw,
    pad,
    d_s0,
    d_s1,
    d_s2,
    w_s0,
    w_s1,
    w_s2,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    idx = pid * BLOCK + tl.arange(0, BLOCK)
    total = ilen * batch * in_c
    imask = idx < total
    it = idx // (batch * in_c)
    rem = idx % (batch * in_c)
    b = rem // in_c
    ic = rem % in_c
    acc = tl.zeros((BLOCK,), tl.float32)
    for kk in range(0, kw):
        o = it - kk + pad
        valid = imask & (o >= 0) & (o < olen)
        safe_o = tl.where(valid, o, 0)
        for oc in range(0, out_c):
            gy = tl.load(
                dy + safe_o * d_s0 + b * d_s1 + oc * d_s2,
                mask=valid,
                other=0.0,
            )
            gy = tl.where(valid, gy.to(tl.float32), 0.0)
            wv = tl.load(
                w + kk * w_s0 + ic * w_s1 + oc * w_s2,
                mask=imask,
                other=0.0,
            )
            acc += gy * wv.to(tl.float32)
    tl.store(gi + idx, acc.to(gi.dtype.element_ty), mask=imask)


def conv_tbc_backward(grad_output, input, weight, bias, pad):
    logger.debug("GEMS_KUNLUNXIN CONV_TBC_BACKWARD")

    ilen, batch, in_c = input.shape
    kw, _, out_c = weight.shape
    olen = grad_output.shape[0]

    grad_output = grad_output.contiguous()
    input = input.contiguous()
    weight = weight.contiguous()

    grad_bias = torch.empty(
        (out_c,), device=grad_output.device, dtype=grad_output.dtype
    )
    grad_weight = torch.empty(
        (kw, in_c, out_c), device=weight.device, dtype=weight.dtype
    )
    grad_input = torch.empty(
        (ilen, batch, in_c), device=input.device, dtype=input.dtype
    )

    go2d = grad_output.reshape(olen * batch, out_c)

    with torch_device_fn.device(grad_output.device):
        BLOCK = 64
        _bias_grad_kernel[(triton.cdiv(out_c, BLOCK),)](
            go2d,
            grad_bias,
            olen * batch,
            out_c,
            go2d.stride(0),
            go2d.stride(1),
            BLOCK=BLOCK,
        )

        w_total = kw * in_c * out_c
        _weight_grad_kernel[(triton.cdiv(w_total, BLOCK),)](
            input,
            grad_output,
            grad_weight,
            ilen,
            batch,
            in_c,
            out_c,
            olen,
            kw,
            pad,
            input.stride(0),
            input.stride(1),
            input.stride(2),
            grad_output.stride(0),
            grad_output.stride(1),
            grad_output.stride(2),
            BLOCK=BLOCK,
        )

        i_total = ilen * batch * in_c
        _input_grad_kernel[(triton.cdiv(i_total, BLOCK),)](
            grad_output,
            weight,
            grad_input,
            ilen,
            batch,
            in_c,
            out_c,
            olen,
            kw,
            pad,
            grad_output.stride(0),
            grad_output.stride(1),
            grad_output.stride(2),
            weight.stride(0),
            weight.stride(1),
            weight.stride(2),
            BLOCK=BLOCK,
        )

    return grad_input, grad_weight, grad_bias
