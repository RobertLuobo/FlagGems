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
import math
import warnings

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)


@triton.jit
def _trilinear_backward_isinf(value):
    if value.dtype == tl.float64:
        bits = value.to(tl.int64, bitcast=True) & 0x7FFFFFFFFFFFFFFF
        return bits == 0x7FF0000000000000
    else:
        bits = value.to(tl.int32, bitcast=True) & 0x7FFFFFFF
        return bits == 0x7F800000


@triton.jit
def _trilinear_backward_weight(
    output_index,
    input_index,
    INPUT_SIZE: tl.constexpr,
    SCALE: tl.constexpr,
    ALIGN_CORNERS: tl.constexpr,
    ACC: tl.constexpr,
):
    scale = tl.full((), SCALE, ACC)
    if ALIGN_CORNERS:
        real = output_index.to(ACC) * scale
    else:
        real = tl.maximum((output_index.to(ACC) + 0.5) * scale - 0.5, 0.0)
    lower = tl.minimum(real, INPUT_SIZE - 1).to(input_index.dtype)
    lower = tl.minimum(lower, INPUT_SIZE - 1)
    upper = tl.minimum(lower + 1, INPUT_SIZE - 1)
    fraction = tl.minimum(real - lower.to(ACC), 1.0)
    weight = tl.where(input_index == lower, 1.0 - fraction, 0.0)
    weight += tl.where(input_index == upper, fraction, 0.0)
    if ACC == tl.float64:
        bits = fraction.to(tl.int64, bitcast=True) & 0x7FFFFFFFFFFFFFFF
        zero_fraction = bits == 0
        one_fraction = bits == 0x3FF0000000000000
    else:
        zero_fraction = fraction == 0.0
        one_fraction = fraction == 1.0
    zero_tap = ((input_index == lower) & one_fraction) | (
        (input_index == upper) & zero_fraction
    )
    match = (input_index == lower) | (input_index == upper)
    return weight, match, zero_tap


@libentry()
@triton.jit
def _upsample_trilinear3d_backward_kernel(
    GradOutput,
    GradInput,
    N: tl.constexpr,
    C: tl.constexpr,
    ID: tl.constexpr,
    IH: tl.constexpr,
    IW: tl.constexpr,
    OD: tl.constexpr,
    OH: tl.constexpr,
    OW: tl.constexpr,
    GS0: tl.constexpr,
    GS1: tl.constexpr,
    GS2: tl.constexpr,
    GS3: tl.constexpr,
    GS4: tl.constexpr,
    IS0: tl.constexpr,
    IS1: tl.constexpr,
    IS2: tl.constexpr,
    IS3: tl.constexpr,
    IS4: tl.constexpr,
    SD: tl.constexpr,
    SH: tl.constexpr,
    SW: tl.constexpr,
    ALIGN_CORNERS: tl.constexpr,
    KD: tl.constexpr,
    KH: tl.constexpr,
    KW: tl.constexpr,
    FP64: tl.constexpr,
    INDEX64: tl.constexpr,
    PROGRAMS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    total = N * C * ID * IH * IW
    for tile in range(tl.cdiv(total, PROGRAMS * BLOCK)):
        block_id = tl.program_id(0) + tile * PROGRAMS
        if INDEX64:
            offsets = block_id.to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
        else:
            offsets = block_id * BLOCK + tl.arange(0, BLOCK)
        valid = offsets < total
        x = offsets % IW
        y = offsets // IW % IH
        z = offsets // (IW * IH) % ID
        c = offsets // (IW * IH * ID) % C
        n = offsets // (IW * IH * ID * C)

        destination = (
            GradInput + n * IS0 + c * IS1 + z * IS2 + y * IS3 + x * IS4
        )
        source = GradOutput + n * GS0 + c * GS1

        if FP64:
            acc_dtype: tl.constexpr = tl.float64
        else:
            acc_dtype: tl.constexpr = tl.float32
        shift: tl.constexpr = 0.0 if ALIGN_CORNERS else 0.5

        if SD == 0.0 or ID == 1:
            start_z = tl.full((BLOCK,), 0, offsets.dtype)
        else:
            inv_d = tl.full((), 1.0 / SD, acc_dtype)
            start_z = tl.maximum(
                tl.floor((z.to(acc_dtype) - 1.0 + shift) * inv_d - shift).to(
                    offsets.dtype
                ),
                0,
            )
        if SH == 0.0 or IH == 1:
            start_y = tl.full((BLOCK,), 0, offsets.dtype)
        else:
            inv_h = tl.full((), 1.0 / SH, acc_dtype)
            start_y = tl.maximum(
                tl.floor((y.to(acc_dtype) - 1.0 + shift) * inv_h - shift).to(
                    offsets.dtype
                ),
                0,
            )
        if SW == 0.0 or IW == 1:
            start_x = tl.full((BLOCK,), 0, offsets.dtype)
        else:
            inv_w = tl.full((), 1.0 / SW, acc_dtype)
            start_x = tl.maximum(
                tl.floor((x.to(acc_dtype) - 1.0 + shift) * inv_w - shift).to(
                    offsets.dtype
                ),
                0,
            )

        result = tl.full((BLOCK,), 0.0, acc_dtype)
        for dz in range(KD):
            oz = start_z + dz
            wz, match_z, zero_z = _trilinear_backward_weight(
                oz, z, ID, SD, ALIGN_CORNERS, acc_dtype
            )
            for dy in range(KH):
                oy = start_y + dy
                wy, match_y, zero_y = _trilinear_backward_weight(
                    oy, y, IH, SH, ALIGN_CORNERS, acc_dtype
                )
                for dx in range(KW):
                    ox = start_x + dx
                    wx, match_x, zero_x = _trilinear_backward_weight(
                        ox, x, IW, SW, ALIGN_CORNERS, acc_dtype
                    )
                    active = (
                        valid
                        & (oz < OD)
                        & (oy < OH)
                        & (ox < OW)
                        & match_z
                        & match_y
                        & match_x
                    )
                    value = tl.load(
                        source + oz * GS2 + oy * GS3 + ox * GS4,
                        active,
                        other=0,
                    ).to(acc_dtype)
                    contribution = (wz * wy * wx) * value
                    contribution = tl.where(
                        (zero_z | zero_y | zero_x)
                        & _trilinear_backward_isinf(value),
                        float("nan"),
                        contribution,
                    )
                    result += tl.where(active, contribution, 0.0)
        tl.store(destination, result.to(GradInput.dtype.element_ty), mask=valid)


def _axis_scale(in_sz, out_sz, align_corners, scale):
    if align_corners:
        return (in_sz - 1) / (out_sz - 1) if out_sz > 1 else 0.0
    s = 1.0 / scale if scale is not None else in_sz / out_sz
    if (out_sz - 0.5) * s <= 0.25:
        s = 0.0
    return s


def _axis_span(in_sz, out_sz, s, align_corners):
    if s == 0 or in_sz == 1:
        return out_sz
    k = min(out_sz, math.ceil(2 / s) + 2)
    shift = 0.0 if align_corners else 0.5
    tail = out_sz - max(0, math.floor((in_sz - 2 + shift) / s - shift))
    return min(out_sz, max(k, tail))


def upsample_trilinear3d_backward(
    grad_output,
    output_size,
    input_size,
    align_corners,
    scales_d=None,
    scales_h=None,
    scales_w=None,
    *,
    grad_input=None,
):
    logging.getLogger("flag_gems.ops.upsample_trilinear3d_backward").debug(
        "GEMS UPSAMPLE_TRILINEAR3D_BACKWARD"
    )
    logger.debug("GEMS_KUNLUNXIN UPSAMPLE_TRILINEAR3D_BACKWARD")
    if len(output_size) != 3 or len(input_size) != 5:
        raise RuntimeError(
            "output_size must have 3 elements and input_size must have 5 elements"
        )
    n, c, idp, ih, iw = (int(v) for v in input_size)
    od, oh, ow = (int(v) for v in output_size)
    if min(idp, ih, iw, od, oh, ow) <= 0 or n < 0 or c < 0:
        raise RuntimeError("input and output spatial sizes must be greater than 0")
    if grad_output.ndim != 5:
        raise RuntimeError("Expected grad_output to be a tensor of dimension 5")
    if tuple(grad_output.shape) != (n, c, od, oh, ow):
        raise RuntimeError("Expected grad_output to have the same shape as output")
    if any(
        scale is not None and not scale > 0 for scale in (scales_d, scales_h, scales_w)
    ):
        raise RuntimeError("scales must be greater than 0")
    if grad_output.dtype not in (
        torch.float16,
        torch.bfloat16,
        torch.float32,
        torch.float64,
    ):
        raise RuntimeError(
            "upsample_trilinear3d_backward requires a floating point dtype"
        )

    if grad_input is None:
        grad_input = torch.empty(
            (n, c, idp, ih, iw), dtype=grad_output.dtype, device=grad_output.device
        )
    else:
        if grad_input.dtype != grad_output.dtype:
            raise RuntimeError(
                "Expected grad_input to have the same dtype as grad_output"
            )
        if grad_input.device != grad_output.device:
            raise RuntimeError("Expected all tensors to be on the same device")
        if tuple(grad_input.shape) != (n, c, idp, ih, iw):
            if grad_input.numel() != 0:
                warnings.warn(
                    "An output with one or more elements was resized because its "
                    "shape did not match input_size",
                    UserWarning,
                    stacklevel=2,
                )
            grad_input.resize_((n, c, idp, ih, iw))
    if n == 0 or c == 0 or grad_input.numel() == 0:
        return grad_input

    grad_output = grad_output.contiguous()
    sd = _axis_scale(idp, od, align_corners, scales_d)
    sh = _axis_scale(ih, oh, align_corners, scales_h)
    sw = _axis_scale(iw, ow, align_corners, scales_w)
    kd = _axis_span(idp, od, sd, align_corners)
    kh = _axis_span(ih, oh, sh, align_corners)
    kw = _axis_span(iw, ow, sw, align_corners)

    gs = grad_output.stride()
    strides = grad_input.stride()
    total = n * c * idp * ih * iw
    block = 128
    largest_offset = max(
        (n - 1) * gs[0]
        + (c - 1) * gs[1]
        + (od - 1) * gs[2]
        + (oh - 1) * gs[3]
        + (ow - 1) * gs[4],
        (n - 1) * strides[0]
        + (c - 1) * strides[1]
        + (idp - 1) * strides[2]
        + (ih - 1) * strides[3]
        + (iw - 1) * strides[4],
    )
    index64 = (
        max(total, n * c * od * oh * ow, largest_offset) + block >= 2**31
        or max(idp, ih, iw, od, oh, ow) >= 2**24
    )

    with torch_device_fn.device(grad_output.device):
        programs = triton.cdiv(total, block)
        _upsample_trilinear3d_backward_kernel[(programs,)](
            grad_output,
            grad_input,
            n,
            c,
            idp,
            ih,
            iw,
            od,
            oh,
            ow,
            *gs,
            *strides,
            sd,
            sh,
            sw,
            align_corners,
            kd,
            kh,
            kw,
            grad_output.dtype == torch.float64,
            index64,
            programs,
            block,
            enable_fp_fusion=False,
        )
    return grad_input


def upsample_trilinear3d_backward_grad_input(
    grad_output,
    output_size,
    input_size,
    align_corners,
    scales_d=None,
    scales_h=None,
    scales_w=None,
    *,
    grad_input,
):
    return upsample_trilinear3d_backward(
        grad_output,
        output_size,
        input_size,
        align_corners,
        scales_d,
        scales_h,
        scales_w,
        grad_input=grad_input,
    )


def _install_into_flag_gems_ops():
    try:
        import flag_gems.ops as _fg_ops
    except Exception:  # pragma: no cover - defensive; import ordering
        return
    _fg_ops.upsample_trilinear3d_backward = upsample_trilinear3d_backward
    _fg_ops.upsample_trilinear3d_backward_grad_input = (
        upsample_trilinear3d_backward_grad_input
    )


_install_into_flag_gems_ops()
