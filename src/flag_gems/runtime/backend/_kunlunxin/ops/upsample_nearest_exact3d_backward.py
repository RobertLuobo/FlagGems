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
import warnings
from functools import lru_cache

import numpy as np
import torch
import triton
import triton.language as tl

from flag_gems.runtime import device as runtime_device
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1024)
def _boundary_indices(length_in, length_out, scale):
    if scale is not None and scale > 0:
        scale_inv = np.float32(1.0 / np.float32(scale))

        def src(o):
            return min(
                int((np.float32(o) + np.float32(0.5)) * scale_inv), length_in - 1
            )

    elif length_in == length_out:

        def src(o):
            return o

    elif length_out == 2 * length_in:

        def src(o):
            return o >> 1

    else:
        scale_inv = np.float32(length_in) / np.float32(length_out)

        def src(o):
            return min(
                int((np.float32(o) + np.float32(0.5)) * scale_inv), length_in - 1
            )

    bounds = [0] * (length_in + 1)
    o = 0
    for i in range(length_in + 1):
        while o < length_out and src(o) < i:
            o += 1
        bounds[i] = o
    return tuple(bounds)


def _is_identity(bounds, length_in, length_out):
    return length_in == length_out and bounds == tuple(range(length_in + 1))


_boundary_tensor_cache = {}


def _boundary_tensor(bounds, device):
    key = (bounds, device.type, device.index)
    tensor = _boundary_tensor_cache.get(key)
    if tensor is None:
        tensor = torch.tensor(bounds, dtype=torch.int32, device=device)
        _boundary_tensor_cache[key] = tensor
    return tensor


@libentry()
@triton.jit
def _upsample_nearest_exact3d_backward_kernel(
    GO,
    GI,
    DB,
    HB,
    WB,
    TOTAL: tl.constexpr,
    C: tl.constexpr,
    ID: tl.constexpr,
    IH: tl.constexpr,
    IW: tl.constexpr,
    OD: tl.constexpr,
    OH: tl.constexpr,
    OW: tl.constexpr,
    GO_N: tl.constexpr,
    GO_C: tl.constexpr,
    GO_D: tl.constexpr,
    GO_H: tl.constexpr,
    GO_W: tl.constexpr,
    GI_N: tl.constexpr,
    GI_C: tl.constexpr,
    GI_D: tl.constexpr,
    GI_H: tl.constexpr,
    GI_W: tl.constexpr,
    IDENTITY: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    index = pid * BLOCK + tl.arange(0, BLOCK)
    valid = index < TOTAL
    index = tl.where(valid, index, 0)
    iw = index % IW
    ih = index // IW % IH
    id_ = index // (IW * IH) % ID
    channel = index // (IW * IH * ID) % C
    batch = index // (C * IW * IH * ID)

    output_offset = (
        batch * GI_N + channel * GI_C + id_ * GI_D + ih * GI_H + iw * GI_W
    )
    input_base = batch * GO_N + channel * GO_C
    if IDENTITY:
        value = tl.load(
            GO + input_base + id_ * GO_D + ih * GO_H + iw * GO_W, valid, other=0
        )
        tl.store(GI + output_offset, value, valid)
    else:
        d0 = tl.load(DB + id_, valid, other=0)
        d1 = tl.load(DB + id_ + 1, valid, other=0)
        h0 = tl.load(HB + ih, valid, other=0)
        h1 = tl.load(HB + ih + 1, valid, other=0)
        w0 = tl.load(WB + iw, valid, other=0)
        w1 = tl.load(WB + iw + 1, valid, other=0)
        if GI.dtype.element_ty == tl.float64:
            acc = tl.full((BLOCK,), 0, tl.float64)
        elif GI.dtype.element_ty == tl.uint8:
            acc = tl.full((BLOCK,), 0, tl.int32)
        else:
            acc = tl.full((BLOCK,), 0, tl.float32)
        # Dynamic scf.for over the runtime source span. A static unroll of the
        # forward mapping miscompiles in TritonXPUUnrollControl on XPU3.
        d_count = tl.max(tl.where(valid, d1 - d0, 0), 0)
        h_count = tl.max(tl.where(valid, h1 - h0, 0), 0)
        w_count = tl.max(tl.where(valid, w1 - w0, 0), 0)
        for dz in range(d_count):
            oz = d0 + dz
            for dy in range(h_count):
                oy = h0 + dy
                for dx in range(w_count):
                    ox = w0 + dx
                    in_range = valid & (oz < d1) & (oy < h1) & (ox < w1)
                    # Clamp to a valid output coordinate so masked-off lanes
                    # never form an out-of-bounds address, then mask the value
                    # explicitly: an XPU masked gather still materialises every
                    # lane, so empty source ranges must be zeroed here.
                    coz = tl.minimum(tl.maximum(oz, 0), OD - 1)
                    coy = tl.minimum(tl.maximum(oy, 0), OH - 1)
                    cox = tl.minimum(tl.maximum(ox, 0), OW - 1)
                    value = tl.load(
                        GO + input_base + coz * GO_D + coy * GO_H + cox * GO_W,
                        in_range,
                        other=0,
                    )
                    acc += tl.where(in_range, value.to(acc.dtype), 0)
        tl.store(GI + output_offset, acc, valid)


@libentry()
@triton.jit
def _upsample_nearest_exact3d_backward_scalar_kernel(
    GO,
    GI,
    DB,
    HB,
    WB,
    TOTAL,
    C: tl.constexpr,
    ID: tl.constexpr,
    IH: tl.constexpr,
    IW: tl.constexpr,
    GO_N: tl.constexpr,
    GO_C: tl.constexpr,
    GO_D: tl.constexpr,
    GO_H: tl.constexpr,
    GO_W: tl.constexpr,
    GI_N: tl.constexpr,
    GI_C: tl.constexpr,
    GI_D: tl.constexpr,
    GI_H: tl.constexpr,
    GI_W: tl.constexpr,
    IDENTITY: tl.constexpr,
):
    # Scalar 64-bit path for tensors whose element strides exceed int32.
    for index in range(tl.program_id(0).to(tl.int64), TOTAL, tl.num_programs(0)):
        iw = index % IW
        ih = index // IW % IH
        id_ = index // (IW * IH) % ID
        channel = index // (IW * IH * ID) % C
        batch = index // (C * IW * IH * ID)
        input_base = GO + batch * GO_N + channel * GO_C
        output = (
            GI + batch * GI_N + channel * GI_C + id_ * GI_D + ih * GI_H + iw * GI_W
        )
        if IDENTITY:
            value = tl.load(input_base + id_ * GO_D + ih * GO_H + iw * GO_W)
        else:
            d0 = tl.load(DB + id_).to(tl.int64)
            d1 = tl.load(DB + id_ + 1).to(tl.int64)
            h0 = tl.load(HB + ih).to(tl.int64)
            h1 = tl.load(HB + ih + 1).to(tl.int64)
            w0 = tl.load(WB + iw).to(tl.int64)
            w1 = tl.load(WB + iw + 1).to(tl.int64)
            if GI.dtype.element_ty == tl.float64:
                value = tl.full((), 0, tl.float64)
            elif GI.dtype.element_ty == tl.uint8:
                value = tl.full((), 0, tl.int32)
            else:
                value = tl.full((), 0, tl.float32)
            for oz in range(d0, d1):
                for oy in range(h0, h1):
                    for ox in range(w0, w1):
                        value += tl.load(
                            input_base + oz * GO_D + oy * GO_H + ox * GO_W
                        ).to(value.dtype)
        tl.store(output, value)


def _upsample_nearest_exact3d_backward_impl(
    grad_output,
    output_size,
    input_size,
    scales_d,
    scales_h,
    scales_w,
    grad_input,
):
    if len(output_size) != 3 or len(input_size) != 5:
        raise RuntimeError("output_size must have length 3 and input_size length 5")
    n, c, id_, ih, iw = input_size
    od, oh, ow = output_size
    if min(id_, ih, iw, od, oh, ow) <= 0 or n < 0 or c < 0:
        raise RuntimeError("Input and output spatial sizes must be greater than 0")
    if grad_output.ndim != 5:
        raise RuntimeError("Expected grad_output to be a tensor of dimension 5")
    if tuple(grad_output.shape) != (n, c, od, oh, ow):
        raise RuntimeError("Expected grad_output to have the same shape as output")
    if grad_output.dtype not in (
        torch.float16,
        torch.bfloat16,
        torch.float32,
        torch.float64,
        torch.uint8,
    ):
        raise RuntimeError(
            "upsample_nearest_exact3d_backward received an unsupported dtype"
        )
    if grad_output.device.type != runtime_device.name:
        raise RuntimeError("grad_output must be on the active accelerator backend")

    grad_output = grad_output.contiguous()

    if grad_input is None:
        grad_input = torch.empty(
            input_size, dtype=grad_output.dtype, device=grad_output.device
        )
    else:
        if grad_input.dtype != grad_output.dtype:
            raise RuntimeError(
                f"Expected out tensor to have dtype {grad_output.dtype}, "
                f"but got {grad_input.dtype} instead"
            )
        if grad_input.device != grad_output.device:
            raise RuntimeError(
                f"Expected out tensor to have device {grad_output.device}, "
                f"but got {grad_input.device} instead"
            )
        if tuple(grad_input.shape) != tuple(input_size):
            if grad_input.numel():
                warnings.warn(
                    "An output with one or more elements was resized since it "
                    "had a different shape from the required output shape.",
                    UserWarning,
                    stacklevel=3,
                )
            grad_input.resize_(input_size)

    total = n * c * id_ * ih * iw
    if total == 0:
        return grad_input

    d_bounds = _boundary_indices(id_, od, scales_d)
    h_bounds = _boundary_indices(ih, oh, scales_h)
    w_bounds = _boundary_indices(iw, ow, scales_w)
    identity = (
        _is_identity(d_bounds, id_, od)
        and _is_identity(h_bounds, ih, oh)
        and _is_identity(w_bounds, iw, ow)
    )
    db = _boundary_tensor(d_bounds, grad_output.device)
    hb = _boundary_tensor(h_bounds, grad_output.device)
    wb = _boundary_tensor(w_bounds, grad_output.device)

    int64_index = any(
        sum((size - 1) * stride for size, stride in zip(t.shape, t.stride()))
        > torch.iinfo(torch.int32).max
        for t in (grad_output, grad_input)
    )

    with torch_device_fn.device(grad_output.device):
        if int64_index:
            _upsample_nearest_exact3d_backward_scalar_kernel[(min(total, 65535),)](
                grad_output,
                grad_input,
                db,
                hb,
                wb,
                total,
                c,
                id_,
                ih,
                iw,
                *grad_output.stride(),
                *grad_input.stride(),
                identity,
            )
            return grad_input
        block = 512 if identity else 256
        _upsample_nearest_exact3d_backward_kernel[(triton.cdiv(total, block),)](
            grad_output,
            grad_input,
            db,
            hb,
            wb,
            total,
            c,
            id_,
            ih,
            iw,
            od,
            oh,
            ow,
            *grad_output.stride(),
            *grad_input.stride(),
            identity,
            block,
        )
    return grad_input


def _upsample_nearest_exact3d_backward(
    grad_output,
    output_size,
    input_size,
    scales_d=None,
    scales_h=None,
    scales_w=None,
):
    logger.debug("GEMS_KUNLUNXIN _UPSAMPLE_NEAREST_EXACT3D_BACKWARD")
    return _upsample_nearest_exact3d_backward_impl(
        grad_output, output_size, input_size, scales_d, scales_h, scales_w, None
    )


def _upsample_nearest_exact3d_backward_grad_input(
    grad_output,
    output_size,
    input_size,
    scales_d=None,
    scales_h=None,
    scales_w=None,
    grad_input=None,
):
    logger.debug("GEMS_KUNLUNXIN _UPSAMPLE_NEAREST_EXACT3D_BACKWARD.GRAD_INPUT")
    return _upsample_nearest_exact3d_backward_impl(
        grad_output,
        output_size,
        input_size,
        scales_d,
        scales_h,
        scales_w,
        grad_input,
    )

