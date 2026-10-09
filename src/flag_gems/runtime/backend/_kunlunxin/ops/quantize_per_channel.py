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

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)


_QRANGE = {
    torch.quint8: (torch.uint8, 0, 255),
    torch.qint8: (torch.int8, -128, 127),
    torch.qint32: (torch.int32, -2147483648, 2147483647),
}


def _scalar_type_name(dtype):
    return {
        torch.float16: "Half",
        torch.bfloat16: "BFloat16",
        torch.float64: "Double",
        torch.float32: "Float",
        torch.int8: "Char",
        torch.uint8: "Byte",
        torch.int16: "Short",
        torch.int32: "Int",
        torch.int64: "Long",
        torch.bool: "Bool",
        torch.quint8: "QUInt8",
        torch.qint8: "QInt8",
        torch.qint32: "QInt32",
        torch.quint4x2: "QUInt4x2",
    }.get(dtype, str(dtype))


# XPU3 libdevice ``nearbyint``/``rint`` lowers to an unsupported extern symbol
# ("Unsupported"), which fails the ConvertTritonXPUToLLVM pass. Round-half-to-
# even is implemented here with the magic-number add/subtract identity, which is
# pure arithmetic:
#   r = (x + C) - C,  C = 1.5 * 2^(mantissa_bits)
# The fused add rounds under IEEE round-to-nearest-even, so ties fall to the
# even neighbour exactly like PyTorch's default rounding mode.
#   fp32: C = 1.5 * 2^23 = 12582912.0            (exact for |x| < 2^22)
#   fp64: C = 1.5 * 2^52 = 6755399441055744.0    (exact for |x| < 2^51)
# The integer scheme stays in fp64 to preserve PyTorch's qint32 accuracy; the
# float-qparams scheme stays in fp32 to match the native CUDA kernel.
@triton.jit
def _rne_f32(x):
    return (x + 12582912.0) - 12582912.0


@triton.jit
def _rne_f64(x):
    return (x + 6755399441055744.0) - 6755399441055744.0


@libentry()
@triton.jit
def quantize_per_channel_kernel(
    x_ptr,
    scales_ptr,
    zero_points_ptr,
    out_ptr,
    n_elements,
    stride_axis,
    shape_axis,
    q_min: tl.constexpr,
    q_max: tl.constexpr,
    USE_FLOAT_ZERO_POINT: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x_val = tl.load(x_ptr + offsets, mask=mask, other=0.0)

    if stride_axis == 1:
        axis_coord = offsets % shape_axis
    else:
        axis_coord = (offsets // stride_axis) % shape_axis

    scale = tl.load(scales_ptr + axis_coord, mask=mask, other=1.0)

    if USE_FLOAT_ZERO_POINT:
        zero_point_f = tl.load(zero_points_ptr + axis_coord, mask=mask, other=0.0)
        inv_scale = tl.math.div_rn(1.0, scale.to(tl.float32))
        q = _rne_f32(tl.math.fma(x_val.to(tl.float32), inv_scale, zero_point_f))
        q = q.to(tl.float64)
    else:
        zero_point_i64 = tl.load(zero_points_ptr + axis_coord, mask=mask, other=0)
        q = _rne_f64(x_val.to(tl.float64) / scale.to(tl.float64))
        q = q + zero_point_i64.to(tl.float64)

    # ``tl.clamp`` lowers to ``tt.clampf`` which XPU3 cannot legalize for fp64,
    # so clamp with explicit minimum/maximum (arith.minnumf/maxnumf) instead.
    q = tl.minimum(tl.maximum(q, q_min * 1.0), q_max * 1.0)
    q = q.to(out_ptr.dtype.element_ty)

    tl.store(out_ptr + offsets, q, mask=mask)


def _is_float_qparams(scales, zero_points):
    return zero_points.dtype.is_floating_point


def _validate_qparams(input, scales, zero_points, axis, dtype):
    if dtype not in _QRANGE:
        raise NotImplementedError(
            f'"quantize_tensor_per_channel_affine" not implemented for '
            f"'{_scalar_type_name(dtype)}'"
        )

    for name, param in (("scales", scales), ("zero_points", zero_points)):
        if param.device != input.device:
            raise RuntimeError(
                f"Expected all tensors to be on the same device, but got {name} is on "
                f"{param.device}, different from other tensors on {input.device} "
                "(when checking argument in method wrapper_CUDA__quantize_per_channel)"
            )

    if input.dtype != torch.float32:
        got = _scalar_type_name(input.dtype)
        if _is_float_qparams(scales, zero_points):
            raise RuntimeError(f"Quantize only works on Float Tensor, got {got}")
        raise RuntimeError(
            f"quantize_tensor_per_channel_affine expects a Float Tensor, got {got}"
        )

    if scales.dim() != 1:
        raise RuntimeError("scale tensor must have dimension 1")
    if zero_points.dim() != 1:
        raise RuntimeError("zero_points tensor must have dimension 1")
    if scales.numel() != zero_points.numel():
        raise RuntimeError("number of elements in scales and zero_points must match")

    if not scales.dtype.is_floating_point:
        raise RuntimeError("scale tensor must be floating point")

    if not (0 <= axis < input.dim()):
        scheme = "float qparams" if _is_float_qparams(scales, zero_points) else "affine"
        raise RuntimeError(
            f"Channel axis out of range in per channel {scheme} quantization. "
            f"Got: {axis}Expected: [0, {input.dim()})"
        )

    n_channels = input.size(axis)
    if scales.numel() != n_channels:
        raise RuntimeError(
            f"length of scales must equal to channel, expected {n_channels} got, "
            f"{scales.numel()}"
        )
    if zero_points.numel() != n_channels:
        raise RuntimeError(
            f"length of zero_points must equal to channel expected {n_channels} got, "
            f"{zero_points.numel()}"
        )

    return _is_float_qparams(scales, zero_points)


def _check_zero_points_on_host(zero_points, q_min, q_max, use_float_zero_point):
    scheme = "float_qparams" if use_float_zero_point else "affine"
    prefix = f"quantize_tensor_per_channel_{scheme}_cuda"
    if zero_points.numel() == 0:
        return
    zp_min = zero_points.min().item()
    if zp_min < q_min:
        raise RuntimeError(f"{prefix}zero_point is below lower bound.")
    zp_max = zero_points.max().item()
    if zp_max > q_max:
        raise RuntimeError(f"{prefix}zero_point is above upper bound.")


def _quantize_per_channel_impl(input, scales, zero_points, axis, dtype):
    use_float_zero_point = _validate_qparams(input, scales, zero_points, axis, dtype)
    int_dtype, q_min, q_max = _QRANGE[dtype]

    input = input.contiguous()
    _check_zero_points_on_host(zero_points, q_min, q_max, use_float_zero_point)
    if use_float_zero_point:
        scales_k = scales.to(torch.float32).contiguous()
        zero_points_k = zero_points.to(torch.float32).contiguous()
    else:
        scales_k = scales.to(torch.float64).contiguous()
        zero_points_k = zero_points.to(torch.int64).contiguous()

    shape_axis = input.shape[axis]
    stride_axis = input.stride(axis)

    int_out = torch.empty_strided(
        input.shape,
        input.stride(),
        dtype=int_dtype,
        device=input.device,
    )

    n_elements = input.numel()
    if n_elements > 0:
        BLOCK_SIZE = 1024
        grid = (triton.cdiv(n_elements, BLOCK_SIZE),)

        with torch_device_fn.device(input.device):
            quantize_per_channel_kernel[grid](
                input,
                scales_k,
                zero_points_k,
                int_out,
                n_elements,
                stride_axis,
                shape_axis,
                q_min,
                q_max,
                USE_FLOAT_ZERO_POINT=use_float_zero_point,
                BLOCK_SIZE=BLOCK_SIZE,
            )

    return torch._make_per_channel_quantized_tensor(
        int_out, scales_k, zero_points_k, axis
    )


def quantize_per_channel(input, scales, zero_points, axis, dtype):
    logger.debug("GEMS QUANTIZE_PER_CHANNEL")
    return _quantize_per_channel_impl(input, scales, zero_points, axis, dtype)


def quantize_per_channel_out(input, scales, zero_points, axis, dtype, *, out=None):
    logger.debug("GEMS QUANTIZE_PER_CHANNEL_OUT")
    if out is None:
        return _quantize_per_channel_impl(input, scales, zero_points, axis, dtype)
    return _quantize_per_channel_out_impl(input, scales, zero_points, axis, dtype, out)


def _quantize_per_channel_out_impl(input, scales, zero_points, axis, dtype, out):
    if not out.is_quantized or out.dtype != dtype:
        raise RuntimeError(
            f"Expected out tensor to have dtype {dtype}, but got {out.dtype} instead"
        )
    float_result = _is_float_qparams(scales, zero_points)
    float_out = out.qscheme() == torch.per_channel_affine_float_qparams
    if float_result != float_out:
        raise RuntimeError("Quantized Copy only works with same qscheme")

    if tuple(out.shape) != tuple(input.shape):
        raise RuntimeError(
            f"The size of tensor a {tuple(out.shape)} must match the size of "
            f"tensor b {tuple(input.shape)} at non-singleton dimension 0"
        )

    result = _quantize_per_channel_impl(input, scales, zero_points, axis, dtype)
    out.copy_(result)
    return out
