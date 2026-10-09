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
import importlib
import logging

import torch
import triton
import triton.language as tl

from flag_gems import runtime
from flag_gems.utils import libentry
from flag_gems.utils.limits import get_dtype_min

logger = logging.getLogger(__name__)

_generic = importlib.import_module("flag_gems.ops.quantized_max_pool3d")


@libentry()
@triton.autotune(
    configs=runtime.get_tuned_config("quantized_max_pool3d"),
    key=[
        "out_d",
        "out_h",
        "out_w",
        "kernel_d",
        "kernel_h",
        "kernel_w",
        "stride_d",
        "stride_h",
        "stride_w",
    ],
)
@triton.jit
def quantized_max_pool3d_forward_kernel(
    input_ptr,
    output_ptr,
    in_stride_n,
    in_stride_c,
    in_stride_d,
    in_stride_h,
    in_stride_w,
    out_stride_n,
    out_stride_c,
    out_stride_d,
    out_stride_h,
    out_stride_w,
    in_c,
    in_d,
    in_h,
    in_w,
    out_d,
    out_h,
    out_w,
    kernel_d: tl.constexpr,
    kernel_h: tl.constexpr,
    kernel_w: tl.constexpr,
    stride_d: tl.constexpr,
    stride_h: tl.constexpr,
    stride_w: tl.constexpr,
    padding_d: tl.constexpr,
    padding_h: tl.constexpr,
    padding_w: tl.constexpr,
    dilation_d: tl.constexpr,
    dilation_h: tl.constexpr,
    dilation_w: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_W: tl.constexpr,
):
    """Windowed max over the integer representation of a quantized tensor.

    Differs from the generic 3-D kernel only in how out-of-image positions are
    read: on the XPU backend a ``tl.load(..., mask=in_mask, other=min_val)``
    whose address is out of bounds still forms the address and returns a
    neighbour value instead of ``other``, which corrupts every window that
    overlaps the padding (any ``padding > 0`` case). The address is clamped to
    an in-image element with ``tl.where`` and loaded unconditionally, then the
    padding lanes are replaced with the dtype minimum via ``tl.where`` -- the
    same sentinel ATen uses and the pattern already proven in the vendor
    ``max_pool2d``/``quantized_max_pool2d`` kernels.
    """
    pid_nc = tl.program_id(0)
    pid_dhw = tl.program_id(1)

    num_h_blocks = tl.cdiv(out_h, BLOCK_H)
    num_w_blocks = tl.cdiv(out_w, BLOCK_W)

    d_block_idx = pid_dhw // (num_h_blocks * num_w_blocks)
    hw_remainder = pid_dhw % (num_h_blocks * num_w_blocks)
    h_block_idx = hw_remainder // num_w_blocks
    w_block_idx = hw_remainder % num_w_blocks

    n_idx = pid_nc // in_c
    c_idx = pid_nc % in_c

    d_out = d_block_idx

    h_out_offsets = h_block_idx * BLOCK_H + tl.arange(0, BLOCK_H)
    w_out_offsets = w_block_idx * BLOCK_W + tl.arange(0, BLOCK_W)

    dtype = input_ptr.type.element_ty
    min_val = get_dtype_min(dtype)
    max_val_acc = tl.full((BLOCK_H, BLOCK_W), min_val, dtype=dtype)

    input_base_ptr = input_ptr + n_idx * in_stride_n + c_idx * in_stride_c

    for kd in tl.static_range(0, kernel_d):
        d_in = d_out * stride_d - padding_d + kd * dilation_d
        d_valid = (d_in >= 0) & (d_in < in_d)
        d_safe = tl.where(d_valid, d_in, 0)
        for kh in tl.static_range(0, kernel_h):
            for kw in tl.static_range(0, kernel_w):
                h_in = h_out_offsets[:, None] * stride_h - padding_h + kh * dilation_h
                w_in = w_out_offsets[None, :] * stride_w - padding_w + kw * dilation_w
                in_mask = (
                    d_valid & (h_in >= 0) & (h_in < in_h) & (w_in >= 0) & (w_in < in_w)
                )
                h_safe = tl.where(in_mask, h_in, 0)
                w_safe = tl.where(in_mask, w_in, 0)
                input_offset = (
                    d_safe * in_stride_d
                    + h_safe * in_stride_h
                    + w_safe * in_stride_w
                )
                current_val = tl.load(input_base_ptr + input_offset)
                current_val = tl.where(in_mask, current_val, min_val)
                max_val_acc = tl.maximum(max_val_acc, current_val)

    out_base_ptr = (
        output_ptr + n_idx * out_stride_n + c_idx * out_stride_c + d_out * out_stride_d
    )
    output_block_ptr = (
        out_base_ptr
        + h_out_offsets[:, None] * out_stride_h
        + w_out_offsets[None, :] * out_stride_w
    )

    out_mask = (h_out_offsets[:, None] < out_h) & (w_out_offsets[None, :] < out_w)
    tl.store(output_block_ptr, max_val_acc, mask=out_mask)


_generic.quantized_max_pool3d_forward_kernel = quantized_max_pool3d_forward_kernel

quantized_max_pool3d = _generic.quantized_max_pool3d


_QINT_TO_INT = {
    torch.quint8: torch.uint8,
    torch.qint8: torch.int8,
    torch.qint32: torch.int32,
}

_quantized_cuda_lib = None


def _int_view(qtensor):
    """Alias a per-tensor quantized tensor's storage as an integer tensor."""
    return torch.empty(0, dtype=_QINT_TO_INT[qtensor.dtype], device=qtensor.device).set_(
        qtensor.untyped_storage(),
        qtensor.storage_offset(),
        tuple(qtensor.shape),
        qtensor.stride(),
    )


def _write_out(out, result):
    """Write ``result`` into ``out`` and make ``out`` adopt its quantizer.

    The generic ``out`` variant finishes with ``out.copy_(result)``. On the XPU
    backend a 5-D quantized ``copy_`` whose source and destination carry
    different quantizers takes a requantising path that calls an unsupported
    ``as_strided`` view-update (``scalar type ... kquint8 is unsupported``) and
    aborts. Reproduce ATen's two observable effects without that kernel: copy
    the integer representation into ``out``'s own storage (strided-safe, so a
    non-contiguous ``out`` keeps its layout), then re-point ``out`` at a
    quantized view of that same storage carrying ``result``'s scale/zero_point.
    ``Tensor.data =`` swaps the quantizer in place while preserving ``out``'s
    storage, sizes, strides, storage offset and Python identity.
    """
    _int_view(out).copy_(_int_view(result))
    requantized = torch._empty_affine_quantized(
        0,
        scale=float(result.q_scale()),
        zero_point=int(result.q_zero_point()),
        dtype=out.dtype,
        device=out.device,
    )
    requantized.set_(
        out.untyped_storage(),
        out.storage_offset(),
        tuple(out.shape),
        out.stride(),
    )
    out.data = requantized


def quantized_max_pool3d_out(
    input: torch.Tensor,
    kernel_size,
    stride=(),
    padding=0,
    dilation=1,
    ceil_mode=False,
    *,
    out: torch.Tensor,
):
    """``out`` variant of :func:`quantized_max_pool3d` for the XPU backend.

    Mirrors the generic contract (validate parameters, then ``out``'s dtype and
    device, then resize ``out`` to the pooled shape, then write the pooled
    values and adopt the input's quantizer) but finishes through
    :func:`_write_out` instead of the generic ``out.copy_(result)``, which is
    broken for 5-D quantized tensors on this backend.
    """
    logger.debug("GEMS_KUNLUNXIN QUANTIZED_MAX_POOL3D_OUT")

    params = _generic._parse_pool3d_params(kernel_size, stride, padding, dilation)
    _generic._check_quantized_input(input)
    out_shape = _generic._quantized_max_pool3d_output_shape(input, params, ceil_mode)

    if out.dtype != input.dtype:
        raise RuntimeError(
            "Expected out tensor to have dtype "
            f"{_generic._aten_dtype_name(input.dtype)}, "
            f"but got {_generic._aten_dtype_name(out.dtype)} instead"
        )
    if out.device != input.device:
        raise RuntimeError(
            f"Expected out tensor to have device {input.device}, "
            f"but got {out.device} instead"
        )

    if tuple(out.shape) != tuple(out_shape):
        _generic._resize_quantized_out(out, out_shape)

    result = quantized_max_pool3d(
        input,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        ceil_mode=ceil_mode,
    )
    _write_out(out, result)
    return out


def _empty_quantized_like(src, memory_format):
    """Allocate a per-tensor quantized tensor matching ``src``'s quantizer.

    Handles both the 4-D ``channels_last`` and 5-D ``channels_last_3d`` cases so
    the single registration can serve the 2-D and 3-D quantized max-pool tests.
    """
    if memory_format in (None, torch.preserve_format):
        if (
            src.dim() == 4
            and src.is_contiguous(memory_format=torch.channels_last)
            and not src.is_contiguous()
        ):
            mem_fmt = torch.channels_last
        elif (
            src.dim() == 5
            and src.is_contiguous(memory_format=torch.channels_last_3d)
            and not src.is_contiguous()
        ):
            mem_fmt = torch.channels_last_3d
        else:
            mem_fmt = torch.contiguous_format
    else:
        mem_fmt = memory_format
    return torch._empty_affine_quantized(
        tuple(src.shape),
        scale=float(src.q_scale()),
        zero_point=int(src.q_zero_point()),
        dtype=src.dtype,
        device=src.device,
        memory_format=mem_fmt,
    )


def _int_repr_impl(self):
    """Zero-copy integer view of a per-tensor quantized tensor."""
    return _int_view(self)


def _as_strided_impl(self, size, stride, storage_offset=None):
    """Strided view of a per-tensor quantized tensor."""
    offset = self.storage_offset() if storage_offset is None else storage_offset
    out = torch._empty_affine_quantized(
        0,
        scale=float(self.q_scale()),
        zero_point=int(self.q_zero_point()),
        dtype=self.dtype,
        device=self.device,
    )
    out.set_(self.untyped_storage(), offset, tuple(size), tuple(stride))
    return out


def _copy_from_impl(self, dst, non_blocking=False):
    """Copy a quantized tensor into ``dst`` through their integer views."""
    _int_view(dst).copy_(_int_view(self), non_blocking=non_blocking)
    return dst


def _clone_impl(self, *, memory_format=torch.preserve_format):
    """Clone a per-tensor quantized tensor, honouring ``memory_format``."""
    out = _empty_quantized_like(self, memory_format)
    _int_view(out).copy_(_int_view(self))
    return out


def _register_quantized_cuda_bridges():
    """Register the missing QuantizedCUDA metadata/copy ops used by the tests.

    All four are pure metadata or integer-copy bridges (no quantized math), so
    they stand in for the native kernels the XPU build does not provide without
    changing any numerical result. The impls here handle both the 4-D and 5-D
    channels-last layouts, so they remain correct for the 2-D overlay too.
    """
    global _quantized_cuda_lib
    if _quantized_cuda_lib is not None:
        return
    _quantized_cuda_lib = torch.library.Library("aten", "IMPL")
    _quantized_cuda_lib.impl(
        "int_repr", _int_repr_impl, "QuantizedCUDA", allow_override=True
    )
    _quantized_cuda_lib.impl(
        "as_strided", _as_strided_impl, "QuantizedCUDA", allow_override=True
    )
    _quantized_cuda_lib.impl(
        "_copy_from", _copy_from_impl, "QuantizedCUDA", allow_override=True
    )
    _quantized_cuda_lib.impl(
        "clone", _clone_impl, "QuantizedCUDA", allow_override=True
    )


_register_quantized_cuda_bridges()
