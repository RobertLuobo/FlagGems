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

# The generic op's Python-level logic (argument parsing, output-shape checks,
# channels-last allocation, out-variant resize/quantizer handling) is correct
# on XPU; the Triton kernel and the native QuantizedCUDA metadata/copy readbacks
# (int_repr, as_strided, _copy_from, clone) are what need fixing.
_generic = importlib.import_module("flag_gems.ops.quantized_max_pool2d")


@libentry()
@triton.autotune(
    configs=runtime.get_tuned_config("quantized_max_pool2d"),
    key=["out_h", "out_w", "kernel_h", "kernel_w", "stride_h", "stride_w"],
)
@triton.jit
def quantized_max_pool2d_kernel(
    input_ptr,
    output_ptr,
    in_stride_n,
    in_stride_c,
    in_stride_h,
    in_stride_w,
    out_stride_n,
    out_stride_c,
    out_stride_h,
    out_stride_w,
    in_c,
    in_h,
    in_w,
    out_h,
    out_w,
    kernel_h: tl.constexpr,
    kernel_w: tl.constexpr,
    stride_h: tl.constexpr,
    stride_w: tl.constexpr,
    padding_h: tl.constexpr,
    padding_w: tl.constexpr,
    dilation_h: tl.constexpr,
    dilation_w: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_W: tl.constexpr,
):
    """Windowed max over the integer representation of a quantized tensor.

    Differs from the generic kernel only in how out-of-image positions are
    read: on the XPU backend a ``tl.load(..., mask=in_mask, other=min_val)``
    whose address is out of bounds still forms the address and returns a
    neighbour value instead of ``other``, which corrupts every window that
    overlaps the padding (any ``padding > 0`` case). The address is clamped to
    an in-image element with ``tl.where`` and loaded unconditionally, then the
    padding lanes are replaced with the dtype minimum via ``tl.where`` -- the
    same sentinel ATen uses and the pattern already proven in the vendor
    ``max_pool2d`` kernel.
    """
    pid_nc = tl.program_id(0)
    pid_hw = tl.program_id(1)
    num_w_blocks = tl.cdiv(out_w, BLOCK_W)
    h_block_idx = pid_hw // num_w_blocks
    w_block_idx = pid_hw % num_w_blocks
    n_idx = pid_nc // in_c
    c_idx = pid_nc % in_c

    h_out_offsets = h_block_idx * BLOCK_H + tl.arange(0, BLOCK_H)
    w_out_offsets = w_block_idx * BLOCK_W + tl.arange(0, BLOCK_W)

    dtype = input_ptr.type.element_ty
    min_val = get_dtype_min(dtype)
    max_val_acc = tl.full((BLOCK_H, BLOCK_W), min_val, dtype=dtype)

    input_base_ptr = input_ptr + n_idx * in_stride_n + c_idx * in_stride_c

    for kh in tl.static_range(0, kernel_h):
        for kw in tl.static_range(0, kernel_w):
            h_in = h_out_offsets[:, None] * stride_h - padding_h + kh * dilation_h
            w_in = w_out_offsets[None, :] * stride_w - padding_w + kw * dilation_w
            in_mask = (h_in >= 0) & (h_in < in_h) & (w_in >= 0) & (w_in < in_w)
            h_safe = tl.where(in_mask, h_in, 0)
            w_safe = tl.where(in_mask, w_in, 0)
            input_offset = h_safe * in_stride_h + w_safe * in_stride_w
            current_val = tl.load(input_base_ptr + input_offset)
            current_val = tl.where(in_mask, current_val, min_val)
            max_val_acc = tl.maximum(max_val_acc, current_val)

    out_base_ptr = output_ptr + n_idx * out_stride_n + c_idx * out_stride_c
    output_block_ptr = (
        out_base_ptr
        + h_out_offsets[:, None] * out_stride_h
        + w_out_offsets[None, :] * out_stride_w
    )
    out_mask = (h_out_offsets[:, None] < out_h) & (w_out_offsets[None, :] < out_w)
    tl.store(output_block_ptr, max_val_acc, mask=out_mask)


# Route the generic launcher (and the test's monkeypatch target) at the fixed
# kernel; everything else in the generic implementation is reused verbatim.
_generic.quantized_max_pool2d_kernel = quantized_max_pool2d_kernel

quantized_max_pool2d = _generic.quantized_max_pool2d
quantized_max_pool2d_out = _generic.quantized_max_pool2d_out


_QINT_TO_INT = {
    torch.quint8: torch.uint8,
    torch.qint8: torch.int8,
    torch.qint32: torch.int32,
}

_quantized_cuda_lib = None


def _int_view(qtensor):
    """Alias a per-tensor quantized tensor's storage as an integer tensor.

    The integer tensor shares storage, offset, sizes and strides with the
    quantized source, so reads and writes go straight to the quantized bytes
    without a device launch. ``int_repr`` returns a copy, but the view is what
    every bridge below needs to move the integers around.
    """
    return torch.empty(0, dtype=_QINT_TO_INT[qtensor.dtype], device=qtensor.device).set_(
        qtensor.untyped_storage(),
        qtensor.storage_offset(),
        tuple(qtensor.shape),
        qtensor.stride(),
    )


def _empty_quantized_like(src, memory_format):
    """Allocate a per-tensor quantized tensor matching ``src``'s quantizer."""
    if memory_format in (None, torch.preserve_format):
        if (
            src.dim() == 4
            and src.is_contiguous(memory_format=torch.channels_last)
            and not src.is_contiguous()
        ):
            mem_fmt = torch.channels_last
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
    """Zero-copy integer view of a per-tensor quantized tensor.

    ``aten::int_repr`` has no kernel in the XPU build (it raises "invalid
    device function"), so reinterpret the quantized storage as its underlying
    integer dtype -- preserving offset, sizes and strides -- which is exactly
    what int_repr returns, without a device launch.
    """
    return _int_view(self)


def _as_strided_impl(self, size, stride, storage_offset=None):
    """Strided view of a per-tensor quantized tensor.

    ``aten::as_strided`` rejects the quantized dtypes in the XPU build
    (``xdnn_pytorch_wrapper`` raises "scalar type ... is unsupported"), yet
    ``Tensor.transpose``/slicing and the device-to-host copy all go through it.
    The view only re-points metadata onto the same storage, so build it with
    ``set_`` instead of a device kernel.
    """
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
    """Copy a quantized tensor into ``dst`` through their integer views.

    The native quantized ``_copy_from`` is the "invalid device function" that
    breaks ``Tensor.cpu()`` on a channels-last result. Moving the integers over
    the integer views reaches the ordinary (non-quantized) copy, which handles
    arbitrary strides and the device-to-host direction, and leaves ``dst``'s
    quantizer untouched.
    """
    _int_view(dst).copy_(_int_view(self), non_blocking=non_blocking)
    return dst


def _clone_impl(self, *, memory_format=torch.preserve_format):
    """Clone a per-tensor quantized tensor, honouring ``memory_format``.

    ``Tensor.contiguous(memory_format=channels_last)`` on a quantized tensor
    clones through this op; the native QuantizedCUDA clone is unavailable, so
    allocate a fresh quantized buffer in the requested layout and move the
    integers through the integer views.
    """
    out = _empty_quantized_like(self, memory_format)
    _int_view(out).copy_(_int_view(self))
    return out


def _register_quantized_cuda_bridges():
    """Register the missing QuantizedCUDA metadata/copy ops used by the tests.

    All four are pure metadata or integer-copy bridges (no quantized math), so
    they stand in for the native kernels the XPU build does not provide without
    changing any numerical result.
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
