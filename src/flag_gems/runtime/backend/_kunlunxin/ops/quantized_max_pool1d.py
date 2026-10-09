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

logger = logging.getLogger(__name__)

# The generic op's Python-level logic (argument parsing, output-shape checks,
# out-variant resize/quantizer handling, non-contiguous gathering) is correct
# on XPU; the Triton kernel and the native QuantizedCUDA metadata/copy readbacks
# (int_repr, as_strided, _copy_from, clone) are what need fixing.
_generic = importlib.import_module("flag_gems.ops.quantized_max_pool1d")


@libentry()
@triton.autotune(
    configs=runtime.get_tuned_config("quantized_max_pool1d"),
    key=["out_l", "kernel_size", "stride", "padding", "dilation"],
)
@triton.jit
def quantized_max_pool1d_forward_kernel(
    in_ptr,
    out_ptr,
    n,
    c,
    in_l,
    out_l,
    in_stride_n,
    in_stride_c,
    in_stride_l,
    out_stride_n,
    out_stride_c,
    out_stride_l,
    kernel_size: tl.constexpr,
    stride: tl.constexpr,
    padding: tl.constexpr,
    dilation: tl.constexpr,
    neutral: tl.constexpr,
    BLOCK_L: tl.constexpr,
):
    """Windowed max over the integer representation of a quantized tensor.

    Differs from the generic kernel only in how out-of-range positions are
    read: on the XPU backend a ``tl.load(..., mask=in_mask, other=neutral)``
    whose address is out of bounds still forms the address and returns a
    neighbour value instead of ``other``, corrupting every window that overlaps
    the padding (any ``padding > 0`` case). The length index is clamped to an
    in-range element with ``tl.where`` and loaded unconditionally, then the
    padding lanes are replaced with the dtype minimum via ``tl.where`` -- the
    same sentinel ATen uses and the pattern already proven in the vendor
    ``quantized_max_pool2d`` kernel.
    """
    pid = tl.program_id(0)
    num_l_blocks = tl.cdiv(out_l, BLOCK_L)
    nc_idx = pid // num_l_blocks
    l_block_idx = pid % num_l_blocks
    n_idx = nc_idx // c
    c_idx = nc_idx % c

    l_out_offsets = l_block_idx * BLOCK_L + tl.arange(0, BLOCK_L)
    out_mask = l_out_offsets < out_l

    max_acc = tl.full((BLOCK_L,), neutral, dtype=in_ptr.type.element_ty)

    in_base = n_idx * in_stride_n + c_idx * in_stride_c
    for k in tl.static_range(0, kernel_size):
        l_in = l_out_offsets * stride - padding + k * dilation
        in_mask = (l_in >= 0) & (l_in < in_l) & out_mask
        l_safe = tl.where(in_mask, l_in, 0)
        current = tl.load(in_ptr + in_base + l_safe * in_stride_l)
        current = tl.where(in_mask, current, neutral)
        max_acc = tl.maximum(max_acc, current)

    out_base = n_idx * out_stride_n + c_idx * out_stride_c
    tl.store(out_ptr + out_base + l_out_offsets * out_stride_l, max_acc, mask=out_mask)


# Route the generic launcher at the fixed kernel; the rest of the generic
# implementation (planning, out-variant handling, int views) is reused verbatim.
_generic.quantized_max_pool1d_forward_kernel = quantized_max_pool1d_forward_kernel


def _sync_out_quantizer(out, scale, zero_point):
    """Re-point ``out`` at the input's per-tensor quantizer, in place.

    The generic op moves the quantizer with ``out.copy_(alias)``, relying on
    the native quantized ``copy_`` to carry scale/zero-point across. On XPU the
    quantized ``_copy_from`` is the "invalid device function" bridged below to
    copy integers only, so it no longer moves the quantizer. Swap it the way
    the vendor ``quantized_max_pool2d`` overlay does instead: assign an alias
    that shares ``out``'s storage, offset, sizes and strides but carries the
    target quantizer to ``out.data`` -- the values are written separately by
    the kernel, so only the quantizer has to move and the caller's ``out``
    object, storage and layout are left intact.
    """
    if out.q_scale() == scale and out.q_zero_point() == zero_point:
        return
    alias = torch._empty_affine_quantized(
        0, scale=scale, zero_point=zero_point, dtype=out.dtype, device=out.device
    )
    alias.set_(
        out.untyped_storage(),
        out.storage_offset(),
        tuple(out.shape),
        out.stride(),
    )
    out.data = alias


_generic._sync_out_quantizer = _sync_out_quantizer

quantized_max_pool1d = _generic.quantized_max_pool1d
quantized_max_pool1d_out = _generic.quantized_max_pool1d_out


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
    without a device launch.
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

    ``aten::as_strided`` rejects the quantized dtypes in the XPU build, yet
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
    breaks ``Tensor.cpu()``. Moving the integers over the integer views reaches
    the ordinary (non-quantized) copy, which handles arbitrary strides and the
    device-to-host direction, and leaves ``dst``'s quantizer untouched.
    """
    _int_view(dst).copy_(_int_view(self), non_blocking=non_blocking)
    return dst


def _clone_impl(self, *, memory_format=torch.preserve_format):
    """Clone a per-tensor quantized tensor, honouring ``memory_format``.

    The native QuantizedCUDA clone is unavailable, so allocate a fresh
    quantized buffer in the requested layout and move the integers through the
    integer views.
    """
    out = _empty_quantized_like(self, memory_format)
    _int_view(out).copy_(_int_view(self))
    return out


def _register_quantized_cuda_bridges():
    """Register the missing QuantizedCUDA metadata/copy ops used by the tests.

    All four are pure metadata or integer-copy bridges (no quantized math), so
    they stand in for the native kernels the XPU build does not provide without
    changing any numerical result. ``allow_override=True`` keeps them
    compatible with the identical bridges the ``quantized_max_pool2d`` overlay
    registers.
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
