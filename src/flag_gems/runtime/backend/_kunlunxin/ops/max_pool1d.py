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

import importlib
import logging

import torch
import triton
import triton.language as tl

from flag_gems.utils import libentry
from flag_gems.utils.limits import get_dtype_min

logger = logging.getLogger(__name__)

_generic = importlib.import_module("flag_gems.ops.max_pool1d")
max_pool1d_output_size = _generic.max_pool1d_output_size
_parse_1d_param = _generic._parse_1d_param


@libentry()
@triton.autotune(
    configs=[
        triton.Config({"BLOCK": 256}, num_warps=4),
        triton.Config({"BLOCK": 512}, num_warps=4),
        triton.Config({"BLOCK": 1024}, num_warps=4),
        triton.Config({"BLOCK": 1024}, num_warps=8),
        triton.Config({"BLOCK": 2048}, num_warps=8),
    ],
    key=["out_l", "total", "kernel_size", "stride"],
)
@triton.jit
def max_pool1d_forward_kernel(
    input_ptr,
    output_ptr,
    in_l,
    out_l,
    total,  # N * C * out_l
    kernel_size: tl.constexpr,
    stride: tl.constexpr,
    padding: tl.constexpr,
    dilation: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    out_idx = pid * BLOCK + tl.arange(0, BLOCK)
    out_mask = out_idx < total

    nc_idx = out_idx // out_l
    ol_idx = out_idx % out_l

    in_base = nc_idx * in_l
    start = ol_idx * stride - padding

    max_val = tl.full((BLOCK,), get_dtype_min(input_ptr.dtype.element_ty), tl.float32)

    for k in tl.static_range(0, kernel_size):
        pos = start + k * dilation
        valid = out_mask & (pos >= 0) & (pos < in_l)
        # XPU3: masked tl.load with other=-inf is unreliable (KB entry 25 /
        # summary point 4) -- invalid lanes silently read the value at the
        # clamped address instead of taking `other`, contaminating the max.
        # Use the value-mask pattern: fully clamp the address to an always
        # in-bounds element, do an UNMASKED load, then neutralize invalid
        # lanes to -inf in registers before the reduction.
        offset = tl.where(valid, in_base + pos, 0)
        val = tl.load(input_ptr + offset)
        val = val.to(tl.float32)
        val = tl.where(valid, val, float("-inf"))
        max_val = tl.maximum(max_val, val)

    tl.store(
        output_ptr + out_idx, max_val.to(output_ptr.dtype.element_ty), mask=out_mask
    )


def max_pool1d(
    input: torch.Tensor,
    kernel_size,
    stride=None,
    padding=0,
    dilation=1,
    ceil_mode=False,
):
    """Max pooling over 1D input (Kunlunxin/XPU implementation).

    Mirrors the generic ``flag_gems.ops.max_pool1d`` wrapper but uses an
    XPU3-safe forward kernel: the generic kernel relies on a masked
    ``tl.load(..., other=-inf)`` whose invalid lanes are miscompiled on XPU3,
    so windows touching the (left/right) padding pick up ``input[..., 0]``
    instead of ``-inf`` and corrupt the max at the borders.
    """
    logger.debug("GEMS_KUNLUNXIN MAX_POOL1D")

    assert input.ndim in (2, 3), f"max_pool1d expects 2D or 3D input, got {input.ndim}D"

    kernel_w = _parse_1d_param(kernel_size, "kernel_size")
    stride_w = _parse_1d_param(stride, "stride", default=kernel_w)
    padding_w = _parse_1d_param(padding, "padding", default=0)
    dilation_w = _parse_1d_param(dilation, "dilation", default=1)

    if stride_w <= 0:
        raise ValueError(f"stride must be positive, but got stride={stride_w}")
    if padding_w < 0:
        raise ValueError(f"padding must be non-negative, but got padding={padding_w}")
    if dilation_w <= 0:
        raise ValueError(f"dilation must be positive, but got dilation={dilation_w}")

    input = input.contiguous()

    unbatched = input.ndim == 2
    x = input.unsqueeze(0) if unbatched else input  # (N, C, L)
    in_n, in_c, in_l = x.shape

    out_l = max_pool1d_output_size(
        in_l, kernel_w, stride_w, padding_w, dilation_w, ceil_mode
    )

    output = torch.empty((in_n, in_c, out_l), device=input.device, dtype=input.dtype)

    if output.numel() == 0:
        return output.squeeze(0) if unbatched else output

    total = in_n * in_c * out_l

    max_pool1d_forward_kernel[lambda meta: (triton.cdiv(total, meta["BLOCK"]),)](
        x,
        output,
        in_l,
        out_l,
        total,
        kernel_w,
        stride_w,
        padding_w,
        dilation_w,
    )

    if unbatched:
        output = output.squeeze(0)
    return output
