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
from flag_gems.utils import dim_compress
from flag_gems.utils import triton_lang_extension as tle

logger = logging.getLogger(__name__)


@triton.jit
def trapezoid_kernel_dx(
    y_ptr,
    out_ptr,
    M,
    N,
    dx: tl.float64,
    M_STRIDE_Y,
    M_STRIDE_OUT,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Trapezoid integration with constant spacing dx (XPU-friendly).

    Reformulated as ``dx * (sum(y) - 0.5 * (y[0] + y[N-1]))`` so the reduction
    axis is read as a single unmasked power-of-two tile (the caller zero-pads a
    non-power-of-two N into a BLOCK_N-wide buffer).  This avoids both the 1D
    store after a 2D reduce (which the XPU core-tiling pass cannot assign a
    consistent encoding to) and masked loads inside the reduction (which
    miscompile on KL3 for non-power-of-two or looped tiles).
    """
    acc_dtype = tl.float64 if y_ptr.type.element_ty == tl.float64 else tl.float32
    pid_m = tle.program_id(0)

    m_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = m_offsets < M

    n_offsets = tl.arange(0, BLOCK_N)
    y_offsets = m_offsets[:, None] * M_STRIDE_Y + n_offsets[None, :]

    # Columns are fully in-bounds (the buffer is BLOCK_N wide); only the row
    # dimension needs masking, so the reduction tile stays unmasked along N.
    y = tl.load(y_ptr + y_offsets, mask=mask_m[:, None], other=0.0).to(acc_dtype)
    total = tl.sum(y, axis=1, keep_dims=True)

    first = tl.load(
        y_ptr + m_offsets[:, None] * M_STRIDE_Y, mask=mask_m[:, None], other=0.0
    ).to(acc_dtype)
    last = tl.load(
        y_ptr + m_offsets[:, None] * M_STRIDE_Y + (N - 1),
        mask=mask_m[:, None],
        other=0.0,
    ).to(acc_dtype)

    acc = total - 0.5 * (first + last)
    result = acc.to(tl.float64) * dx

    out_offsets = m_offsets[:, None] * M_STRIDE_OUT + tl.arange(0, 1)[None, :]
    tl.store(out_ptr + out_offsets, result.to(out_ptr.type.element_ty), mask=mask_m[:, None])


def _normalize_dim(dim, ndim):
    if ndim == 0:
        raise IndexError(f"Dimension specified as {dim} but tensor has no dimensions")
    if not (-ndim <= dim < ndim):
        raise IndexError(
            f"Dimension out of range (expected to be in range of "
            f"[{-ndim}, {ndim - 1}], but got {dim})"
        )
    return dim % ndim


class TrapzOp(torch.autograd.Function):
    """Custom autograd function for the trapezoidal rule."""

    @staticmethod
    def forward(ctx, y, dx, dim):
        logger.debug("GEMS_KUNLUNXIN TRAPEZOID_DX")
        logger.debug("GEMS TRAPEZOID_DX")

        dim = _normalize_dim(dim, y.ndim)
        shape = y.shape
        N = shape[dim]
        out_shape = shape[:dim] + shape[dim + 1 :]

        ctx.dx = dx
        ctx.dim = dim
        ctx.N = N
        ctx.y_shape = shape
        ctx.y_dtype = y.dtype

        # Empty tensors or a degenerate reduction dimension (N <= 1) integrate
        # to zero; the reduction dimension is always removed from the shape.
        if y.numel() == 0 or N <= 1:
            return torch.zeros(out_shape, dtype=y.dtype, device=y.device)

        y_compressed = dim_compress(y, dim)
        M = y_compressed.numel() // N

        # Pad the reduction axis to a power of two with zeros so the kernel can
        # read it as a single unmasked tile (zeros do not change the sum, and
        # the first/last endpoints are read from the original [0, N-1] slots).
        block_n = triton.next_power_of_2(N)
        if block_n != N:
            padded = torch.zeros(
                [M, block_n], dtype=y_compressed.dtype, device=y_compressed.device
            )
            padded[:, :N] = y_compressed.reshape(M, N)
            y_compressed = padded
            m_stride_y = block_n
        else:
            m_stride_y = N

        compute_dtype = torch.float64 if y.dtype == torch.float64 else torch.float32
        output = torch.empty([M, 1], dtype=compute_dtype, device=y.device)

        # Keep the reduction tile within the exact-sum budget (<= 8192 lanes on
        # KL3); wider rows take fewer rows per program.
        block_m = max(1, min(32, 4096 // block_n))
        grid = (triton.cdiv(M, block_m),)
        with torch_device_fn.device(y.device):
            trapezoid_kernel_dx[grid](
                y_compressed,
                output,
                M,
                N,
                dx,
                m_stride_y,
                1,
                BLOCK_M=block_m,
                BLOCK_N=block_n,
            )

        result = output.reshape(out_shape)
        if compute_dtype != y.dtype:
            result = result.to(y.dtype)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        dx = ctx.dx
        dim = ctx.dim
        N = ctx.N
        y_shape = ctx.y_shape
        y_dtype = ctx.y_dtype

        if N <= 1:
            return (
                torch.zeros(y_shape, dtype=y_dtype, device=grad_output.device),
                None,
                None,
            )

        acc_dtype = torch.float64 if y_dtype == torch.float64 else torch.float32
        w = torch.ones(N, dtype=acc_dtype, device=grad_output.device)
        w[0] = 0.5
        w[-1] = 0.5
        w_shape = [1] * len(y_shape)
        w_shape[dim] = N
        w = w.view(w_shape)

        grad_y = grad_output.to(acc_dtype).unsqueeze(dim) * w * dx
        return grad_y.to(y_dtype), None, None


def trapz(y, dx=1.0, dim=-1):
    """Compute the trapezoidal rule along a dimension with constant spacing."""
    if y.dtype == torch.bool:
        raise RuntimeError(
            "trapezoid: received a bool input for `y`, but bool is not supported"
        )

    if y.is_complex():
        raise RuntimeError(
            "trapz: complex inputs are not supported by the FlagGems Triton kernel"
        )

    if not y.is_floating_point():
        y = y.to(torch.float32)

    if isinstance(dx, torch.Tensor):
        dx = dx.item()
    else:
        dx = float(dx)

    return TrapzOp.apply(y, dx, dim)
