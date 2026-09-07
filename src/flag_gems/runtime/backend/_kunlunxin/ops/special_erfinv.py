"""Kunlunxin special_erfinv / special_erfinv_out vendor override.

Reuses the erfinv.py log-form body (`_erfinv_body`, see its module docstring
for the math and XPU rationale): erfinv(x) = sgn(x)*sqrt(w)*H(w) with
w = -log(1 - x^2) (series+min/max-ramp blend, no selects) and a degree-4 fit
of H (fp32 max err ~1.8e-6 on |x| <= 0.99, atol 1e-4).  This replaces the
previous deg-8 Horner in x^2 (only accurate on |x| <= 0.9) and the two
selects, and is ~2x faster.  Edge semantics fall out of the arithmetic:
|x| > 1 -> NaN, |x| == 1 -> sign(x)*inf, x == +-0 -> +-0 (see erfinv.py).
"""

import logging

import torch
import triton
import triton.language as tl

from .erfinv import _erfinv_body, _pick_block

logger = logging.getLogger(__name__)

UNROLL_NUM = 8
BUFFER_SIZE_LIMIT = 8192
IS_CLOSE_MEMORY_ASYNC = False


@triton.jit
def _special_erfinv_kernel(x_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = _erfinv_body(x.to(tl.float32)).to(x.dtype)
    tl.store(out_ptr + offsets, y, mask=mask)


@triton.jit
def _special_erfinv_kernel_unmasked(x_ptr, out_ptr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    x = tl.load(x_ptr + offsets)
    y = _erfinv_body(x.to(tl.float32)).to(x.dtype)
    tl.store(out_ptr + offsets, y)


def _launch_special_erfinv_kernel(x: torch.Tensor, out: torch.Tensor):
    assert x.device == out.device, "Input and output must be on the same device"
    assert (
        x.numel() == out.numel()
    ), "Input and output must have the same number of elements"
    assert x.dtype == out.dtype, "Input and output must have the same dtype"
    n_elements = x.numel()
    if n_elements == 0:
        return
    block_size, num_warps, masked = _pick_block(n_elements)
    if masked:
        grid = (triton.cdiv(n_elements, block_size),)
        _special_erfinv_kernel[grid](
            x,
            out,
            n_elements,
            BLOCK_SIZE=block_size,
            num_warps=num_warps,
            unroll_num=UNROLL_NUM,
            buffer_size_limit=BUFFER_SIZE_LIMIT,
            isCloseMemoryAsync=IS_CLOSE_MEMORY_ASYNC,
        )
    else:
        grid = (n_elements // block_size,)
        _special_erfinv_kernel_unmasked[grid](
            x,
            out,
            BLOCK_SIZE=block_size,
            num_warps=num_warps,
            unroll_num=UNROLL_NUM,
            buffer_size_limit=BUFFER_SIZE_LIMIT,
            isCloseMemoryAsync=IS_CLOSE_MEMORY_ASYNC,
        )


def special_erfinv(x: torch.Tensor):
    """Special erfinv function"""
    logger.debug("GEMS KUNLUNXIN special_erfinv")
    x_in = x if x.is_contiguous() else x.contiguous()
    out = torch.empty_like(x_in)
    _launch_special_erfinv_kernel(x_in, out)
    return out


def special_erfinv_out(x: torch.Tensor, out: torch.Tensor):
    """Special erfinv out function"""
    logger.debug("GEMS KUNLUNXIN special_erfinv_out")
    # Resize out to match input shape if necessary
    if out.shape != x.shape:
        out.resize_(x.shape)
    # Ensure dtype matches input dtype for aten out semantics
    assert out.dtype == x.dtype, "out tensor must have the same dtype as input"
    x_in = x if x.is_contiguous() else x.contiguous()
    if out.is_contiguous():
        _launch_special_erfinv_kernel(x_in, out)
        return out
    else:
        tmp = torch.empty_like(out, memory_format=torch.contiguous_format)
        _launch_special_erfinv_kernel(x_in, tmp)
        out.copy_(tmp)
        return out


def special_erfinv_(x: torch.Tensor):
    """Special erfinv_ in-place function"""
    logger.debug("GEMS KUNLUNXIN special_erfinv_")
    original_shape = x.shape
    original_stride = x.stride()
    x_in = x if x.is_contiguous() else x.contiguous()
    tmp = torch.empty_like(x_in)
    _launch_special_erfinv_kernel(x_in, tmp)
    x.copy_(tmp)
    # Restore original shape and stride if needed
    if x.shape != original_shape or x.stride() != original_stride:
        x = x.reshape(original_shape).as_strided(original_shape, original_stride)
    return x