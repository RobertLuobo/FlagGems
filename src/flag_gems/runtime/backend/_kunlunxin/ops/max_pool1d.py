import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils.limits import get_dtype_min

logger = logging.getLogger(__name__)


def max_pool1d_output_size(
    in_size: int,
    kernel_size: int,
    stride: int,
    padding: int,
    dilation: int,
    ceil_mode: bool = False,
) -> int:
    effective_kernel_size = (kernel_size - 1) * dilation + 1
    numerator = in_size + 2 * padding - effective_kernel_size
    if ceil_mode:
        output_size = (numerator + stride - 1) // stride + 1
        if (output_size - 1) * stride >= in_size + padding:
            output_size -= 1
    else:
        output_size = numerator // stride + 1

    return output_size


@libentry()
@triton.jit
def max_pool1d_forward_flat_kernel(
    input_ptr,
    output_ptr,
    total: tl.constexpr,
    in_l: tl.constexpr,
    out_l: tl.constexpr,
    kernel_l: tl.constexpr,
    stride_l: tl.constexpr,
    padding_l: tl.constexpr,
    dilation_l: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    output_mask = offsets < total
    nc_idx = offsets // out_l
    ol = offsets % out_l
    nc_safe = tl.where(output_mask, nc_idx, 0)

    dtype = input_ptr.type.element_ty
    min_val = get_dtype_min(dtype)
    max_val = tl.full((BLOCK,), float("-inf"), tl.float32)

    for kl in tl.static_range(kernel_l):
        il = ol * stride_l - padding_l + kl * dilation_l
        valid = output_mask & (il >= 0) & (il < in_l)
        # Clamp the gather offset in-bounds and load unconditionally; XPU
        # masked out-of-bounds loads read neighbour memory, so the window
        # boundary is enforced on the VALUE via tl.where (padding = -inf).
        il_safe = tl.where(valid, il, 0)
        input_offset = nc_safe * in_l + il_safe
        value = tl.load(input_ptr + input_offset).to(tl.float32)
        value = tl.where(valid, value, float("-inf"))
        max_val = tl.maximum(max_val, value)

    tl.store(output_ptr + offsets, max_val.to(dtype), mask=output_mask)


def _parse_1d_param(param, name, default=None):
    if param is None or (isinstance(param, (list, tuple)) and len(param) == 0):
        return default
    if isinstance(param, int):
        return param
    if isinstance(param, (list, tuple)) and len(param) == 1:
        return param[0]
    raise ValueError(f"Invalid {name}: {param}")


def max_pool1d(
    input: torch.Tensor,
    kernel_size,
    stride=None,
    padding=0,
    dilation=1,
    ceil_mode=False,
):
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
    x = input.unsqueeze(0) if unbatched else input
    in_n, in_c, in_l = x.shape

    out_l = max_pool1d_output_size(
        in_l, kernel_w, stride_w, padding_w, dilation_w, ceil_mode
    )

    output = torch.empty((in_n, in_c, out_l), device=input.device, dtype=input.dtype)

    total = in_n * in_c * out_l
    if total != 0:
        block = 1024
        grid = (triton.cdiv(total, block),)
        with torch_device_fn.device(input.device):
            max_pool1d_forward_flat_kernel[grid](
                x,
                output,
                total,
                in_l,
                out_l,
                kernel_w,
                stride_w,
                padding_w,
                dilation_w,
                block,
                num_warps=1,
                buffer_size_limit=2048,
                isCloseVectorization=True,
            )

    if unbatched:
        output = output.squeeze(0)
    return output
