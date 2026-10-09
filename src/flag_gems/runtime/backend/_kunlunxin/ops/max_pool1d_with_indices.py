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
    indices_ptr,
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
    max_val = tl.full((BLOCK,), min_val, dtype=dtype)
    max_idx = tl.full((BLOCK,), -1, tl.int64)

    for kl in tl.static_range(kernel_l):
        il = ol * stride_l - padding_l + kl * dilation_l
        valid = output_mask & (il >= 0) & (il < in_l)
        # Clamp the gather offset to an in-bounds element and load
        # unconditionally; XPU masked out-of-bounds loads read neighbour
        # memory, so the window boundary is enforced on the VALUE via
        # tl.where (padding = min_val), never on the masked load.
        il_safe = tl.where(valid, il, 0)
        input_offset = nc_safe * in_l + il_safe
        value = tl.load(input_ptr + input_offset)
        value = tl.where(valid, value, min_val)
        # Strict '>' + ascending kl => first-max wins, matching torch.
        is_new_max = valid & (value > max_val)
        max_val = tl.where(is_new_max, value, max_val)
        max_idx = tl.where(is_new_max, il_safe, max_idx)

    tl.store(output_ptr + offsets, max_val, mask=output_mask)
    tl.store(indices_ptr + offsets, max_idx, mask=output_mask)


def _parse_pool_params(kernel_size, stride, padding, dilation):
    def _single(value, name):
        if isinstance(value, (list, tuple)):
            if len(value) != 1:
                raise ValueError(f"{name} must be a single int for 1D pooling")
            return int(value[0])
        return int(value)

    kernel_l = _single(kernel_size, "kernel_size")

    if stride is None or (isinstance(stride, (list, tuple)) and len(stride) == 0):
        stride_l = kernel_l
    else:
        stride_l = _single(stride, "stride")

    padding_l = _single(padding, "padding")
    dilation_l = _single(dilation, "dilation")

    if kernel_l <= 0:
        raise ValueError(f"kernel_size must be positive, but got {kernel_l}")
    if stride_l <= 0:
        raise ValueError(f"stride must be positive, but got {stride_l}")
    if padding_l < 0:
        raise ValueError(f"padding must be non-negative, but got {padding_l}")
    if dilation_l <= 0:
        raise ValueError(f"dilation must be positive, but got {dilation_l}")

    return kernel_l, stride_l, padding_l, dilation_l


def max_pool1d_with_indices(
    input: torch.Tensor,
    kernel_size,
    stride=None,
    padding=0,
    dilation=1,
    ceil_mode=False,
):
    logger.debug("GEMS_KUNLUNXIN MAX_POOL1D_WITH_INDICES")

    kernel_l, stride_l, padding_l, dilation_l = _parse_pool_params(
        kernel_size, stride, padding, dilation
    )

    squeeze_batch = input.dim() == 2
    if squeeze_batch:
        input = input.unsqueeze(0)

    input = input.contiguous()
    in_n, in_c, in_l = input.shape
    out_l = max_pool1d_output_size(
        in_l, kernel_l, stride_l, padding_l, dilation_l, ceil_mode
    )

    output = torch.empty((in_n, in_c, out_l), device=input.device, dtype=input.dtype)
    indices = torch.empty((in_n, in_c, out_l), device=input.device, dtype=torch.int64)

    total = output.numel()
    if total != 0:
        block = 1024
        grid = (triton.cdiv(total, block),)
        with torch_device_fn.device(input.device):
            max_pool1d_forward_flat_kernel[grid](
                input,
                output,
                indices,
                total,
                in_l,
                out_l,
                kernel_l,
                stride_l,
                padding_l,
                dilation_l,
                block,
                num_warps=1,
                buffer_size_limit=2048,
                isCloseVectorization=True,
            )

    if squeeze_batch:
        output = output.squeeze(0)
        indices = indices.squeeze(0)

    return output, indices
