import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)


@libentry()
@triton.jit
def _fractional_max_pool3d_forward_kernel(
    input_ptr,
    output_ptr,
    indices_ptr,
    random_samples_ptr,
    numel,
    input_depth: tl.constexpr,
    input_height: tl.constexpr,
    input_width: tl.constexpr,
    output_depth: tl.constexpr,
    output_height: tl.constexpr,
    output_width: tl.constexpr,
    kernel_depth: tl.constexpr,
    kernel_height: tl.constexpr,
    kernel_width: tl.constexpr,
    alpha_depth,
    alpha_height,
    alpha_width,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    store_mask = offsets < numel
    safe_offsets = tl.where(store_mask, offsets, 0)

    output_col = safe_offsets % output_width
    ohw = safe_offsets // output_width
    output_row = ohw % output_height
    odt = ohw // output_height
    output_dep = odt % output_depth
    nc = odt // output_depth

    sample_depth = tl.load(random_samples_ptr + nc * 3).to(tl.float32)
    sample_height = tl.load(random_samples_ptr + nc * 3 + 1).to(tl.float32)
    sample_width = tl.load(random_samples_ptr + nc * 3 + 2).to(tl.float32)

    sample_alpha_d = (sample_depth * alpha_depth).to(tl.int32)
    sample_alpha_h = (sample_height * alpha_height).to(tl.int32)
    sample_alpha_w = (sample_width * alpha_width).to(tl.int32)

    start_depth = ((output_dep.to(tl.float32) + sample_depth) * alpha_depth).to(
        tl.int32
    ) - sample_alpha_d
    start_height = ((output_row.to(tl.float32) + sample_height) * alpha_height).to(
        tl.int32
    ) - sample_alpha_h
    start_width = ((output_col.to(tl.float32) + sample_width) * alpha_width).to(
        tl.int32
    ) - sample_alpha_w

    start_depth = tl.where(
        output_dep == output_depth - 1, input_depth - kernel_depth, start_depth
    )
    start_height = tl.where(
        output_row == output_height - 1, input_height - kernel_height, start_height
    )
    start_width = tl.where(
        output_col == output_width - 1, input_width - kernel_width, start_width
    )

    plane_base = input_ptr + nc * (input_depth * input_height * input_width)
    max_value = tl.full((BLOCK_SIZE,), -float("inf"), tl.float32)
    max_index = tl.full((BLOCK_SIZE,), -1, tl.int64)
    for kernel_dep in tl.static_range(0, kernel_depth):
        input_dep = start_depth + kernel_dep
        for kernel_row in tl.static_range(0, kernel_height):
            input_row = start_height + kernel_row
            for kernel_col in tl.static_range(0, kernel_width):
                input_col = start_width + kernel_col
                flat = (
                    input_dep * input_height * input_width
                    + input_row * input_width
                    + input_col
                )
                value = tl.load(plane_base + flat).to(tl.float32)
                update = value > max_value
                max_value = tl.where(update, value, max_value)
                max_index = tl.where(update, flat.to(tl.int64), max_index)

    tl.store(
        output_ptr + offsets,
        max_value.to(output_ptr.dtype.element_ty),
        mask=store_mask,
    )
    tl.store(indices_ptr + offsets, max_index, mask=store_mask)


def _parse_size3(value):
    if isinstance(value, (int, float)):
        return value, value, value
    return value[0], value[1], value[2]


def fractional_max_pool3d(
    input,
    kernel_size,
    output_size=None,
    output_ratio=None,
    return_indices=True,
    _random_samples=None,
):
    logger.debug("GEMS_KUNLUNXIN FRACTIONAL_MAX_POOL3D")
    if isinstance(output_ratio, torch.Tensor) and _random_samples is None:
        _random_samples = output_ratio
        output_ratio = None
    assert input.dim() == 5, f"Expected 5D input, got {input.dim()}D"
    input = input.contiguous()
    batch_size, channels, input_depth, input_height, input_width = input.shape
    kernel_depth, kernel_height, kernel_width = _parse_size3(kernel_size)
    if output_size is not None:
        output_depth, output_height, output_width = _parse_size3(output_size)
    elif output_ratio is not None:
        ratio_depth, ratio_height, ratio_width = _parse_size3(output_ratio)
        output_depth = int(input_depth * ratio_depth)
        output_height = int(input_height * ratio_height)
        output_width = int(input_width * ratio_width)
    else:
        raise ValueError("Either output_size or output_ratio must be specified")
    assert output_depth + kernel_depth - 1 <= input_depth
    assert output_height + kernel_height - 1 <= input_height
    assert output_width + kernel_width - 1 <= input_width

    if _random_samples is None:
        _random_samples = torch.rand(
            batch_size, channels, 3, device=input.device, dtype=input.dtype
        )
    else:
        assert _random_samples.shape == (batch_size, channels, 3)
        _random_samples = _random_samples.to(dtype=input.dtype).contiguous()

    output = torch.empty(
        (batch_size, channels, output_depth, output_height, output_width),
        device=input.device,
        dtype=input.dtype,
    )
    indices = torch.empty(
        (batch_size, channels, output_depth, output_height, output_width),
        device=input.device,
        dtype=torch.int64,
    )
    if output.numel() == 0:
        return (output, indices) if return_indices else output

    alpha_depth = (
        (input_depth - kernel_depth) / (output_depth - 1) if output_depth > 1 else 0.0
    )
    alpha_height = (
        (input_height - kernel_height) / (output_height - 1)
        if output_height > 1
        else 0.0
    )
    alpha_width = (
        (input_width - kernel_width) / (output_width - 1) if output_width > 1 else 0.0
    )
    numel = output.numel()
    block_size = 128 if numel >= 8192 else 64
    grid = (triton.cdiv(numel, block_size),)
    with torch_device_fn.device(input.device):
        _fractional_max_pool3d_forward_kernel[grid](
            input,
            output,
            indices,
            _random_samples.reshape(batch_size * channels, 3),
            numel,
            input_depth,
            input_height,
            input_width,
            output_depth,
            output_height,
            output_width,
            kernel_depth,
            kernel_height,
            kernel_width,
            alpha_depth,
            alpha_height,
            alpha_width,
            BLOCK_SIZE=block_size,
            num_warps=1,
            isCloseVectorization=True,
            buffer_size_limit=2048,
        )
    if return_indices:
        return output, indices
    return output
