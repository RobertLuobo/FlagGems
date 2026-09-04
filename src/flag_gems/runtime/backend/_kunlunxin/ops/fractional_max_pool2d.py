import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)


@libentry()
@triton.jit
def _fractional_max_pool2d_forward_kernel(
    input_ptr,
    output_ptr,
    indices_ptr,
    random_samples_ptr,
    input_channels: tl.constexpr,
    input_height: tl.constexpr,
    input_width: tl.constexpr,
    output_height: tl.constexpr,
    output_width: tl.constexpr,
    kernel_height: tl.constexpr,
    kernel_width: tl.constexpr,
    alpha_height,
    alpha_width,
):
    output_column = tl.program_id(0)
    row_id = tl.program_id(1)
    channel = tl.program_id(2)
    batch = row_id // output_height
    output_row = row_id % output_height
    sample_offset = (batch * input_channels + channel) * 2
    sample_height = tl.load(random_samples_ptr + sample_offset).to(tl.float32)
    sample_width = tl.load(random_samples_ptr + sample_offset + 1).to(tl.float32)

    start_height = ((output_row.to(tl.float32) + sample_height) * alpha_height).to(
        tl.int32
    ) - (sample_height * alpha_height).to(tl.int32)
    start_width = ((output_column.to(tl.float32) + sample_width) * alpha_width).to(
        tl.int32
    ) - (sample_width * alpha_width).to(tl.int32)
    start_height = tl.where(
        output_row == output_height - 1,
        input_height - kernel_height,
        start_height,
    )
    start_width = tl.where(
        output_column == output_width - 1,
        input_width - kernel_width,
        start_width,
    )

    max_value = tl.full((), -float("inf"), tl.float32)
    max_index = tl.full((), -1, tl.int64)
    for kernel_row in tl.static_range(0, kernel_height):
        for kernel_column in tl.static_range(0, kernel_width):
            input_row = start_height + kernel_row
            input_column = start_width + kernel_column
            input_offset = (
                (batch * input_channels + channel) * input_height + input_row
            ) * input_width + input_column
            value = tl.load(input_ptr + input_offset).to(tl.float32)
            update = value > max_value
            max_value = tl.where(update, value, max_value)
            max_index = tl.where(
                update,
                input_row * input_width + input_column,
                max_index,
            )

    output_offset = (
        (batch * input_channels + channel) * output_height + output_row
    ) * output_width + output_column
    tl.store(output_ptr + output_offset, max_value)
    tl.store(indices_ptr + output_offset, max_index)



@libentry()
@triton.jit
def _fractional_max_pool2d_backward_probe_kernel(
    grad_output_ptr,
    indices_ptr,
    grad_input_ptr,
    total,
    in_hw,
    in_w,
    out_h,
    out_w,
    out_hw,
    alpha_h,
    alpha_w,
    kernel_h: tl.constexpr,
    kernel_w: tl.constexpr,
    BLOCK: tl.constexpr,
    PROBE: tl.constexpr,
    SLACK: tl.constexpr,
):
    """Input-parallel bounded-probe backward kernel.

    Each lane is one input position p (flat within the (n,c) plane).  The
    output positions that can scatter into p form a contiguous candidate
    rectangle [q_lo_h, q_hi_h] x [q_lo_w, q_hi_w] (start functions are
    monotonically non-decreasing with alpha >= 1), so we only probe a
    compile-time-fixed window around the estimate floor(p / alpha) instead of
    scanning every output element.  Membership is verified exactly via the
    forward-produced indices (idx == p), extra probes are masked out.
    """
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    input_mask = offsets < total
    nc = offsets // in_hw
    rem = offsets % in_hw
    h = rem // in_w
    w = rem % in_w
    nc_safe = tl.where(input_mask, nc, 0)

    # Guarded alpha (alpha == 0 only when out == 1, where the probe range is
    # forced to {0} below).
    a_h = tl.where(alpha_h > 0, alpha_h, 1.0)
    a_w = tl.where(alpha_w > 0, alpha_w, 1.0)

    e_h_lo = (h + 1 - kernel_h).to(tl.float32) / a_h
    e_h_hi = (h + 1).to(tl.float32) / a_h
    q_h_lo = e_h_lo.to(tl.int32) - SLACK
    q_h_hi = e_h_hi.to(tl.int32) + SLACK
    q_h_lo = tl.where(out_h > 1, q_h_lo, 0)
    q_h_hi = tl.where(out_h > 1, q_h_hi, out_h - 1)
    q_h_lo = tl.maximum(q_h_lo, 0)
    q_h_hi = tl.minimum(q_h_hi, out_h - 1)

    e_w_lo = (w + 1 - kernel_w).to(tl.float32) / a_w
    e_w_hi = (w + 1).to(tl.float32) / a_w
    q_w_lo = e_w_lo.to(tl.int32) - SLACK
    q_w_hi = e_w_hi.to(tl.int32) + SLACK
    q_w_lo = tl.where(out_w > 1, q_w_lo, 0)
    q_w_hi = tl.where(out_w > 1, q_w_hi, out_w - 1)
    q_w_lo = tl.maximum(q_w_lo, 0)
    q_w_hi = tl.minimum(q_w_hi, out_w - 1)

    out_base = nc_safe * out_hw
    grad_acc = tl.zeros((BLOCK,), dtype=tl.float32)
    for th in tl.static_range(PROBE):
        oh = q_h_lo + th
        valid_h = th <= (q_h_hi - q_h_lo)
        oh_safe = tl.where(valid_h, oh, 0)
        for tw in tl.static_range(PROBE):
            ow = q_w_lo + tw
            valid = input_mask & valid_h & (tw <= (q_w_hi - q_w_lo))
            ow_safe = tl.where(valid, ow, 0)
            o_off = out_base + oh_safe * out_w + ow_safe
            idx_val = tl.load(indices_ptr + o_off, mask=valid, other=-1)
            grad_val = tl.load(grad_output_ptr + o_off, mask=valid, other=0.0)
            match = valid & (idx_val.to(tl.int32) == rem)
            grad_acc += tl.where(match, grad_val.to(tl.float32), 0.0)

    tl.store(grad_input_ptr + offsets, grad_acc, mask=input_mask)


@libentry()
@triton.jit
def _fractional_max_pool2d_backward_scatter_kernel(
    grad_output_ptr,
    indices_ptr,
    grad_input_ptr,
    total_out,
    in_hw,
    out_hw,
    BLOCK: tl.constexpr,
):
    """Output-parallel non-atomic scatter backward kernel (alpha >= 2).

    When alpha >= 2 the pooling windows start(i + 1) >= start(i) + k, so the
    windows are pairwise disjoint and every input position is the maximum of
    at most one output position: the index map idx -> input is injective.
    Therefore each build only stores grad_output[o] to grad_input[idx[o]]
    exactly once; no atomic accumulation is required.
    """
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    out_mask = offsets < total_out
    nc = offsets // out_hw
    idx = tl.load(indices_ptr + offsets, mask=out_mask, other=0)
    g = tl.load(grad_output_ptr + offsets, mask=out_mask, other=0.0)
    tl.store(grad_input_ptr + nc * in_hw + idx, g, mask=out_mask)

def _parse_size(value):
    if isinstance(value, (int, float)):
        return value, value
    return value[0], value[1]


def fractional_max_pool2d(
    input,
    kernel_size,
    output_size=None,
    output_ratio=None,
    return_indices=True,
    _random_samples=None,
):
    logger.debug("GEMS_KUNLUNXIN FRACTIONAL_MAX_POOL2D")
    if isinstance(output_ratio, torch.Tensor) and _random_samples is None:
        _random_samples = output_ratio
        output_ratio = None
    assert input.dim() == 4, f"Expected 4D input, got {input.dim()}D"
    input = input.contiguous()
    batch_size, channels, input_height, input_width = input.shape
    kernel_height, kernel_width = _parse_size(kernel_size)
    if output_size is not None:
        output_height, output_width = _parse_size(output_size)
    elif output_ratio is not None:
        ratio_height, ratio_width = _parse_size(output_ratio)
        output_height = int(input_height * ratio_height)
        output_width = int(input_width * ratio_width)
    else:
        raise ValueError("Either output_size or output_ratio must be specified")
    assert output_height + kernel_height - 1 <= input_height
    assert output_width + kernel_width - 1 <= input_width

    if _random_samples is None:
        _random_samples = torch.rand(
            batch_size,
            channels,
            2,
            device=input.device,
            dtype=input.dtype,
        )
    else:
        assert _random_samples.shape == (batch_size, channels, 2)
        _random_samples = _random_samples.to(dtype=input.dtype).contiguous()

    output = torch.empty(
        (batch_size, channels, output_height, output_width),
        device=input.device,
        dtype=input.dtype,
    )
    indices = torch.empty(
        (batch_size, channels, output_height, output_width),
        device=input.device,
        dtype=torch.int64,
    )
    alpha_height = (
        (input_height - kernel_height) / (output_height - 1)
        if output_height > 1
        else 0.0
    )
    alpha_width = (
        (input_width - kernel_width) / (output_width - 1) if output_width > 1 else 0.0
    )
    grid = (output_width, batch_size * output_height, channels)
    with torch_device_fn.device(input.device):
        _fractional_max_pool2d_forward_kernel[grid](
            input,
            output,
            indices,
            _random_samples.reshape(batch_size * channels, 2),
            channels,
            input_height,
            input_width,
            output_height,
            output_width,
            kernel_height,
            kernel_width,
            alpha_height,
            alpha_width,
            isCloseVectorization=True,
            buffer_size_limit=2048,
        )
    if return_indices:
        return output, indices
    return output


def fractional_max_pool2d_backward(
    grad_output,
    input,
    kernel_size,
    output_size,
    indices,
):
    logger.debug("GEMS_KUNLUNXIN FRACTIONAL_MAX_POOL2D_BACKWARD")
    input = input.contiguous()
    grad_output = grad_output.contiguous()
    indices = indices.contiguous()
    batch_size, channels, input_height, input_width = input.shape
    output_height, output_width = _parse_size(output_size)
    kernel_height, kernel_width = _parse_size(kernel_size)
    total = input.numel()
    if total == 0:
        return torch.empty_like(input)
    in_hw = input_height * input_width
    out_hw = output_height * output_width
    alpha_h = (
        (input_height - kernel_height) / (output_height - 1)
        if output_height > 1
        else 0.0
    )
    alpha_w = (
        (input_width - kernel_width) / (output_width - 1) if output_width > 1 else 0.0
    )
    core_kernel = max(kernel_height, kernel_width)
    if float(alpha_h) >= 2.0 and float(alpha_w) >= 2.0:
        # alpha >= 2: windows are pairwise disjoint (start(i+1) >= start(i) + k),
        # so the idx -> input map is injective and a non-atomic output-parallel
        # scatter is exact.  Positions covered by no window are left at 0.
        grad_input = torch.zeros_like(input)
        _fractional_max_pool2d_backward_scatter_kernel[
            (triton.cdiv(grad_output.numel(), 256),)
        ](
            grad_output,
            indices,
            grad_input,
            grad_output.numel(),
            in_hw,
            out_hw,
            BLOCK=256,
            isCloseVectorization=True,
            buffer_size_limit=2048,
        )
        return grad_input
    # Generic alpha in [1, 2): overlapping windows; keep the conservative
    # bounded-scan loop (correct for all alpha >= 1).
    grad_input = torch.empty_like(input)
    probe = 4 * core_kernel + 8
    slack = 2 * core_kernel + 2
    block = 128
    grid = (triton.cdiv(total, block),)
    with torch_device_fn.device(input.device):
        _fractional_max_pool2d_backward_probe_kernel[grid](
            grad_output,
            indices,
            grad_input,
            total,
            in_hw,
            input_width,
            output_height,
            output_width,
            out_hw,
            alpha_h,
            alpha_w,
            kernel_height,
            kernel_width,
            BLOCK=block,
            PROBE=probe,
            SLACK=slack,
            isCloseVectorization=True,
            buffer_size_limit=2048,
        )
    return grad_input
