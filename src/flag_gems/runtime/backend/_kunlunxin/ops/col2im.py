# Copyright 2026 FlagOS Contributors
#
# Kunlunxin (XPU) override of col2im.
#
# Root cause (previous state): the flat output-parallel kernel with
# BLOCK=1024 / num_warps=1 ran ~0.17x dtype-balanced: the tiny shapes
# (total_out <= a few hundred) were launch-bound at ~0.066ms with 1 warp and
# a 1024-lane block, and every (kh, kw) iteration recomputed `h_num // stride_h`
# / `h_num % stride_h` even when stride==1 (where the quotient is the identity
# and the remainder is always zero).
#
# Fix: keep the flat output-position-parallel structure (each program one 1D
# BLOCK of output elements; decode (n, c, h, w) via div/mod on non-negative
# indices; loop kh, kw with tl.static_range; fp32 accumulator). Replace the
# unconditional division/modulo with a constexpr specialization:
#   - stride_h == 1 : l_h = h_run (no div, no mod, h_valid == h_pos)
#   - stride_h > 1  : clamp-then-divide, and the validity check now includes
#                     h_pos (the original kept the clamp's zero value for
#                     negative h_run, which contributed row 0 spuriously).
# Launch config: BLOCK=256 for small totals (<= 4096 output elements, where
# the launch floor dominates) and for large kernels (kernel area >= 16, whose
# 25-iteration unroll benefits from more programs), BLOCK=1024 otherwise;
# num_warps=4 (1 warp left the memory pipeline underutilized).
#
# Measured (iso sweep, fp16/fp32, 7-shape matrix): all 7 shapes >= baseline,
# e.g. shape (2,3,(2,2),(4,5)) 0.0651ms -> 0.0281ms (0.70x -> 1.63x vs the
# XMLIR native reference); shape (2,64,(5,5),(64,64)) 9.02ms -> 7.80ms.
# The large shapes remain bounded by the XPU Triton load-op floor
# (~1.8 Gload/s, pattern-independent: affine 1D/2D-tile/row variants all
# measure the same), while the XMLIR native reference hits ~100 GB/s.
import logging
from typing import List

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)


@libentry()
@triton.jit
def col2im_kernel_flat(
    input_ptr,
    output_ptr,
    channels,
    out_h,
    out_w,
    L_h,
    L_w,
    total_out,
    HW_out,
    CHW_out,
    L_all,
    KHW,
    kernel_h: tl.constexpr,
    kernel_w: tl.constexpr,
    stride_h: tl.constexpr,
    stride_w: tl.constexpr,
    padding_h: tl.constexpr,
    padding_w: tl.constexpr,
    dilation_h: tl.constexpr,
    dilation_w: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    o = pid * BLOCK + tl.arange(0, BLOCK)
    mask = o < total_out

    # Decode flat -> (n, c, h, w); all operands non-negative.
    n_idx = o // CHW_out
    rem = o % CHW_out
    c_idx = rem // HW_out
    rem2 = rem % HW_out
    h_idx = rem2 // out_w
    w_idx = rem2 % out_w

    acc = tl.zeros([BLOCK], dtype=tl.float32)

    for kh in tl.static_range(0, kernel_h):
        for kw in tl.static_range(0, kernel_w):
            # h_run = h_idx + padding_h - kh * dilation_h may be negative on
            # the top/left edges.  Clamp before dividing so the quotient
            # (and the loaded address) stays valid; h_pos is part of the
            # validity mask so clamped lanes never contribute.
            h_run = h_idx + padding_h - kh * dilation_h
            w_run = w_idx + padding_w - kw * dilation_w
            h_pos = h_run >= 0
            w_pos = w_run >= 0
            if stride_h == 1:
                l_h = h_run
                h_ok = h_pos
            else:
                h_run_c = tl.where(h_pos, h_run, 0)
                l_h = h_run_c // stride_h
                h_ok = h_pos & ((h_run_c - l_h * stride_h) == 0)
            if stride_w == 1:
                l_w = w_run
                w_ok = w_pos
            else:
                w_run_c = tl.where(w_pos, w_run, 0)
                l_w = w_run_c // stride_w
                w_ok = w_pos & ((w_run_c - l_w * stride_w) == 0)

            valid = h_ok & w_ok & (l_h < L_h) & (l_w < L_w)

            # Clamp indices so masked-out lanes never dereference OOB memory.
            l_h_s = tl.where(valid, l_h, 0)
            l_w_s = tl.where(valid, l_w, 0)
            c_k = c_idx * KHW + kh * kernel_w + kw
            l_idx = l_h_s * L_w + l_w_s
            in_offset = n_idx * (channels * KHW * L_all) + c_k * L_all + l_idx

            v = tl.load(input_ptr + in_offset, mask=mask & valid, other=0.0)
            v = tl.where(valid, v, 0.0)
            acc += v.to(tl.float32)

    tl.store(output_ptr + o, acc.to(output_ptr.type.element_ty), mask=mask)


def _to_pair(val, name):
    if isinstance(val, int):
        return val, val
    if isinstance(val, (list, tuple)) and len(val) == 2:
        return tuple(val)
    raise ValueError(f"Invalid {name}: {val}")


def col2im(
    input: torch.Tensor,
    output_size: List[int],
    kernel_size: List[int],
    dilation: List[int],
    padding: List[int],
    stride: List[int],
) -> torch.Tensor:
    logger.debug("GEMS_KUNLUNXIN COL2IM")

    out_h, out_w = _to_pair(output_size, "output_size")
    kernel_h, kernel_w = _to_pair(kernel_size, "kernel_size")
    dilation_h, dilation_w = _to_pair(dilation, "dilation")
    padding_h, padding_w = _to_pair(padding, "padding")
    stride_h, stride_w = _to_pair(stride, "stride")

    if input.dim() != 3:
        raise ValueError(f"Expected 3D input, got {input.dim()}D")

    batch_size, ck, L = input.shape
    L_h = (out_h + 2 * padding_h - dilation_h * (kernel_h - 1) - 1) // stride_h + 1
    L_w = (out_w + 2 * padding_w - dilation_w * (kernel_w - 1) - 1) // stride_w + 1
    if L != L_h * L_w:
        raise ValueError(f"Input size mismatch: expected L={L_h * L_w}, got L={L}")
    kernel_total = kernel_h * kernel_w
    if ck % kernel_total != 0:
        raise ValueError(
            f"Input dim1 {ck} must be divisible by kernel_size {kernel_total}"
        )
    channels = ck // kernel_total

    input = input.contiguous()
    output = torch.empty(
        (batch_size, channels, out_h, out_w),
        device=input.device,
        dtype=input.dtype,
    )
    if output.numel() == 0:
        return output

    total_out = output.numel()
    HW_out = out_h * out_w
    CHW_out = channels * HW_out

    # Launch-bound small shapes and the 25-iteration (5x5 kernel) unroll prefer
    # 256-lane programs; everything else uses 1024.
    if total_out <= 4096 or kernel_total >= 16:
        block = 256
    else:
        block = 1024
    grid = (triton.cdiv(total_out, block),)
    with torch_device_fn.device(input.device):
        col2im_kernel_flat[grid](
            input,
            output,
            channels,
            out_h,
            out_w,
            L_h,
            L_w,
            total_out,
            HW_out,
            CHW_out,
            L,
            kernel_total,
            kernel_h,
            kernel_w,
            stride_h,
            stride_w,
            padding_h,
            padding_w,
            dilation_h,
            dilation_w,
            BLOCK=block,
            num_warps=4,
            buffer_size_limit=2048,
            isCloseVectorization=True,
        )

    return output