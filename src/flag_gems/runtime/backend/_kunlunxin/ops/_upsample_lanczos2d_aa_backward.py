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

import logging
import math

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)


@triton.jit
def _lanczos_aa_filter(x):
    pix = math.pi * x
    sinc = tl.where(x == 0.0, 1.0, tl.sin(pix) / pix)
    sinc_three = tl.where(x == 0.0, 1.0, tl.sin(pix / 3.0) / (pix / 3.0))
    return tl.where(x < 3.0, sinc * sinc_three, 0.0)


@triton.jit
def _f2i(x):
    _LO: tl.constexpr = -2147483648.0
    _HI: tl.constexpr = 2147483520.0
    return tl.minimum(tl.maximum(x, _LO), _HI).to(tl.int32)


@triton.jit
def _precompute_weights_kernel(
    weight_ptr,
    start_ptr,
    output_size,
    input_size,
    scale,
    support,
    invscale,
    MAX_KSIZE: tl.constexpr,
    BLOCK_KSIZE: tl.constexpr,
):
    oi = tl.program_id(0)
    if oi >= output_size:
        return

    offs = tl.arange(0, BLOCK_KSIZE)
    offs_mask = offs < MAX_KSIZE
    center = scale * (oi + 0.5)
    xmin = tl.maximum(_f2i(center - support + 0.5), 0)
    xsize = tl.minimum(_f2i(center + support + 0.5), input_size) - xmin
    xsize = tl.minimum(tl.maximum(xsize, 0), MAX_KSIZE)
    xmin_f = xmin.to(tl.float32)

    arg = tl.abs((offs.to(tl.float32) + xmin_f - center + 0.5) * invscale)
    raw_w = tl.where((offs < xsize) & offs_mask, _lanczos_aa_filter(arg), 0.0)
    total = tl.sum(raw_w, axis=0)
    weights = tl.where(total != 0.0, raw_w / total, 0.0)

    tl.store(weight_ptr + oi.to(tl.int64) * MAX_KSIZE + offs, weights, mask=offs_mask)
    tl.store(start_ptr + oi, xmin)


@triton.jit
def _fused_precomputed_backward_kernel(
    grad_out_ptr,
    grad_in_ptr,
    wx_ptr,
    wx_start_ptr,
    wy_ptr,
    wy_start_ptr,
    H_in,
    H_out,
    W_in,
    W_out,
    support_h,
    support_w,
    inv_h_scale,
    inv_w_scale,
    stride_go_nc,
    BLOCK_IW: tl.constexpr,
    MAX_OH: tl.constexpr,
    MAX_OW: tl.constexpr,
    MAX_KSIZE_H: tl.constexpr,
    MAX_KSIZE_W: tl.constexpr,
):
    pid_row = tl.program_id(0)
    pid_col = tl.program_id(1)
    nc = pid_row // H_in
    ih = pid_row % H_in
    iws = pid_col * BLOCK_IW + tl.arange(0, BLOCK_IW)
    iw_mask = iws < W_in

    oh_start = tl.maximum(
        _f2i((ih.to(tl.float32) + 0.5 - support_h) * inv_h_scale - 0.5), 0
    )
    ow_starts = tl.maximum(
        _f2i((iws.to(tl.float32) + 0.5 - support_w) * inv_w_scale - 0.5), 0
    )
    accum = tl.zeros((BLOCK_IW,), dtype=tl.float32)
    go_nc_base = nc.to(tl.int64) * stride_go_nc

    for d_ow in tl.static_range(MAX_OW):
        ow = ow_starts + d_ow
        ow_valid = iw_mask & (ow < W_out)
        ow_safe = tl.minimum(ow, W_out - 1)
        xmin = tl.load(wx_start_ptr + ow_safe, mask=ow_valid, other=0)
        kx = iws - xmin
        x_valid = ow_valid & (kx >= 0) & (kx < MAX_KSIZE_W)
        kx_safe = tl.minimum(tl.maximum(kx, 0), MAX_KSIZE_W - 1)
        wx = tl.load(
            wx_ptr + ow_safe.to(tl.int64) * MAX_KSIZE_W + kx_safe.to(tl.int64),
            mask=x_valid,
            other=0.0,
        )

        for d_oh in tl.static_range(MAX_OH):
            oh = oh_start + d_oh
            oh_valid = oh < H_out
            oh_safe = tl.minimum(oh, H_out - 1)
            ymin = tl.load(wy_start_ptr + oh_safe, mask=oh_valid, other=0)
            ky = ih - ymin
            y_valid = oh_valid & (ky >= 0) & (ky < MAX_KSIZE_H)
            ky_safe = tl.minimum(tl.maximum(ky, 0), MAX_KSIZE_H - 1)
            wy = tl.load(
                wy_ptr + oh_safe.to(tl.int64) * MAX_KSIZE_H + ky_safe.to(tl.int64),
                mask=y_valid,
                other=0.0,
            )
            valid = x_valid & y_valid
            grad = tl.load(
                grad_out_ptr
                + go_nc_base
                + oh_safe.to(tl.int64) * W_out
                + ow_safe.to(tl.int64),
                mask=valid,
                other=0.0,
            )
            accum += wx * wy * grad

    grad_in_offset = pid_row.to(tl.int64) * W_in + iws.to(tl.int64)
    tl.store(
        grad_in_ptr + grad_in_offset,
        accum.to(grad_in_ptr.dtype.element_ty),
        mask=iw_mask,
    )


@triton.jit
def _pass1_w_gather_nchw_kernel(
    grad_out_ptr,
    buf_ptr,
    wx_ptr,
    wx_start_ptr,
    W_in,
    W_out,
    support_w,
    inv_w_scale,
    BLOCK_IW: tl.constexpr,
    MAX_OW: tl.constexpr,
    MAX_KSIZE_W: tl.constexpr,
):
    pid_row = tl.program_id(0)
    pid_col = tl.program_id(1)

    iw_base = pid_col * BLOCK_IW
    iws = iw_base + tl.arange(0, BLOCK_IW)
    iw_mask = iws < W_in
    iw_f = iws.to(tl.float32)

    go_base = pid_row.to(tl.int64) * W_out
    buf_base = pid_row.to(tl.int64) * W_in

    ow_starts = tl.maximum(_f2i((iw_f + 0.5 - support_w) * inv_w_scale - 0.5), 0)

    accum = tl.zeros([BLOCK_IW], dtype=tl.float32)

    for d_ow in tl.static_range(MAX_OW):
        ow = ow_starts + d_ow
        ow_valid = iw_mask & (ow >= 0) & (ow < W_out)
        ow_safe = tl.maximum(tl.minimum(ow, W_out - 1), 0)

        xmin = tl.load(wx_start_ptr + ow_safe)
        k = iws - xmin
        in_range = ow_valid & (k >= 0) & (k < MAX_KSIZE_W)
        k_safe = tl.minimum(tl.maximum(k, 0), MAX_KSIZE_W - 1)
        wx_raw = tl.load(
            wx_ptr + ow_safe.to(tl.int64) * MAX_KSIZE_W + k_safe.to(tl.int64)
        )
        wx = tl.where(in_range, wx_raw, 0.0)

        g = tl.load(
            grad_out_ptr + go_base + ow_safe.to(tl.int64), mask=in_range, other=0.0
        )
        accum += wx * g

    tl.store(buf_ptr + buf_base + iws.to(tl.int64), accum, mask=iw_mask)


@triton.jit
def _pass2_h_gather_nchw_kernel(
    buf_ptr,
    grad_in_ptr,
    wy_ptr,
    wy_start_ptr,
    H_in,
    W_in,
    H_out,
    support_h,
    inv_h_scale,
    stride_buf_hw,
    BLOCK_IW: tl.constexpr,
    MAX_OH: tl.constexpr,
    MAX_KSIZE_H: tl.constexpr,
):
    pid_row = tl.program_id(0)
    pid_col = tl.program_id(1)

    nc = pid_row // H_in
    ih = pid_row % H_in
    ih_f = ih.to(tl.float32)

    iw_base = pid_col * BLOCK_IW
    iws = iw_base + tl.arange(0, BLOCK_IW)
    iw_mask = iws < W_in

    oh_start = tl.maximum(_f2i((ih_f + 0.5 - support_h) * inv_h_scale - 0.5), 0)

    buf_nc_base = nc.to(tl.int64) * stride_buf_hw

    accum = tl.zeros([BLOCK_IW], dtype=tl.float32)

    for d_oh in tl.static_range(MAX_OH):
        oh = oh_start + d_oh
        oh_valid = (oh >= 0) & (oh < H_out)
        oh_safe = tl.maximum(tl.minimum(oh, H_out - 1), 0)

        ymin = tl.load(wy_start_ptr + oh_safe)
        k = ih - ymin
        ih_in_range = oh_valid & (k >= 0) & (k < MAX_KSIZE_H)
        k_safe = tl.minimum(tl.maximum(k, 0), MAX_KSIZE_H - 1)
        wy_raw = tl.load(
            wy_ptr + oh_safe.to(tl.int64) * MAX_KSIZE_H + k_safe.to(tl.int64)
        )
        wy = tl.where(ih_in_range, wy_raw, 0.0)

        buf_off = buf_nc_base + oh_safe.to(tl.int64) * W_in + iws.to(tl.int64)
        b = tl.load(buf_ptr + buf_off, mask=iw_mask & ih_in_range, other=0.0)

        accum += wy * b

    gi_off = pid_row.to(tl.int64) * W_in + iws.to(tl.int64)
    tl.store(
        grad_in_ptr + gi_off,
        accum.to(grad_in_ptr.dtype.element_ty),
        mask=iw_mask,
    )


def _compute_scale(input_size, output_size, align_corners, scale=None):
    if align_corners:
        return float(input_size - 1) / (output_size - 1) if output_size > 1 else 0.0
    else:
        return (
            (1.0 / scale)
            if (scale is not None and scale > 0)
            else float(input_size) / output_size
        )


_FUSE_THRESHOLD = 1 << 20  # 1M elements


def _should_use_fused_path(total_elems):
    # XPU3: the fused kernel nests two tl.static_range loops (MAX_OW * MAX_OH),
    # whose unrolled body overflows the local-memory budget ("Failed to tune
    # buffer size."). The separable 2-pass kernels unroll a single loop each, so
    # always take that path on this backend.
    return False


def upsample_lanczos2d_aa_backward(
    grad_output: torch.Tensor,
    output_size,  # [H_out, W_out]
    input_size,  # [N, C, H_in, W_in]
    align_corners: bool,
    scales_h=None,
    scales_w=None,
) -> torch.Tensor:
    logger.debug("GEMS_KUNLUNXIN UPSAMPLE_LANCZOS2D_AA_BACKWARD")
    if grad_output.ndim != 4:
        raise RuntimeError(
            "Expected grad_output to be a tensor of dimension 4 but got: "
            f"dimension {grad_output.ndim}"
        )
    if len(output_size) != 2 or len(input_size) != 4:
        raise RuntimeError("Expected output_size[2] and input_size[4]")
    if not grad_output.is_floating_point():
        raise RuntimeError(
            f'"upsample_lanczos2d_aa_backward" not implemented for {grad_output.dtype}'
        )

    N, C, H_in, W_in = map(int, input_size)
    H_out, W_out = output_size

    if tuple(grad_output.shape) != (N, C, H_out, W_out):
        raise RuntimeError(
            f"Expected grad_output shape ({N}, {C}, {H_out}, {W_out}), "
            f"but got {tuple(grad_output.shape)}"
        )

    NC = N * C
    if NC == 0 or H_in == 0 or W_in == 0 or H_out == 0 or W_out == 0:
        return grad_output.new_zeros(input_size)

    # NOTE: the generic op routes fp64 (and some shapes) through a torch.mm GEMM
    # path.  On XPU that is both a numeric fallback (disallowed) and fp64 is
    # silently downcast anyway, so the vendor overlay always runs the Triton
    # input-side gather kernels.
    grad_out_flat = grad_output.contiguous().reshape(NC, H_out, W_out)

    h_scale = _compute_scale(H_in, H_out, align_corners, scales_h)
    w_scale = _compute_scale(W_in, W_out, align_corners, scales_w)

    INTERP_SIZE = 6
    support_h = (INTERP_SIZE * 0.5) * h_scale if h_scale >= 1.0 else INTERP_SIZE * 0.5
    support_w = (INTERP_SIZE * 0.5) * w_scale if w_scale >= 1.0 else INTERP_SIZE * 0.5
    invscale_h = 1.0 / h_scale if h_scale >= 1.0 else 1.0
    invscale_w = 1.0 / w_scale if w_scale >= 1.0 else 1.0

    MAX_KSIZE_H = math.ceil(support_h) * 2 + 1
    MAX_KSIZE_W = math.ceil(support_w) * 2 + 1

    _EPS = 1e-10
    inv_h_scale = 1.0 / max(h_scale, _EPS)
    inv_w_scale = 1.0 / max(w_scale, _EPS)

    MAX_OH = min(math.ceil(2 * support_h * inv_h_scale) + 2, max(H_out, 1))
    MAX_OW = min(math.ceil(2 * support_w * inv_w_scale) + 2, max(W_out, 1))

    BLOCK_IW = min(triton.next_power_of_2(max(W_in, 1)), 256)
    if BLOCK_IW < 32:
        BLOCK_IW = 32

    total_elems = NC * max(H_in * W_in, H_out * W_out)
    use_fused = _should_use_fused_path(total_elems)

    wy = torch.empty(
        max(H_out, 1), MAX_KSIZE_H, dtype=torch.float32, device=grad_output.device
    )
    wx = torch.empty(
        max(W_out, 1), MAX_KSIZE_W, dtype=torch.float32, device=grad_output.device
    )
    wy_start = torch.empty(max(H_out, 1), dtype=torch.int32, device=grad_output.device)
    wx_start = torch.empty(max(W_out, 1), dtype=torch.int32, device=grad_output.device)
    _precompute_weights_kernel[(H_out,)](
        wy,
        wy_start,
        H_out,
        H_in,
        h_scale,
        support_h,
        invscale_h,
        MAX_KSIZE=MAX_KSIZE_H,
        BLOCK_KSIZE=triton.next_power_of_2(MAX_KSIZE_H),
    )
    _precompute_weights_kernel[(W_out,)](
        wx,
        wx_start,
        W_out,
        W_in,
        w_scale,
        support_w,
        invscale_w,
        MAX_KSIZE=MAX_KSIZE_W,
        BLOCK_KSIZE=triton.next_power_of_2(MAX_KSIZE_W),
    )

    if use_fused:
        grad_in_flat = torch.empty(
            NC, H_in, W_in, dtype=grad_output.dtype, device=grad_output.device
        )
        grid = (NC * H_in, triton.cdiv(W_in, BLOCK_IW))
        _fused_precomputed_backward_kernel[grid](
            grad_out_flat,
            grad_in_flat,
            wx,
            wx_start,
            wy,
            wy_start,
            H_in,
            H_out,
            W_in,
            W_out,
            support_h,
            support_w,
            inv_h_scale,
            inv_w_scale,
            H_out * W_out,
            BLOCK_IW=BLOCK_IW,
            MAX_OH=MAX_OH,
            MAX_OW=MAX_OW,
            MAX_KSIZE_H=MAX_KSIZE_H,
            MAX_KSIZE_W=MAX_KSIZE_W,
        )
        return grad_in_flat.reshape(N, C, H_in, W_in)

    buf = torch.empty(NC, H_out, W_in, dtype=torch.float32, device=grad_output.device)
    grid1 = (NC * H_out, triton.cdiv(W_in, BLOCK_IW))
    _pass1_w_gather_nchw_kernel[grid1](
        grad_out_flat,
        buf,
        wx,
        wx_start,
        W_in,
        W_out,
        support_w,
        inv_w_scale,
        BLOCK_IW=BLOCK_IW,
        MAX_OW=MAX_OW,
        MAX_KSIZE_W=MAX_KSIZE_W,
    )

    grad_in_flat = torch.empty(
        NC, H_in, W_in, dtype=grad_output.dtype, device=grad_output.device
    )
    grid2 = (NC * H_in, triton.cdiv(W_in, BLOCK_IW))
    _pass2_h_gather_nchw_kernel[grid2](
        buf,
        grad_in_flat,
        wy,
        wy_start,
        H_in,
        W_in,
        H_out,
        support_h,
        inv_h_scale,
        H_out * W_in,
        BLOCK_IW=BLOCK_IW,
        MAX_OH=MAX_OH,
        MAX_KSIZE_H=MAX_KSIZE_H,
    )

    return grad_in_flat.reshape(N, C, H_in, W_in)


def upsample_lanczos2d_aa_backward_grad_input(
    grad_output: torch.Tensor,
    output_size,
    input_size,
    align_corners: bool,
    scales_h=None,
    scales_w=None,
    *,
    grad_input: torch.Tensor,
) -> torch.Tensor:
    logger.debug("GEMS_KUNLUNXIN UPSAMPLE_LANCZOS2D_AA_BACKWARD_GRAD_INPUT")
    if grad_input.device != grad_output.device:
        raise RuntimeError(
            f"Expected grad_input on {grad_output.device}, but got {grad_input.device}"
        )
    if grad_input.dtype != grad_output.dtype:
        raise RuntimeError(
            f"Expected grad_input dtype {grad_output.dtype}, but got {grad_input.dtype}"
        )
    result = upsample_lanczos2d_aa_backward(
        grad_output,
        output_size,
        input_size,
        align_corners,
        scales_h,
        scales_w,
    )
    grad_input.resize_(result.shape)
    grad_input.copy_(result)
    return grad_input


def _install_into_flag_gems_ops():
    """Route generic submodule/package names to the vendor Triton impl.

    Callers import these entry points directly from the generic submodule,
    whose kernels use sinpi (libdevice) and fail to link on XPU3 with an
    "Unsupported" symbol error.
    """
    try:
        import flag_gems.ops as _fg_ops
        import flag_gems.ops._upsample_lanczos2d_aa_backward as _fg_mod
    except Exception:  # pragma: no cover - defensive; import ordering
        return
    for target in (_fg_ops, _fg_mod):
        target.upsample_lanczos2d_aa_backward = upsample_lanczos2d_aa_backward
        target.upsample_lanczos2d_aa_backward_grad_input = (
            upsample_lanczos2d_aa_backward_grad_input
        )


_install_into_flag_gems_ops()
