import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)


def pool3d_output_size(
    in_size: int,
    kernel_size: int,
    stride: int,
    padding: int,
    dilation: int,
    ceil_mode: bool = False,
) -> int:
    """Compute one spatial dimension of the 3-D max-pool output."""
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
def max_pool3d_forward_kernel(
    input_ptr,
    output_ptr,
    total,
    in_d,
    in_h,
    in_w,
    out_d,
    out_h,
    out_w,
    kcount,
    kernel_h: tl.constexpr,
    kernel_w: tl.constexpr,
    stride_d: tl.constexpr,
    stride_h: tl.constexpr,
    stride_w: tl.constexpr,
    padding_d: tl.constexpr,
    padding_h: tl.constexpr,
    padding_w: tl.constexpr,
    dilation_d: tl.constexpr,
    dilation_h: tl.constexpr,
    dilation_w: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Forward kernel for 3-D max pooling, max values only (1-D scan based).

    Grid: (cdiv(N * C * D * H * W, BLOCK),)
    Each lane handles one flattened output voxel and scans the kd*kh*kw
    window with a small runtime loop (``kcount = kd*kh*kw``), keeping the
    compiled IR tiny.  A fully unrolled kd*kh*kw peel or a [BLOCK, KK] 2-D
    tile with a cross-lane reduction makes the Kunlunxin compiler explode;
    this 1-D scan compiles in seconds.  Semantics match ATen: ties pick the
    first window position in (D, H, W) order (strictly-greater update keeps
    the earliest tap).  Out-of-window (padding) taps are clamped to an
    in-bounds address and the loaded value is discarded with ``tl.where``,
    so the loads stay unmasked (the XPU backend treats compound i1-masked
    loads as a slow path).
    """
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    output_mask = offsets < total
    out_dhw = out_d * out_h * out_w
    nc_idx = offsets // out_dhw
    rem = offsets % out_dhw
    od = rem // (out_h * out_w)
    rem2 = rem % (out_h * out_w)
    oh = rem2 // out_w
    ow = rem2 % out_w
    in_hw = in_h * in_w
    d_base = od * stride_d - padding_d
    h_base = oh * stride_h - padding_h
    w_base = ow * stride_w - padding_w
    ncfg = kernel_h * kernel_w
    base_off = nc_idx * (in_d * in_hw)

    max_val = tl.full((BLOCK,), float("-inf"), dtype=tl.float32)
    for k in range(0, kcount):
        kd = k // ncfg
        kh = (k // kernel_w) % kernel_h
        kw = k % kernel_w
        id_ = d_base + kd * dilation_d
        ih_ = h_base + kh * dilation_h
        iw_ = w_base + kw * dilation_w
        active = (
            output_mask
            & (id_ >= 0)
            & (id_ < in_d)
            & (ih_ >= 0)
            & (ih_ < in_h)
            & (iw_ >= 0)
            & (iw_ < in_w)
        )
        id_s = tl.where(active, id_, 0)
        ih_s = tl.where(active, ih_, 0)
        iw_s = tl.where(active, iw_, 0)
        io = base_off + id_s * in_hw + ih_s * in_w + iw_s
        value = tl.load(input_ptr + io).to(tl.float32)
        value = tl.where(active, value, float("-inf"))
        max_val = tl.where(value > max_val, value, max_val)

    tl.store(output_ptr + offsets, max_val, mask=output_mask)


def _parse_pool3d_params(kernel_size, stride, padding, dilation):
    """Parse and validate 3-D pooling parameters.

    Each parameter can be an int (applied to all 3 spatial dims) or a
    3-element tuple/list (D, H, W).
    """

    def _parse_param(param, name, default=None):
        if param is None or (isinstance(param, (list, tuple)) and len(param) == 0):
            return default
        if isinstance(param, int):
            return param, param, param
        if isinstance(param, (list, tuple)) and len(param) == 3:
            return tuple(param)
        raise ValueError(f"Invalid {name}: {param}")

    kd, kh, kw = _parse_param(kernel_size, "kernel_size")
    sd, sh, sw = _parse_param(stride, "stride", default=(kd, kh, kw))
    pd, ph, pw = _parse_param(padding, "padding", default=(0, 0, 0))
    dd, dh, dw = _parse_param(dilation, "dilation", default=(1, 1, 1))

    if sd <= 0 or sh <= 0 or sw <= 0:
        raise ValueError(f"stride must be positive, but got stride=({sd}, {sh}, {sw})")
    if pd < 0 or ph < 0 or pw < 0:
        raise ValueError(
            f"padding must be non-negative, but got padding=({pd}, {ph}, {pw})"
        )
    if dd <= 0 or dh <= 0 or dw <= 0:
        raise ValueError(
            f"dilation must be positive, but got dilation=({dd}, {dh}, {dw})"
        )

    return kd, kh, kw, sd, sh, sw, pd, ph, pw, dd, dh, dw


def max_pool3d(
    input: torch.Tensor,
    kernel_size,
    stride=None,
    padding=0,
    dilation=1,
    ceil_mode=False,
):
    """Compute 3-D max pooling, returning the pooled max values.

    This matches ``aten::max_pool3d`` which returns only the value tensor
    (no argmax indices).
    """
    logger.debug("GEMS_KUNLUNXIN MAX_POOL3D")
    input = input.contiguous()

    params = _parse_pool3d_params(kernel_size, stride, padding, dilation)
    kd, kh, kw, sd, sh, sw, pd, ph, pw, dd, dh, dw = params

    in_n, in_c, in_d, in_h, in_w = input.shape
    out_d = pool3d_output_size(in_d, kd, sd, pd, dd, ceil_mode)
    out_h = pool3d_output_size(in_h, kh, sh, ph, dh, ceil_mode)
    out_w = pool3d_output_size(in_w, kw, sw, pw, dw, ceil_mode)

    output = torch.empty(
        (in_n, in_c, out_d, out_h, out_w), device=input.device, dtype=input.dtype
    )

    if output.numel() == 0:
        return output

    total = output.numel()
    block = 128
    kcount = kd * kh * kw

    grid = (triton.cdiv(total, block),)

    with torch_device_fn.device(input.device):
        max_pool3d_forward_kernel[grid](
            input,
            output,
            total,
            in_d,
            in_h,
            in_w,
            out_d,
            out_h,
            out_w,
            kcount,
            kh,
            kw,
            sd,
            sh,
            sw,
            pd,
            ph,
            pw,
            dd,
            dh,
            dw,
            block,
            num_warps=2,
            buffer_size_limit=2048,
            isCloseVectorization=True,
        )

    return output
