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

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)


_DIM_ARGS = [
    "in_n",
    "input_height",
    "input_width",
    "out_c",
    "out_height",
    "out_width",
    "input_n_stride",
    "input_c_stride",
    "input_height_stride",
    "input_width_stride",
    "weight_n_stride",
    "weight_c_stride",
    "weight_height_stride",
    "weight_width_stride",
    "output_n_stride",
    "output_c_stride",
    "output_height_stride",
    "output_width_stride",
    "out_per_group",
    "weight_height",
    "weight_width",
    "stride_height",
    "stride_width",
    "padding_height",
    "padding_width",
    "dilation_height",
    "dilation_width",
    "groups",
    "has_bias",
]


@triton.jit(do_not_specialize=_DIM_ARGS)
def conv_transpose2d_forward_kernel(
    input_pointer,
    weight_pointer,
    output_pointer,
    bias_pointer,
    in_n,
    input_height,
    input_width,
    out_c,
    out_height,
    out_width,
    input_n_stride,
    input_c_stride,
    input_height_stride,
    input_width_stride,
    weight_n_stride,
    weight_c_stride,
    weight_height_stride,
    weight_width_stride,
    output_n_stride,
    output_c_stride,
    output_height_stride,
    output_width_stride,
    out_per_group,
    weight_height,
    weight_width,
    stride_height,
    stride_width,
    padding_height,
    padding_width,
    dilation_height,
    dilation_width,
    groups,
    has_bias,
    CPG: tl.constexpr,
    BLOCK: tl.constexpr,
):
    # One output element per lane (output-side gather, no cross-program
    # atomics). m enumerates the flat (n, oc, oh, ow) output index.
    m = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    total = in_n * out_c * out_height * out_width
    mmask = m < total

    ow = m % out_width
    t = m // out_width
    oh = t % out_height
    t = t // out_height
    oc = t % out_c
    ni = t // out_c

    g = oc // out_per_group
    oc_local = oc % out_per_group

    acc = tl.zeros((BLOCK,), tl.float32)
    for r in range(0, weight_height):
        hnum = oh + padding_height - r * dilation_height
        ih = hnum // stride_height
        hvalid = (hnum == ih * stride_height) & (ih >= 0) & (ih < input_height)
        safe_ih = tl.where(hvalid, ih, 0)
        for s in range(0, weight_width):
            wnum = ow + padding_width - s * dilation_width
            iw = wnum // stride_width
            wvalid = (wnum == iw * stride_width) & (iw >= 0) & (iw < input_width)
            safe_iw = tl.where(wvalid, iw, 0)
            valid = mmask & hvalid & wvalid
            for ic_local in range(0, CPG):
                ic = g * CPG + ic_local
                xv = tl.load(
                    input_pointer
                    + ni * input_n_stride
                    + ic * input_c_stride
                    + safe_ih * input_height_stride
                    + safe_iw * input_width_stride,
                    mask=valid,
                    other=0.0,
                )
                wv = tl.load(
                    weight_pointer
                    + ic * weight_n_stride
                    + oc_local * weight_c_stride
                    + r * weight_height_stride
                    + s * weight_width_stride,
                    mask=mmask,
                    other=0.0,
                )
                acc += tl.where(valid, xv.to(tl.float32) * wv.to(tl.float32), 0.0)

    if has_bias:
        b = tl.load(bias_pointer + oc, mask=mmask, other=0.0)
        acc += b.to(tl.float32)

    tl.store(
        output_pointer + m,
        acc.to(output_pointer.dtype.element_ty),
        mask=mmask,
    )


def conv_transpose2d(
    input,
    weight,
    bias=None,
    stride=1,
    padding=0,
    output_padding=0,
    groups=1,
    dilation=1,
):
    logger.debug("GEMS_KUNLUNXIN CONV_TRANSPOSE2D")

    from flag_gems.ops.conv_transpose2d import (
        _unsupported_conv_transpose2d,
        _validate_conv_transpose2d_args,
    )

    def _pair2(v):
        if isinstance(v, (list, tuple)):
            if len(v) == 1:
                return int(v[0]), int(v[0])
            if len(v) != 2:
                raise RuntimeError("expected a single int or a pair of ints")
            return int(v[0]), int(v[1])
        return v, v

    stride_h, stride_w = _pair2(stride)
    padding_h, padding_w = _pair2(padding)
    output_padding_h, output_padding_w = _pair2(output_padding)
    dilation_h, dilation_w = _pair2(dilation)

    input_was_unbatched = input.dim() == 3
    if input_was_unbatched:
        input = input.unsqueeze(0)

    if not input.is_contiguous():
        input = input.contiguous()
    if not weight.is_contiguous():
        weight = weight.contiguous()
    if bias is not None and not bias.is_contiguous():
        bias = bias.contiguous()

    if not _validate_conv_transpose2d_args(
        input,
        weight,
        bias,
        stride_h,
        stride_w,
        padding_h,
        padding_w,
        output_padding_h,
        output_padding_w,
        groups,
        dilation_h,
        dilation_w,
    ):
        _unsupported_conv_transpose2d(
            input,
            weight,
            bias,
            stride_h,
            stride_w,
            padding_h,
            padding_w,
            output_padding_h,
            output_padding_w,
            groups,
            dilation_h,
            dilation_w,
        )

    orig_dtype = input.dtype

    n, c, h, w = input.shape
    kh, kw = weight.shape[2], weight.shape[3]
    out_per_group = weight.shape[1]
    out_c = out_per_group * groups
    channels_per_group = c // groups

    out_h = (
        (h - 1) * stride_h
        - 2 * padding_h
        + dilation_h * (kh - 1)
        + output_padding_h
        + 1
    )
    out_w = (
        (w - 1) * stride_w
        - 2 * padding_w
        + dilation_w * (kw - 1)
        + output_padding_w
        + 1
    )

    out = torch.empty((n, out_c, out_h, out_w), device=input.device, dtype=orig_dtype)

    total = n * out_c * out_h * out_w
    if total > 0:
        has_bias = 0 if bias is None else 1
        if bias is None:
            bias_arg = out
        else:
            bias_arg = bias
        BLOCK = 32
        grid = (triton.cdiv(total, BLOCK),)
        conv_transpose2d_forward_kernel[grid](
            input,
            weight,
            out,
            bias_arg,
            n,
            h,
            w,
            out_c,
            out_h,
            out_w,
            input.stride(0),
            input.stride(1),
            input.stride(2),
            input.stride(3),
            weight.stride(0),
            weight.stride(1),
            weight.stride(2),
            weight.stride(3),
            out.stride(0),
            out.stride(1),
            out.stride(2),
            out.stride(3),
            out_per_group,
            kh,
            kw,
            stride_h,
            stride_w,
            padding_h,
            padding_w,
            dilation_h,
            dilation_w,
            groups,
            has_bias,
            CPG=channels_per_group,
            BLOCK=BLOCK,
        )

    if input_was_unbatched:
        out = out.squeeze(0)
    return out
