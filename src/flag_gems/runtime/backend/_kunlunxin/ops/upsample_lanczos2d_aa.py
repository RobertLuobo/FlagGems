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

import triton
import triton.language as tl

from flag_gems.ops import upsample_lanczos2d_aa as _generic
from flag_gems.ops.upsample_lanczos2d_aa import _lanczos3

logger = logging.getLogger(__name__)


@triton.jit
def _lanczos_weights_kernel(
    weights,
    index_mins,
    index_sizes,
    input_size,
    output_size,
    SCALE: tl.constexpr,
    SUPPORT: tl.constexpr,
    INVSCALE: tl.constexpr,
    MAX_TAPS: tl.constexpr,
    IS_FP64: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < output_size
    compute_dtype: tl.constexpr = tl.float64 if IS_FP64 else tl.float32
    center = SCALE * (offsets.to(compute_dtype) + 0.5)
    index_min = tl.maximum((center - SUPPORT + 0.5).to(tl.int64), 0).to(tl.int32)
    index_size = tl.minimum((center + SUPPORT + 0.5).to(tl.int64), input_size).to(
        tl.int32
    )
    index_size = tl.minimum(tl.maximum(index_size - index_min, 0), MAX_TAPS)
    total_weight = tl.zeros((BLOCK_SIZE,), dtype=compute_dtype)

    for tap in tl.static_range(MAX_TAPS):
        tap_mask = mask & (tap < index_size)
        weight = _lanczos3((tap + index_min - center + 0.5) * INVSCALE)
        weight = tl.where(tap_mask, weight, 0.0)
        tl.store(weights + offsets * MAX_TAPS + tap, weight, mask=mask)
        total_weight += weight

    total_weight = tl.where(total_weight != 0.0, total_weight, 1.0)
    for tap in tl.static_range(MAX_TAPS):
        weight = tl.load(weights + offsets * MAX_TAPS + tap, mask=mask)
        tl.store(
            weights + offsets * MAX_TAPS + tap,
            weight / total_weight,
            mask=mask,
        )
    tl.store(index_mins + offsets, index_min.to(tl.int64), mask=mask)
    tl.store(index_sizes + offsets, index_size.to(tl.int64), mask=mask)


@triton.jit
def _lanczos_horizontal_kernel(
    input,
    output,
    weights,
    index_mins,
    index_sizes,
    numel,
    input_w,
    output_w,
    SCALE: tl.constexpr,
    SUPPORT: tl.constexpr,
    INVSCALE: tl.constexpr,
    MAX_TAPS: tl.constexpr,
    IS_UINT8: tl.constexpr,
    IS_FP64: tl.constexpr,
    PRECOMPUTED: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < numel
    output_x = offsets % output_w
    row = offsets // output_w
    compute_dtype: tl.constexpr = tl.float64 if IS_FP64 else tl.float32
    if PRECOMPUTED:
        index_min = tl.load(index_mins + output_x, mask=mask).to(tl.int32)
        index_size = tl.load(index_sizes + output_x, mask=mask).to(tl.int32)
    else:
        center = SCALE * (output_x.to(compute_dtype) + 0.5)
        index_min = tl.maximum((center - SUPPORT + 0.5).to(tl.int64), 0).to(tl.int32)
        index_size = tl.minimum(
            (center + SUPPORT + 0.5).to(tl.int64), input_w
        ).to(tl.int32)
        index_size = tl.minimum(tl.maximum(index_size - index_min, 0), MAX_TAPS)
    total_weight = tl.zeros((BLOCK_SIZE,), dtype=compute_dtype)
    value = tl.zeros((BLOCK_SIZE,), dtype=compute_dtype)
    for tap in tl.static_range(MAX_TAPS):
        tap_mask = mask & (tap < index_size)
        if PRECOMPUTED:
            weight = tl.load(
                weights + output_x * MAX_TAPS + tap,
                mask=tap_mask,
                other=0.0,
            )
        else:
            weight = _lanczos3((tap + index_min - center + 0.5) * INVSCALE)
        sample = tl.load(
            input + row * input_w + index_min + tap,
            mask=tap_mask,
            other=0.0,
        ).to(compute_dtype)
        weight = tl.where(tap_mask, weight, 0.0)
        sample = tl.where(tap_mask, sample, 0.0)
        value += sample * weight
        total_weight += weight

    value /= tl.where(total_weight != 0.0, total_weight, 1.0)
    if IS_UINT8:
        value = tl.floor(tl.minimum(tl.maximum(value, 0.0), 255.0) + 0.5)
    tl.store(output + offsets, value, mask=mask)


@triton.jit
def _lanczos_vertical_kernel(
    input,
    output,
    weights,
    index_mins,
    index_sizes,
    numel,
    input_h,
    output_h,
    output_w,
    SCALE: tl.constexpr,
    SUPPORT: tl.constexpr,
    INVSCALE: tl.constexpr,
    MAX_TAPS: tl.constexpr,
    IS_UINT8: tl.constexpr,
    IS_FP64: tl.constexpr,
    PRECOMPUTED: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < numel
    output_x = offsets % output_w
    output_y = (offsets // output_w) % output_h
    nc = offsets // (output_h * output_w)
    compute_dtype: tl.constexpr = tl.float64 if IS_FP64 else tl.float32
    if PRECOMPUTED:
        index_min = tl.load(index_mins + output_y, mask=mask).to(tl.int32)
        index_size = tl.load(index_sizes + output_y, mask=mask).to(tl.int32)
    else:
        center = SCALE * (output_y.to(compute_dtype) + 0.5)
        index_min = tl.maximum((center - SUPPORT + 0.5).to(tl.int64), 0).to(tl.int32)
        index_size = tl.minimum(
            (center + SUPPORT + 0.5).to(tl.int64), input_h
        ).to(tl.int32)
        index_size = tl.minimum(tl.maximum(index_size - index_min, 0), MAX_TAPS)
    total_weight = tl.zeros((BLOCK_SIZE,), dtype=compute_dtype)
    value = tl.zeros((BLOCK_SIZE,), dtype=compute_dtype)
    for tap in tl.static_range(MAX_TAPS):
        tap_mask = mask & (tap < index_size)
        if PRECOMPUTED:
            weight = tl.load(
                weights + output_y * MAX_TAPS + tap,
                mask=tap_mask,
                other=0.0,
            )
        else:
            weight = _lanczos3((tap + index_min - center + 0.5) * INVSCALE)
        input_offset = (nc * input_h + index_min + tap) * output_w + output_x
        sample = tl.load(input + input_offset, mask=tap_mask, other=0.0).to(
            compute_dtype
        )
        weight = tl.where(tap_mask, weight, 0.0)
        sample = tl.where(tap_mask, sample, 0.0)
        value += sample * weight
        total_weight += weight

    value /= tl.where(total_weight != 0.0, total_weight, 1.0)
    if IS_UINT8:
        value = tl.floor(tl.minimum(tl.maximum(value, 0.0), 255.0) + 0.5)
    tl.store(output + offsets, value, mask=mask)


_generic._lanczos_weights_kernel = _lanczos_weights_kernel
_generic._lanczos_horizontal_kernel = _lanczos_horizontal_kernel
_generic._lanczos_vertical_kernel = _lanczos_vertical_kernel


def _upsample_lanczos2d_aa(
    input, output_size, align_corners=False, scales_h=None, scales_w=None
):
    logger.debug("GEMS_KUNLUNXIN UPSAMPLE_LANCZOS2D_AA")
    logger.debug("GEMS UPSAMPLE LANCZOS2D AA")
    return _generic._upsample_lanczos2d_aa(
        input, output_size, align_corners, scales_h, scales_w
    )


def _upsample_lanczos2d_aa_out(
    input, output_size, align_corners=False, scales_h=None, scales_w=None, *, out
):
    logger.debug("GEMS_KUNLUNXIN UPSAMPLE_LANCZOS2D_AA_OUT")
    return _generic._upsample_lanczos2d_aa_out(
        input, output_size, align_corners, scales_h, scales_w, out=out
    )


def _upsample_lanczos2d_aa_vec(input, output_size, align_corners, scale_factors):
    logger.debug("GEMS_KUNLUNXIN UPSAMPLE_LANCZOS2D_AA_VEC")
    return _generic._upsample_lanczos2d_aa_vec(
        input, output_size, align_corners, scale_factors
    )
