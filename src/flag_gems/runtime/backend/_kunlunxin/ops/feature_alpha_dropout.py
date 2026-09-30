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

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils.random_utils import (
    philox_backend_seed_offset,
    uint_to_uniform_float,
)

logger = logging.getLogger("flag_gems.ops.feature_alpha_dropout")

_ALPHA = 1.7580993408473766
_TILE_FLOOR = 1024
_CAP = 16384
_FLAT_BLOCK = 8192

@triton.jit
def feature_alpha_dropout_signed_zero_kernel(
    in_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    active = offsets < n_elements
    value = tl.load(in_ptr + offsets, mask=active)
    value_f32 = value.to(tl.float32)
    bits = value_f32.to(tl.uint32, bitcast=True)
    sign = (bits & 0x80000000).to(tl.uint32)
    zero_signed = sign.to(tl.float32, bitcast=True)
    product = value_f32 * 0.0
    finite = product == product
    out = tl.where(finite, zero_signed, float("nan"))
    tl.store(out_ptr + offsets, out.to(value.dtype), mask=active)


@libentry()
@triton.jit(do_not_specialize=["p", "scale", "shift", "dropped_value", "seed", "offset"])
def feature_alpha_dropout_tiled_kernel(
    input_ptr,
    output_ptr,
    spatial_size,
    p,
    scale,
    shift,
    dropped_value,
    seed,
    offset,
    TILE: tl.constexpr,
    NEED_MASK: tl.constexpr,
):
    feature = tl.program_id(0)
    tile = tl.program_id(1)
    seed = seed.to(tl.int64)
    offset = offset.to(tl.int64)
    c0 = (offset & 0xFFFFFFFF).to(tl.uint32) + feature.to(tl.uint32)
    c1 = ((offset >> 32) & 0xFFFFFFFF).to(tl.uint32)
    zero = c0 * 0
    random, _, _, _ = tl.philox(seed, c0, c1, zero, zero)
    keep = uint_to_uniform_float(random) > p
    local = tile * TILE + tl.arange(0, TILE)
    ptr = feature * spatial_size + local
    if NEED_MASK:
        active = local < spatial_size
        value = tl.load(input_ptr + ptr, mask=active)
        out = tl.where(keep, value * scale + shift, dropped_value)
        tl.store(output_ptr + ptr, out, mask=active)
    else:
        value = tl.load(input_ptr + ptr)
        out = tl.where(keep, value * scale + shift, dropped_value)
        tl.store(output_ptr + ptr, out)
@libentry()
@triton.jit(do_not_specialize=["p", "scale", "shift", "dropped_value", "seed", "offset"])
def feature_alpha_dropout_flat_kernel(
    input_ptr,
    output_ptr,
    n_elements,
    p,
    scale,
    shift,
    dropped_value,
    seed,
    offset,
    LOG2_SPATIAL: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    gid = pid * BLOCK + tl.arange(0, BLOCK)
    active = gid < n_elements
    feature = gid >> LOG2_SPATIAL
    seed = seed.to(tl.int64)
    offset = offset.to(tl.int64)
    c0 = (offset & 0xFFFFFFFF).to(tl.uint32) + feature.to(tl.uint32)
    c1 = ((offset >> 32) & 0xFFFFFFFF).to(tl.uint32)
    zero = (feature * 0).to(tl.uint32)
    random, _, _, _ = tl.philox(seed, c0, c1, zero, zero)
    keep = uint_to_uniform_float(random) > p
    value = tl.load(input_ptr + gid, mask=active)
    out = tl.where(keep, value * scale + shift, dropped_value)
    tl.store(output_ptr + gid, out, mask=active)


def _affine(p):
    scale = 1.0 / math.sqrt((_ALPHA * _ALPHA * p + 1.0) * (1.0 - p))
    shift = _ALPHA * scale * p
    dropped_value = _ALPHA * scale * (p - 1.0)
    return scale, shift, dropped_value


def _choose_tile(spatial_size):
    if spatial_size >= _CAP:
        ntile = triton.cdiv(spatial_size, _CAP)
        need = (spatial_size % _CAP) != 0
        return _CAP, ntile, need
    tile = 1 << (spatial_size - 1).bit_length()
    tile = max(tile, _TILE_FLOOR)
    need = (spatial_size % tile) != 0
    ntile = triton.cdiv(spatial_size, tile)
    return tile, ntile, need


def _run_apply(work, output, p):
    n_elements = work.numel()
    n_features = work.shape[0] * work.shape[1]
    spatial_size = n_elements // n_features
    scale, shift, dropped_value = _affine(p)
    is_pow2 = (spatial_size & (spatial_size - 1)) == 0
    with torch_device_fn.device(work.device):
        philox_seed, philox_offset = philox_backend_seed_offset(n_features)
        if is_pow2 and _FLAT_BLOCK <= n_elements and spatial_size < _FLAT_BLOCK:
            log2_spatial = spatial_size.bit_length() - 1
            grid = (triton.cdiv(n_elements, _FLAT_BLOCK),)
            feature_alpha_dropout_flat_kernel[grid](
                work, output, n_elements, p, scale, shift, dropped_value,
                philox_seed, philox_offset,
                LOG2_SPATIAL=log2_spatial, BLOCK=_FLAT_BLOCK,
            )
        else:
            tile, ntile, need = _choose_tile(spatial_size)
            grid = (n_features, ntile)
            feature_alpha_dropout_tiled_kernel[grid](
                work, output, spatial_size, p, scale, shift, dropped_value,
                philox_seed, philox_offset,
                TILE=tile, NEED_MASK=need,
            )


def _signed_zero(work, output):
    n_elements = work.numel()
    grid = (triton.cdiv(n_elements, _TILE_FLOOR),)
    with torch_device_fn.device(work.device):
        feature_alpha_dropout_signed_zero_kernel[grid](
            work, output, n_elements, BLOCK_SIZE=_TILE_FLOOR
        )
def feature_alpha_dropout(input, p=0.5, train=True):
    logger.debug("GEMS_KUNLUNXIN FEATURE_ALPHA_DROPOUT FORWARD")

    if not (0.0 <= p <= 1.0):
        raise RuntimeError(
            f"dropout probability has to be between 0 and 1, but got {p}"
        )
    if p == 0.0 or not train or input.numel() == 0:
        return input
    if p == 1.0:
        original_input = input
        work = input.contiguous()
        output = torch.empty_like(work)
        _signed_zero(work, output)
        output = output.view(input.shape)
        if original_input.is_contiguous():
            return output
        result = torch.empty_like(original_input)
        result.copy_(output)
        return result
    if input.ndim < 2:
        raise RuntimeError(
            "Feature dropout requires at least 2 dimensions in the input"
        )
    if not input.dtype.is_floating_point:
        raise RuntimeError(
            "feature_alpha_dropout only supports floating-point inputs"
        )

    original_input = input
    work = input.contiguous()
    output = torch.empty_like(work)
    _run_apply(work, output, p)

    if original_input.is_contiguous():
        return output
    result = torch.empty_like(original_input)
    result.copy_(output)
    return result


def feature_alpha_dropout_(input, p=0.5, train=True):
    logger.debug("GEMS_KUNLUNXIN FEATURE_ALPHA_DROPOUT_ INPLACE FORWARD")

    if not (0.0 <= p <= 1.0):
        raise RuntimeError(
            f"dropout probability has to be between 0 and 1, but got {p}"
        )
    if p == 0.0 or not train or input.numel() == 0:
        return input
    if p == 1.0:
        work = input.contiguous()
        _signed_zero(work, work)
        if work is not input:
            input.copy_(work)
        return input
    if input.ndim < 2:
        raise RuntimeError(
            "Feature dropout requires at least 2 dimensions in the input"
        )
    if not input.dtype.is_floating_point:
        raise RuntimeError(
            "feature_alpha_dropout_ only supports floating-point inputs"
        )

    work = input if input.is_contiguous() else input.contiguous()
    _run_apply(work, work, p)

    if work is not input:
        input.copy_(work)
    return input
