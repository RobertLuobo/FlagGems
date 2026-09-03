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
from typing import Optional

import torch
import triton
import triton.language as tl

from ..utils.codegen_config_utils import CodeGenConfig
from ..utils.pointwise_dynamic import pointwise_dynamic

logger = logging.getLogger(__name__)

# NOTE: config_ intentionally keeps the original is_scatter_slice=True +
# buffer_size_limit=512 + isCloseOffsetAnalysis=True for the STRIDED
# copy_slice path.  Experimenting with the copy-family recipe
# (buffer_size_limit=4096, isCloseVectorization=False, kunlunAutoGrid=True,
# unroll_num=8) here made large rank-4 strided copies (e.g. contiguous() of a
# 16M-element [::2] view) pathological: test_contiguous
# [dtype3-shape3] went from 15.4s (pre-fix) to >900s (timeout), because the
# 1d-tile codegen then widens to next_pow2(numel/12) lanes for the strided
# gather.  All large-contiguous speedups below come from _copy_flat_kernel,
# which is config-independent, so the strided path keeps the proven original
# config.
config_ = CodeGenConfig(
    512,
    (65536, 65536, 65536),
    32,
    True,
    prefer_1d_tile=True,
    is_scatter_slice=True,
)


# @pointwise_dynamic(is_tensor=(True,), promotion_methods=[(0, "DEFAULT")])
# @triton.jit
# def copy(src):
#     return src


@pointwise_dynamic(
    is_tensor=(True,), promotion_methods=[(0, "DEFAULT")], config=config_
)
@triton.jit
def copy_slice(src):
    return src


@pointwise_dynamic(is_tensor=[True], promotion_methods=[(0, "DEFAULT")])
@triton.jit
def _copy_kernel(src):
    return src


@triton.jit
def _copy_e8m0_to_fp32_kernel(src, dst, n_elements, BLOCK_SIZE: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    exponent_bits = tl.load(src + offsets, mask=mask).to(tl.uint32) << 23
    values = exponent_bits.to(tl.float32, bitcast=True)
    tl.store(dst + offsets, values, mask=mask)


@triton.jit
def _copy_flat_kernel(src_ptr, dst_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    """Bounded-tile flat block-DMA copy for contiguous same-dtype tensors.

    The pointwise codegen widens the 1d tile to next_pow2(numel/12) (12-CTAs
    "XPU BLOCK_NUM" partition), which measures 0.073-0.16ms on a 16M-element
    copy but 1.08ms on bool (i1 bytes cannot reuse the wide-tile path).  A
    fixed 65536-lane tile with a full grid keeps every access a contiguous
    block DMA and measures 0.061ms fp16 / 0.19ms bool on the same shape.
    """
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    tl.store(dst_ptr + offsets, tl.load(src_ptr + offsets, mask=mask), mask=mask)


def _is_e8m0(tensor: torch.Tensor) -> bool:
    return hasattr(torch, "float8_e8m0fnu") and tensor.dtype is torch.float8_e8m0fnu


def _validate_triton_copy(dst: torch.Tensor, src: torch.Tensor) -> None:
    if dst.layout != torch.strided or src.layout != torch.strided:
        raise NotImplementedError("copy_ only supports strided tensors on Kunlunxin")
    if dst.is_quantized or src.is_quantized:
        raise NotImplementedError(
            "copy_ for quantized tensors is not supported on Kunlunxin"
        )
    if src.is_complex() or dst.is_complex():
        raise NotImplementedError(
            "copy_ for complex tensors is not supported on Kunlunxin"
        )


def _expand_like(src: torch.Tensor, target_shape: torch.Size) -> torch.Tensor:
    if src.shape == target_shape:
        return src
    return src.expand(target_shape)


def copy(
    template: torch.Tensor, src: torch.Tensor, *, non_blocking: Optional[bool] = False
):
    logger.debug("GEMS_KUNLUNXIN COPY")
    out = torch.empty_strided(
        template.size(), template.stride(), dtype=template.dtype, device=template.device
    )
    copy_(out, src, non_blocking=bool(non_blocking))
    return out


def copy_(dst: torch.Tensor, src: torch.Tensor, non_blocking: bool = False):
    if not isinstance(src, torch.Tensor):
        raise TypeError("src must be a Tensor")

    # this is the same as PyTorch's check
    if dst._is_zerotensor():
        raise RuntimeError("ZeroTensors are immutable. Call clone() before copy_.")
    if src._is_zerotensor():
        return dst.zero_()

    aliases = torch._C._is_alias_of(dst, src)
    if aliases and (
        dst.storage_offset() == src.storage_offset()
        and dst.stride() == src.stride()
        and dst.size() == src.size()
        and dst.dtype == src.dtype
        and dst.device == src.device
        and dst.is_conj() == src.is_conj()
        and dst.is_neg() == src.is_neg()
    ):
        return dst

    if dst.device != src.device:
        raise NotImplementedError("copy_ across devices is not supported on Kunlunxin")

    _validate_triton_copy(dst, src)
    logger.debug("GEMS_KUNLUNXIN COPY_")

    try:
        broadcast_shape = torch.broadcast_shapes(dst.shape, src.shape)
    except RuntimeError as exc:
        raise RuntimeError(str(exc)) from exc

    if torch.Size(broadcast_shape) != dst.shape:
        raise RuntimeError(
            f"The broadcast shape {broadcast_shape} does not match destination shape {tuple(dst.shape)}"
        )
    if dst.numel() == 0:
        return dst

    expanded_src = _expand_like(src, dst.shape)
    if _is_e8m0(expanded_src):
        if _is_e8m0(dst):
            overload = _copy_kernel.instantiate(expanded_src.ndim)
            overload(expanded_src.view(torch.uint8), out0=dst.view(torch.uint8))
            return dst
        if (
            dst.dtype is torch.float32
            and expanded_src.is_contiguous()
            and dst.is_contiguous()
        ):
            block_size = 256
            _copy_e8m0_to_fp32_kernel[(triton.cdiv(expanded_src.numel(), block_size),)](
                expanded_src.view(torch.uint8),
                dst,
                expanded_src.numel(),
                BLOCK_SIZE=block_size,
            )
            return dst
        raise NotImplementedError(
            "copy_ from float8_e8m0fnu only supports float8_e8m0fnu and contiguous float32 destinations on Kunlunxin"
        )

    # Contiguous, same-dtype, non-aliasing copy: bounded-tile flat block-DMA is
    # measurably faster than both the multi-dim pointwise path (its per-lane
    # i0*s0 + i1*s1 gather + wide 12-CTA tile is ~2.8x slower and fails to
    # vectorize bool) and the 1d variant.  Non-contiguous / broadcast / mixed
    # dtype / aliasing copies keep the strided pointwise kernel below.
    if (
        not aliases
        and expanded_src.is_contiguous()
        and dst.is_contiguous()
        and expanded_src.dtype == dst.dtype
    ):
        block_size = 65536
        if expanded_src.dtype is torch.bool:
            # bool is stored as one byte; the XPU i1 load/store lowering is
            # byte-serial (~0.19ms for 16M elements) while a u8 view hits the
            # vectorized block-DMA path (~0.032ms, 6x).  Bit-exact by
            # construction (1 byte per element, no value reinterpretation).
            _copy_flat_kernel[(triton.cdiv(expanded_src.numel(), block_size),)](
                expanded_src.view(torch.uint8),
                dst.view(torch.uint8),
                expanded_src.numel(),
                BLOCK_SIZE=block_size,
                num_warps=32,
            )
        else:
            _copy_flat_kernel[(triton.cdiv(expanded_src.numel(), block_size),)](
                expanded_src,
                dst,
                expanded_src.numel(),
                BLOCK_SIZE=block_size,
                num_warps=32,
            )
        return dst

    overload = copy_slice.instantiate(expanded_src.ndim)
    if aliases:
        snapshot = torch.empty(dst.shape, dtype=src.dtype, device=src.device)
        overload(expanded_src, out0=snapshot)
        expanded_src = snapshot
    overload(expanded_src, out0=dst)
    return dst
