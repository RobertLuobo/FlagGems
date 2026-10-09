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

from flag_gems import runtime
from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as ext

logger = logging.getLogger(__name__)


@triton.jit
def _round_bf16(x):
    # Round fp32 -> bfloat16 (round-to-nearest-even) and return the result in
    # fp32. torch evaluates blackman_window op-by-op in the output dtype, so the
    # reference carries bf16 rounding after every elementary op. The XPU3 Triton
    # backend folds consecutive `.to(bf16).to(fp32)` casts away and keeps the
    # whole chain in fp32, which is numerically more accurate but no longer
    # bit-matches torch. Doing the round through integer bit ops cannot be
    # elided, so the intermediate rounding survives and we reproduce torch.
    b = x.to(tl.int32, bitcast=True)
    b = b + 0x7FFF + ((b >> 16) & 1)
    b = b & -65536  # 0xFFFF0000 as a signed int32
    return b.to(tl.float32, bitcast=True)


@libentry()
@triton.jit
def blackman_window_kernel(
    out_ptr,
    window_length,
    scale,
    LOW_PRECISION: tl.constexpr,
    IS_BF16: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = ext.program_id(0)
    idx = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = idx < window_length

    dtype = out_ptr.dtype.element_ty

    if LOW_PRECISION:
        if IS_BF16:
            n = _round_bf16(idx.to(tl.float32))
            x = _round_bf16(n * scale)
            c2 = _round_bf16(tl.cos(_round_bf16(x * 2.0)))
            c4 = _round_bf16(tl.cos(_round_bf16(x * 4.0)))
            t4 = _round_bf16(c4 * 0.08)
            t2 = _round_bf16(c2 * 0.5)
            term = _round_bf16(t4 - t2)
            val = (term + 0.42).to(dtype)
        else:
            n = idx.to(dtype).to(tl.float32)
            x = (n * scale).to(dtype).to(tl.float32)
            c2 = tl.cos((x * 2.0).to(dtype).to(tl.float32)).to(dtype).to(tl.float32)
            c4 = tl.cos((x * 4.0).to(dtype).to(tl.float32)).to(dtype).to(tl.float32)
            t4 = (c4 * 0.08).to(dtype).to(tl.float32)
            t2 = (c2 * 0.5).to(dtype).to(tl.float32)
            term = (t4 - t2).to(dtype).to(tl.float32)
            val = (term + 0.42).to(dtype)
    else:
        n = idx.to(dtype)
        x = n * scale
        val = 0.08 * tl.cos(x * 4.0) - 0.5 * tl.cos(x * 2.0) + 0.42
        val = val.to(dtype)

    tl.store(out_ptr + idx, val, mask=mask)


def blackman_window(
    window_length,
    periodic=True,
    *,
    dtype=None,
    layout=torch.strided,
    device=None,
    pin_memory=None,
):
    logger.debug("GEMS_KUNLUNXIN BLACKMAN_WINDOW")
    assert window_length >= 0, "window_length must be non-negative"

    if dtype is None:
        dtype = torch.get_default_dtype()
    assert dtype.is_floating_point, "blackman_window only supports floating point dtype"

    if layout != torch.strided:
        raise ValueError("Only strided layout is supported for blackman_window.")

    if device is None:
        device = torch.device(runtime.device.name)

    out = torch.empty(
        (window_length,),
        dtype=dtype,
        layout=layout,
        device=device,
        pin_memory=pin_memory,
    )

    if window_length == 0:
        return out
    if window_length == 1:
        return torch.fill(out, 1.0)

    full_length = window_length + 1 if periodic else window_length
    scale = math.pi / (full_length - 1)

    low_precision = dtype in (torch.float16, torch.bfloat16)
    is_bf16 = dtype == torch.bfloat16
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(window_length, BLOCK_SIZE),)
    blackman_window_kernel[grid](
        out, window_length, scale, low_precision, is_bf16, BLOCK_SIZE=BLOCK_SIZE
    )
    return out


def blackman_window_periodic(
    window_length,
    periodic,
    *,
    dtype=None,
    layout=torch.strided,
    device=None,
    pin_memory=None,
):
    return blackman_window(
        window_length,
        periodic,
        dtype=dtype,
        layout=layout,
        device=device,
        pin_memory=pin_memory,
    )
