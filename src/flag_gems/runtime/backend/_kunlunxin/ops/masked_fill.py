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
import os

import torch
import triton
import triton.language as tl
from _kunlunxin.utils.codegen_config_utils import CodeGenConfig
from flag_gems.runtime import torch_device_fn

from flag_gems.utils import broadcastable_to

from ..utils.pointwise_dynamic import pointwise_dynamic

logger = logging.getLogger("flag_gems").getChild(__name__.lstrip("."))

# ---------------------------------------------------------------------------
# Generic pointwise path (fallback for non-contiguous inputs, broadcastable
# masks, tensor values and tiny shapes). Tuned on XPU: isCloseVectorization
# keeps the mixed i1-mask/dtype tl.where vectorized.
_config = CodeGenConfig(
    512,
    (65536, 65536, 65536),
    32,
    True,
    prefer_1d_tile=True,
    buffer_size_limit=4096,
    isCloseVectorization=True,
    kunlunAutoGrid=True,
    unroll_num=8,
)


@pointwise_dynamic(
    is_tensor=[True, True, False],
    promotion_methods=[(0, "NO_OPMATH")],
    config=_config,
)
@triton.jit
def masked_fill_kernel(inp, expand_mask, value):
    return tl.where(expand_mask, value, inp)


@pointwise_dynamic(
    is_tensor=[True, True, True],
    promotion_methods=[(0, "NO_OPMATH")],
    config=_config,
)
@triton.jit
def masked_fill_tensor_value_kernel(inp, expand_mask, value):
    return tl.where(expand_mask, value, inp)


# ---------------------------------------------------------------------------
# Flat fast path (scalar value, contiguous fp16/bf16/fp32, numel >= gate).
#
# XPU 6 probe (2026-08-14, 268435456-elem shapes): the bottleneck is NOT the
# bool mask read (mask-reduced to bytes reads fine; a no-select kernel reading
# the mask == plain copy) and NOT the allocation. It is the per-lane `sel`
# that `tl.where(bool_mask, value, x)` lowers to: measured ~5x the fp32 copy
# time for the same traffic, and the select-with-splat is even worse. The
# cheapest exact form found is a select in the integer view of the data,
#   r = xi + (V - xi) * m                (m = mask byte 0/1 -> int view)
# which is bit-identical to where(mask, V(pattern), x) with 2-3x less per
# lane work: measured +1.6x fp16 / +1.3x fp32 over the pointwise path on all
# mid/large benchmark shapes, no regression on small shapes (gated below
# _FAST_MIN_NUMEL where the pointwise path stays).
_FAST_TILE = 131072
_FAST_MIN_NUMEL = 1 << 20
_FAST_DTYPES = (torch.float16, torch.bfloat16, torch.float32)


@triton.jit
def masked_fill_fast_kernel(
    out_ptr, x_ptr, mask_ptr, V: tl.constexpr, TILE: tl.constexpr
):
    pid = tl.program_id(0)
    tid = pid * TILE + tl.arange(0, TILE)
    xi = tl.load(x_ptr + tid)
    m = tl.load(mask_ptr + tid).to(xi.dtype)
    r = xi + (V - xi) * m
    tl.store(out_ptr + tid, r)


@triton.jit
def masked_fill_fast_masked_kernel(
    out_ptr, x_ptr, mask_ptr, numel, V: tl.constexpr, TILE: tl.constexpr
):
    pid = tl.program_id(0)
    tid = pid * TILE + tl.arange(0, TILE)
    m0 = tid < numel
    xi = tl.load(x_ptr + tid, mask=m0)
    m = tl.load(mask_ptr + tid, mask=m0).to(xi.dtype)
    r = xi + (V - xi) * m
    tl.store(out_ptr + tid, r, mask=m0)


def _fast_bits(value, dtype):
    iview = torch.int16 if dtype in (torch.float16, torch.bfloat16) else torch.int32
    return int(torch.tensor([value], dtype=dtype).view(iview).item())


def _masked_fill_fast(inp, mask, value, out):
    n = inp.numel()
    bits = _fast_bits(value, inp.dtype)
    xi = inp.view(
        torch.int16 if inp.dtype in (torch.float16, torch.bfloat16) else torch.int32
    )
    oi = out.view(xi.dtype)
    mask8 = mask.view(torch.int8)
    launch = dict(
        num_warps=4,
        buffer_size_limit=8192,
        unroll_num=16,
        isCloseMemoryAsync=False,
    )
    if n % _FAST_TILE == 0:
        masked_fill_fast_kernel[(n // _FAST_TILE,)](
            oi, xi, mask8, V=bits, TILE=_FAST_TILE, **launch
        )
    else:
        masked_fill_fast_masked_kernel[(math.ceil(n / _FAST_TILE),)](
            oi, xi, mask8, n, V=bits, TILE=_FAST_TILE, **launch
        )
    return out


# ---------------------------------------------------------------------------
# tle.raw fast path for the large-shape in-place masked_fill_ (P800 xpu3,
# cluster C payload in masked_fill_raw.xpu).
#
# Why a raw payload: the compile-time select of the pointwise kernel lowers to
# a SCALAR-predicated select on this backend (~5x the fp32 copy time), and the
# integer-view mul-add of the flat fast path is still ~4x a pure copy because
# every per-lane op is scalarized (no SIMD ALU). The hand-written payload
# drives per-core GM2LM/LM2GM DMA with the same footprint as ATen and does the
# per-lane select with a single masked-HOLD bitwise and:
#   vvand_*_mh(y, ONES, x, m) == m ? y : x
# (bit-exact for every value including -0.0, inf and signaling NaNs), with the
# bool bytes expanded to lane masks by the hardware u8->f32 convert +
# compare (vfix82float_* + vvneq) -- the same recipe that makes
# nan_to_num_ reach ~2TB/s while the lt/ne-family payloads (which pay an
# expand4 multiply) only reach ~1.3TB/s.
#
# In-place only: the payload writes into A (out == in); the out-of-place
# `masked_fill` would need 3 GM streams against the pointwise kernel's 2 and
# keeps the existing fast path.
try:
    import triton.experimental.tle as tle

    _TLE_OK = True
except ImportError:
    tle = None
    _TLE_OK = False

_HERE = os.path.dirname(os.path.abspath(__file__))
_NCLUSTER = 12  # P800 (xpu3): one Triton program == one cluster of 64 cores
# Payload scalars are i32 (do_not_specialize); guard the byte range.
_RAW_MAX_ELEMS = 2**31 - 1
# Must match CHUNK_BYTES in masked_fill_raw.xpu (the chunk-grid partition
# contract). 1280B is the largest 64B-aligned chunk that keeps 2 input + 2
# output + 2 mask buffers (5*C bytes) under the compiler's local-memory
# budget (nan_to_num proves 2+2*1792 = 7168B).
_RAW_CHUNK_BYTES = 1280

_RAW_TYPE_CODE = {
    torch.float32: 0,
    torch.float16: 1,
    torch.bfloat16: 2,
}

if _TLE_OK:

    @tle.raw.dialect("xpu3", file=os.path.join(_HERE, "masked_fill_raw.xpu"))
    def masked_fill_raw_(in_, mask, numel, esz, type_code, value_bits,
                         chunk_start, chunk_count):
        ...

    @triton.jit(do_not_specialize=[
        "numel", "esz", "type_code", "value_bits", "chunk_count",
    ])
    def masked_fill_raw_kernel(In, Mask, numel, esz, type_code, value_bits,
                               chunk_count):
        pid = tl.program_id(0)
        tle.raw.call(masked_fill_raw_, (In, Mask, numel, esz, type_code,
                                        value_bits, pid * chunk_count,
                                        chunk_count))


def _raw_masked_fill_(inp, mask, value):
    """In-place masked_fill_ via the raw payload, or None when it does not
    apply (non-contiguous / broadcast mask / unsupported dtype / empty)."""
    if not _TLE_OK or not inp.is_contiguous() or not mask.is_contiguous():
        return None
    type_code = _RAW_TYPE_CODE.get(inp.dtype)
    if type_code is None:
        return None
    if tuple(mask.shape) != tuple(inp.shape):
        return None
    M = inp.numel()
    if M == 0 or M > _RAW_MAX_ELEMS:
        return None
    value_bits = _fast_bits(value, inp.dtype)
    esz = inp.element_size()
    # partition by payload chunks (CHUNK_BYTES/esz elements each): every
    # program and core gets whole chunks so all GM2LM/LM2GM transfers are
    # CHUNK_BYTES-aligned in global memory.
    chunk_elems = _RAW_CHUNK_BYTES // esz
    total_chunks = (M + chunk_elems - 1) // chunk_elems
    per = (total_chunks + _NCLUSTER - 1) // _NCLUSTER
    with torch_device_fn.device(inp.device):
        masked_fill_raw_kernel[(_NCLUSTER,)](
            inp.view(torch.uint8), mask.view(torch.uint8), M, esz, type_code,
            value_bits, per)
    return inp


# At this element count the payload launch overhead (12 programs, chunk-grid
# arithmetic host side) is amortized and it beats the flat-fast kernel (whose
# per-lane int mul-add is scalarized on XPU); small shapes keep the
# pointwise path, which is already bit-exact.
_RAW_MIN_ELEMS = 65536


def _use_fast_path(inp, mask, value):
    if torch.is_tensor(value):
        return False
    if inp.dtype not in _FAST_DTYPES:
        return False
    if not (inp.is_contiguous() and mask.is_contiguous()):
        return False
    if tuple(mask.shape) != tuple(inp.shape):
        return False
    return inp.numel() >= _FAST_MIN_NUMEL


def masked_fill(inp, mask, value):
    logger.debug("GEMS_KUNLUNXIN MASKED_FILL")
    assert (
        (torch.is_tensor(value) and value.ndim == 0)
        or isinstance(value, int)
        or isinstance(value, float)
    ), "masked_fill_ only supports a 0-dimensional value tensor"
    if torch.is_tensor(value):
        if value.device != inp.device:
            raise RuntimeError("masked_fill value must be on the input device")
        kernel = masked_fill_tensor_value_kernel
    else:
        kernel = masked_fill_kernel
    assert broadcastable_to(
        mask.shape, inp.shape
    ), "The shape of mask must be broadcastable with the shape of the underlying tensor"

    if inp.ndim == 0:
        out = torch.empty_like(inp)
        kernel(inp, mask, value, out0=out)
        return out

    out = torch.empty_like(inp, dtype=inp.dtype, device=inp.device)
    if inp.numel() == 0:
        return out

    if _use_fast_path(inp, mask, value):
        return _masked_fill_fast(inp, mask, value, out)

    if inp.is_contiguous() and tuple(mask.shape) == tuple(inp.shape):
        # Common case (mask matches inp): one flat stride-1 pass, which is
        # what the tuned 1D config accelerates.
        mask = mask.contiguous()
        kernel(inp.view(-1), mask.view(-1), value, out0=out.view(-1))
    else:
        expand_mask = mask.expand(inp.shape)
        kernel.instantiate(inp.ndim)
        kernel(inp, expand_mask, value, out0=out)
    return out


def masked_fill_(inp, mask, value):
    logger.debug("GEMS_KUNLUNXIN MASKED_FILL_")
    assert (
        (torch.is_tensor(value) and value.ndim == 0)
        or isinstance(value, int)
        or isinstance(value, float)
    ), "masked_fill_ only supports a 0-dimensional value tensor"
    if torch.is_tensor(value):
        if value.device != inp.device:
            raise RuntimeError("masked_fill value must be on the input device")
        kernel = masked_fill_tensor_value_kernel
    else:
        kernel = masked_fill_kernel
    assert broadcastable_to(
        mask.shape, inp.shape
    ), "The shape of mask must be broadcastable with the shape of the underlying tensor"

    if inp.ndim == 0:
        kernel(inp, mask, value, out0=inp)
        return inp

    if inp.numel() == 0:
        return inp

    if _use_fast_path(inp, mask, value):
        if inp.numel() >= _RAW_MIN_ELEMS:
            raw_out = _raw_masked_fill_(inp, mask, value)
            if raw_out is not None:
                return raw_out
        return _masked_fill_fast(inp, mask, value, inp)

    if inp.is_contiguous() and tuple(mask.shape) == tuple(inp.shape):
        mask = mask.contiguous()
        kernel(inp.view(-1), mask.view(-1), value, out0=inp.view(-1))
    else:
        expand_mask = mask.expand(inp.shape)
        kernel.instantiate(inp.ndim)
        kernel(inp, expand_mask, value, out0=inp)
    return inp
