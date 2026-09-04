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

"""Kunlunxin (XPU) index_fill / index_fill_ override.

The generic implementation in ``flag_gems.ops.index_fill`` decomposes the fill
into a flat index ``m = outer * index_len + k`` and derives ``(outer, k)`` with
a per-lane ``//``/``%`` pair.  Integer division is software-emulated on XPU, so
the pure-scatter case (``dim`` is the last dim, inner_size == 1; e.g. a
full-length index on an ``(N, M)`` tensor) spends most of its time in that
emulation: measured 937 ms for 16.7M scattered stores on ``(4096, 4096)``
dim=1, i.e. ~2-130x slower than the vendor reference.

This override maps ``(index, outer)`` onto the two grid axes instead, so the
kernel needs no division at all, and only specializes the inner_size == 1
case: 128 x 8 tiles measure 17-33x faster than the flat-m version on the
benchmark matrix.  All other layouts (inner_size > 1, strided, empty) keep the
dispatched generic Triton implementation; in particular no native/CPU
fallback is used.
"""

import torch
import triton
import triton.language as tl

from flag_gems.ops.index_fill import (
    index_fill as _generic_index_fill,
    index_fill_ as _generic_index_fill_,
    _native_clone,
    _native_copy_,
    _prepare_index,
    _prepare_tensor_value,
)
from flag_gems.utils import libentry

_SCATTER_BLOCK_K = 128
_SCATTER_BLOCK_O = 8


@libentry()
@triton.jit
def index_fill_scatter_kernel(
    out,
    index,
    value,
    index_len,
    dim_size,
    outer_size,
    VALUE_IS_TENSOR: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_O: tl.constexpr,
):
    # Pure-scatter path (dim is the last dim, inner_size == 1): fill
    # out[o, index[k]] = value for all (o, k).  The two grid axes map
    # directly to (index position, outer position), so no integer division
    # is needed; BLOCK_K x BLOCK_O gives 1024 outstanding stores per program.
    pid_k = tl.program_id(0)
    pid_o = tl.program_id(1)
    k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    o = pid_o * BLOCK_O + tl.arange(0, BLOCK_O)
    k_mask = k < index_len
    o_mask = o < outer_size
    # Clamp the load address into the index buffer: XPU's masked load ignores
    # `other=` on out-of-bounds lanes (reads adjacent memory instead), so a
    # raw `index + k` would leak garbage into `raw_index` for k >= index_len.
    # The store mask below still discards those lanes; the clamp only keeps
    # the read itself in bounds.
    load_k = tl.where(k_mask, k, 0)
    raw_index = tl.load(index + load_k, mask=k_mask, other=0).to(tl.int64)
    valid_index = (raw_index >= -dim_size) & (raw_index < dim_size)
    normalized_index = tl.where(raw_index < 0, raw_index + dim_size, raw_index)
    offsets = o[:, None].to(tl.int64) * dim_size + normalized_index[None, :]
    store_mask = o_mask[:, None] & (k_mask[None, :] & valid_index[None, :])
    if VALUE_IS_TENSOR:
        fill_value = tl.load(value)
    else:
        fill_value = value
    # Out-of-range index entries are skipped silently by the store mask.
    # (PyTorch reports them as an error, but tl.device_assert fails to
    # compile on non-CUDA FlagGems backends, so the check is omitted.)
    tl.store(out + offsets, fill_value, mask=store_mask)


def _index_fill_scatter_launch(out, index, value, value_is_tensor, dim_size):
    index_len = index.numel()
    outer_size = out.numel() // dim_size
    grid = (
        triton.cdiv(index_len, _SCATTER_BLOCK_K),
        triton.cdiv(outer_size, _SCATTER_BLOCK_O),
    )
    index_fill_scatter_kernel[grid](
        out,
        index,
        value,
        index_len,
        dim_size,
        outer_size,
        VALUE_IS_TENSOR=value_is_tensor,
        BLOCK_K=_SCATTER_BLOCK_K,
        BLOCK_O=_SCATTER_BLOCK_O,
    )


def index_fill(inp, dim, index, value):
    # Entry for both `index_fill.int_Scalar` and `index_fill.int_Tensor`: the
    # dispatcher routes by value type, so a 0-dimensional tensor value arrives
    # as a Tensor and a plain number as a Python scalar.
    dim, index = _prepare_index(inp, dim, index)
    if isinstance(value, torch.Tensor):
        value_is_tensor, value = _prepare_tensor_value(inp, value)
    else:
        value_is_tensor = False
    if inp.numel() == 0 or index.numel() == 0:
        return _native_clone(inp)
    if inp.is_contiguous() and dim == inp.ndim - 1:
        out = torch.empty_like(inp)
        _native_copy_(out, inp)
        _index_fill_scatter_launch(out, index, value, value_is_tensor, inp.size(dim))
        return out
    return _generic_index_fill(inp, dim, index, value)


def index_fill_(inp, dim, index, value):
    # Entry for both `index_fill_.int_Scalar` and `index_fill_.int_Tensor`.
    dim, index = _prepare_index(inp, dim, index)
    if isinstance(value, torch.Tensor):
        value_is_tensor, value = _prepare_tensor_value(inp, value)
    else:
        value_is_tensor = False
    if inp.numel() == 0 or index.numel() == 0:
        return inp
    if inp.is_contiguous() and dim == inp.ndim - 1:
        _index_fill_scatter_launch(inp, index, value, value_is_tensor, inp.size(dim))
        return inp
    return _generic_index_fill_(inp, dim, index, value)