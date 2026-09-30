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

import builtins
import logging

import torch
import triton
import triton.language as tl
from torch import Tensor

from flag_gems.ops.nanquantile import (
    MAX_REDUCTION_SIZE,
    _validate_args,
)
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as ext

from .sort import sort_stable

logger = logging.getLogger(__name__)

_COUNT_BLOCK_N = 2048
_COUNT_BLOCK_M = 128
_COUNT_SMALL_M = 4096
_COUNT_HUGE_N = 32768
_COUNT_SMALL_BLOCK_M = 8


@libentry()
@triton.jit
def _nan_to_posinf_kernel(
    inp,
    out,
    n_elements,
    IS_F64: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Replace NaN with +inf so a plain ascending sort pushes every NaN to the
    # tail of each row. NaN is detected by integer bit compare because the
    # setup-stage unordered fp compare (`v != v`) crashes the XPU LLVM backend.
    offsets = ext.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    v = tl.load(inp + offsets, mask=mask, other=0.0)
    if IS_F64:
        b = v.to(tl.int64, bitcast=True)
        is_nan = (b & 0x7FFFFFFFFFFFFFFF) > 0x7FF0000000000000
    else:
        b = v.to(tl.int32, bitcast=True)
        is_nan = (b & 0x7FFFFFFF) > 0x7F800000
    v = tl.where(is_nan, float("inf"), v)
    tl.store(out + offsets, v, mask=mask)


@libentry()
@triton.jit
def _nan_count_rows_kernel(
    inp,
    out,
    M,
    N,
    IS_F64: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    # Per-row count of non-NaN elements via reduce-OUTSIDE: accumulate a
    # [BLOCK_M, BLOCK_N] int32 tile column-block by column-block, then a single
    # tl.sum(axis=1) after the loop (this axis-1 reduction shape is XPU-safe;
    # the crashing generic kernels use axis-0 reductions over 2D sort tiles).
    pid = ext.program_id(0)
    rows = pid * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]
    inp = inp + rows * N
    out = out + rows
    row_mask = rows < M
    _cnt = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.int32)
    for off in range(0, N, BLOCK_N):
        cols = off + tl.arange(0, BLOCK_N)[None, :]
        mask = row_mask and (cols < N)
        val = tl.load(inp + cols, mask, other=0.0)
        if IS_F64:
            b = val.to(tl.int64, bitcast=True)
            is_nan = (b & 0x7FFFFFFFFFFFFFFF) > 0x7FF0000000000000
        else:
            b = val.to(tl.int32, bitcast=True)
            is_nan = (b & 0x7FFFFFFF) > 0x7F800000
        valid = mask and (not is_nan)
        _cnt += valid.to(tl.int32)
    c = tl.sum(_cnt, axis=1)
    tl.store(out, c[:, None], row_mask)


def _block_qn(Q, N):
    # BLOCK_Q >= 256 miscompiles this gather on XPU3 (offset/vectorize codegen
    # bug: wrong indices -> reads into the +inf tail -> inf-inf NaNs), so cap
    # both tile dims at 128. A square tile also crashes the legalize pass, so
    # perturb one dim when they match (masking keeps the result correct).
    block_q = builtins.min(triton.next_power_of_2(Q), 128)
    block_n = builtins.min(triton.next_power_of_2(N), 128)
    if block_n == block_q:
        block_n = builtins.max(block_n // 2, 1)
    return block_q, block_n


@libentry()
@triton.jit
def _nanquantile_gather_kernel(
    sorted_inp,
    q,
    counts,
    out,
    N,
    M,
    Q,
    BLOCK_Q: tl.constexpr,
    BLOCK_N: tl.constexpr,
    interpolation: tl.constexpr,
):
    # sorted_inp is [N rows, M] ascending with all NaNs moved to the tail.
    # counts[row] = number of non-NaN elements in that row, so the valid
    # sorted values occupy indices [0, counts-1]. No reduction here: pure
    # gather + interpolation over a non-square 2D tile (XPU-safe pattern,
    # mirrors the validated `quantile` overlay kernel).
    pid_Q = ext.program_id(0)
    pid_N = ext.program_id(1)
    ctype = sorted_inp.dtype.element_ty

    offsets_Q = pid_Q * BLOCK_Q + tl.arange(0, BLOCK_Q)
    mask_Q = offsets_Q < Q
    offsets_N = pid_N * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_N = offsets_N < N

    cnt = tl.load(counts + offsets_N, mask_N, 0)
    last_valid = tl.maximum(cnt - 1, 0).to(ctype)
    q_vals = tl.load(q + offsets_Q, mask_Q, 0.0).to(ctype)

    q_block = last_valid[:, None] * q_vals[None, :]
    q_lower = tl.floor(q_block).to(tl.int32)
    q_upper = tl.ceil(q_block).to(tl.int32)

    base = offsets_N[:, None] * M
    inp_lower = tl.load(sorted_inp + base + q_lower, mask_N[:, None], 0.0)
    inp_upper = tl.load(sorted_inp + base + q_upper, mask_N[:, None], 0.0)

    if interpolation == "linear":
        q_frac = q_block - q_lower.to(ctype)
        result = inp_lower + (inp_upper - inp_lower) * q_frac
    elif interpolation == "lower":
        result = inp_lower
    elif interpolation == "higher":
        result = inp_upper
    elif interpolation == "nearest":
        q_round = tl.extra.xpu.libdevice.rint(q_block)
        result = tl.where(q_round == q_upper.to(ctype), inp_upper, inp_lower)
    else:  # midpoint
        result = (inp_lower + inp_upper) * 0.5

    result = tl.where(cnt[:, None] == 0, float("nan"), result)

    out_ptrs = out + offsets_N[:, None] * Q + offsets_Q[None, :]
    mask_out = mask_N[:, None] & mask_Q[None, :]
    tl.store(out_ptrs, result, mask_out)


def _launch_counts(rows2d, counts, M_rows, reduction_size, is_f64):
    block_n = builtins.min(triton.next_power_of_2(reduction_size), _COUNT_BLOCK_N)
    if M_rows <= _COUNT_SMALL_M and reduction_size >= _COUNT_HUGE_N:
        block_m = _COUNT_SMALL_BLOCK_M
    else:
        block_m = _COUNT_BLOCK_M
    grid = (triton.cdiv(M_rows, block_m),)
    with torch_device_fn.device(rows2d.device):
        _nan_count_rows_kernel[grid](
            rows2d,
            counts,
            M_rows,
            reduction_size,
            is_f64,
            block_m,
            block_n,
            buffer_size_limit=2048,
        )


def _nanquantile_impl(
    inp, q, dim=None, keepdim=False, interpolation="linear", out=None
) -> Tensor:
    logger.debug("GEMS_KUNLUNXIN NANQUANTILE")
    _validate_args(inp, q, dim, interpolation, out)

    original_ndim = inp.ndim
    dim_was_none = dim is None
    if dim_was_none:
        reduced = inp.contiguous().view(-1)
        dim = 0
    else:
        if inp.ndim == 0:
            if dim not in (-1, 0):
                raise IndexError("Dimension out of range")
            dim = 0
            reduced = inp.reshape(1)
        else:
            reduced = torch.movedim(inp, dim, -1)
            dim %= inp.ndim
            reduced = reduced.contiguous()

    q_is_tensor = isinstance(q, torch.Tensor)
    q_is_scalar = not q_is_tensor or q.dim() == 0
    if not q_is_tensor:
        q_value = float(q)
        if not 0.0 <= q_value <= 1.0:
            raise RuntimeError(
                f"quantile() q must be in the range [0, 1] but got {q_value}"
            )
        q = torch.tensor(q_value, dtype=inp.dtype, device=inp.device)
    q_contiguous = q.contiguous().reshape(-1)

    reduction_size = reduced.size(-1)
    if reduction_size > MAX_REDUCTION_SIZE:
        raise RuntimeError(
            "quantile() input tensor is too large; the reduction dimension "
            f"must not exceed {MAX_REDUCTION_SIZE} elements"
        )
    n_rows = reduced.numel() // reduction_size
    q_size = q_contiguous.numel()
    internal_shape = (*reduced.shape[:-1], q_size)
    if q_is_scalar:
        result_shape = list(reduced.shape[:-1])
        if keepdim:
            if dim_was_none:
                result_shape = [1] * original_ndim
            else:
                result_shape.insert(dim, 1)
    else:
        result_shape = [q_size, *reduced.shape[:-1]]
        if keepdim:
            if dim_was_none:
                result_shape = [q_size, *([1] * original_ndim)]
            else:
                result_shape.insert(dim + 1, 1)
    direct_out = False
    if out is not None and q_is_scalar:
        out.resize_(result_shape)
        out_view = out.reshape(reduced.shape[:-1])
        if out_view.is_contiguous():
            internal = out_view.unsqueeze(-1)
            direct_out = True
        else:
            internal = torch.empty(internal_shape, dtype=inp.dtype, device=inp.device)
    elif out is not None and n_rows == 1:
        out.resize_(result_shape)
        if out.is_contiguous():
            internal = out.reshape(internal_shape)
            direct_out = True
        else:
            internal = torch.empty(internal_shape, dtype=inp.dtype, device=inp.device)
    else:
        internal = torch.empty(internal_shape, dtype=inp.dtype, device=inp.device)

    if q_size and reduction_size:
        is_f64 = inp.dtype == torch.float64
        rows2d = reduced.reshape(n_rows, reduction_size)

        # 1) NaN -> +inf so a plain ascending sort pushes NaNs to each row tail.
        keys = torch.empty_like(rows2d)
        with torch_device_fn.device(inp.device):
            _nan_to_posinf_kernel[(triton.cdiv(rows2d.numel(), 1024),)](
                rows2d, keys, rows2d.numel(), is_f64, BLOCK_SIZE=1024
            )
        # 2) Backend radix sort (XPU-validated); NaNs (now +inf) land at tail.
        sorted_values, _ = sort_stable(keys, stable=False, dim=-1, descending=False)
        sorted_values = sorted_values.contiguous()
        # 3) Per-row non-NaN count (from the ORIGINAL values, not the keys).
        counts = torch.empty(n_rows, dtype=torch.int32, device=inp.device)
        _launch_counts(rows2d, counts, n_rows, reduction_size, is_f64)
        # 4) Gather + interpolate over indices [0, counts-1].
        block_q, block_n = _block_qn(q_size, n_rows)
        grid = (triton.cdiv(q_size, block_q), triton.cdiv(n_rows, block_n))
        internal_rows = internal.reshape(n_rows, q_size)
        with torch_device_fn.device(inp.device):
            _nanquantile_gather_kernel[grid](
                sorted_values,
                q_contiguous,
                counts,
                internal_rows,
                n_rows,
                reduction_size,
                q_size,
                BLOCK_Q=block_q,
                BLOCK_N=block_n,
                interpolation=interpolation,
            )

    if q_is_scalar:
        result = internal.squeeze(-1)
    else:
        result = internal.movedim(-1, 0)
    if keepdim:
        result = result.reshape(result_shape)

    if out is not None:
        if direct_out:
            return out
        out.resize_(result.shape)
        out.copy_(result)
        return out
    return result


def nanquantile(inp, q, dim=None, keepdim=False, interpolation="linear") -> Tensor:
    return _nanquantile_impl(inp, q, dim, keepdim, interpolation)


def nanquantile_scalar(
    inp, q, dim=None, keepdim=False, interpolation="linear"
) -> Tensor:
    return _nanquantile_impl(inp, q, dim, keepdim, interpolation)


def nanquantile_out(
    inp, q, dim=None, keepdim=False, interpolation="linear", out=None
) -> Tensor:
    return _nanquantile_impl(inp, q, dim, keepdim, interpolation, out)


def nanquantile_scalar_out(
    inp, q, dim=None, keepdim=False, interpolation="linear", out=None
) -> Tensor:
    return _nanquantile_impl(inp, q, dim, keepdim, interpolation, out)



