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
from torch import Tensor

from flag_gems.ops.contiguous import contiguous
from flag_gems.ops.copy import copy_
from flag_gems.runtime import torch_device_fn
from flag_gems.runtime.backend._kunlunxin.ops.sort import sort as _xpu_sort
from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as ext

logger = logging.getLogger(__name__)

INTERPOLATION_METHODS = ("linear", "lower", "higher", "nearest", "midpoint")
MAX_REDUCTION_SIZE = 1 << 24


def _block_qn(Q, N):
    import builtins

    # The data-dependent gather along the Q axis miscompiles on XPU when the
    # trailing block is large (>=256 yields periodic garbage/NaN lanes), so cap
    # BLOCK_Q at 128 and let the launch grid tile the remaining q values.
    block_q = builtins.min(triton.next_power_of_2(Q), 128)
    block_n = builtins.min(triton.next_power_of_2(N), 1024)
    # A square 2D tile triggers a uni_sram overflow / PassManager crash in
    # ConvertTritonXPUToLLVM on XPU for this gather kernel. Perturb one dim so
    # the tile is never square (masking keeps the result correct).
    if block_n == block_q:
        block_n = builtins.min(block_n * 2, 1024) if block_n < 1024 else block_n // 2
    return block_q, block_n


@libentry()
@triton.jit
def _nanquantile_prep_kernel(inp, keys, counts, M, BLOCK: tl.constexpr):
    row = ext.program_id(0)
    base = row * M
    acc = tl.zeros((BLOCK,), dtype=tl.int32)
    for off in tl.range(0, M, BLOCK):
        cols = off + tl.arange(0, BLOCK)
        mask = cols < M
        v = tl.load(inp + base + cols, mask=mask, other=0.0)
        is_num = v == v
        valid = mask & is_num
        acc += valid.to(tl.int32)
        key = tl.where(is_num, v, float("inf"))
        tl.store(keys + base + cols, key, mask=mask)
    total = tl.sum(acc, axis=0)
    tl.store(counts + row, total)


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
    pid_Q = ext.program_id(0)
    pid_N = ext.program_id(1)
    ctype = sorted_inp.dtype.element_ty

    offsets_Q = pid_Q * BLOCK_Q + tl.arange(0, BLOCK_Q)
    mask_Q = offsets_Q < Q

    offsets_N = pid_N * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_N = offsets_N < N

    cnt = tl.load(counts + offsets_N, mask_N, 0)
    all_nan = cnt == 0
    last_valid = tl.maximum(cnt - 1, 0).to(ctype)

    q_vals = tl.load(q + offsets_Q, mask_Q, 0.0).to(ctype)
    q_block = last_valid[:, None] * q_vals[None, :]
    q_lower = tl.floor(q_block).to(tl.int32)
    q_upper = tl.ceil(q_block).to(tl.int32)

    out_ptrs = out + offsets_N[:, None] * Q + offsets_Q[None, :]
    mask_out = mask_N[:, None] & mask_Q[None, :]
    row_start = offsets_N[:, None] * M

    inp_lower = tl.load(sorted_inp + row_start + q_lower, mask_out, 0.0)
    inp_upper = tl.load(sorted_inp + row_start + q_upper, mask_out, 0.0)

    if interpolation == "linear":
        q_frac = q_block - q_lower
        result = inp_lower + (inp_upper - inp_lower) * q_frac
    elif interpolation == "lower":
        result = inp_lower
    elif interpolation == "higher":
        result = inp_upper
    elif interpolation == "nearest":
        q_round = tl.extra.xpu.libdevice.rint(q_block)
        result = tl.where(q_round == q_upper, inp_upper, inp_lower)
    else:
        result = (inp_lower + inp_upper) / 2

    result = tl.where(all_nan[:, None], float("nan"), result)
    tl.store(out_ptrs, result, mask_out)


def _validate_args(inp, q, dim, interpolation, out):
    if inp.dtype not in (torch.float32, torch.float64):
        raise RuntimeError(
            "quantile() input tensor must be either float or double dtype"
        )
    if inp.numel() == 0:
        raise RuntimeError("quantile() input tensor must be non-empty")
    if interpolation not in INTERPOLATION_METHODS:
        raise RuntimeError(
            "quantile() interpolation must be one of linear, lower, higher, "
            f"midpoint or nearest, but got {interpolation}"
        )
    if dim is not None and not isinstance(dim, int):
        raise TypeError("quantile() dim must be an integer or None")
    if isinstance(q, torch.Tensor):
        if q.dim() > 1:
            raise RuntimeError("quantile() q must be a scalar or 1D tensor")
        if q.dtype != inp.dtype:
            raise RuntimeError(
                "quantile() q tensor must be same dtype as the input tensor"
            )
        if q.device != inp.device:
            raise RuntimeError(
                "quantile() q tensor must be on the same device as the input"
            )
    if out is not None:
        if out.dtype != inp.dtype:
            raise RuntimeError(
                "quantile() out tensor must be same dtype as the input tensor"
            )
        if out.device != inp.device:
            raise RuntimeError(
                "quantile() out tensor must be on the same device as the input"
            )


def _nanquantile_impl(
    inp, q, dim=None, keepdim=False, interpolation="linear", out=None
) -> Tensor:
    logger.debug("GEMS_KUNLUNXIN NANQUANTILE")
    _validate_args(inp, q, dim, interpolation, out)

    original_ndim = inp.ndim
    dim_was_none = dim is None
    if dim_was_none:
        reduced = contiguous(inp).view(-1)
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
            reduced = contiguous(reduced)

    q_is_tensor = isinstance(q, torch.Tensor)
    q_is_scalar = not q_is_tensor or q.dim() == 0
    if not q_is_tensor:
        q_value = float(q)
        if not 0.0 <= q_value <= 1.0:
            raise RuntimeError(
                f"quantile() q must be in the range [0, 1] but got {q_value}"
            )
        q = torch.tensor(q_value, dtype=inp.dtype, device=inp.device)
    q_contiguous = contiguous(q).reshape(-1)

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
        if q_is_tensor and q_contiguous.numel():
            if not (
                bool(torch.all(q_contiguous >= 0.0))
                and bool(torch.all(q_contiguous <= 1.0))
            ):
                raise RuntimeError(
                    "quantile() q values must be in the range [0, 1]"
                )
        keys = torch.empty_like(reduced)
        counts = torch.empty(n_rows, dtype=torch.int32, device=inp.device)
        prep_block = min(triton.next_power_of_2(reduction_size), 2048)
        with torch_device_fn.device(inp.device):
            _nanquantile_prep_kernel[(n_rows,)](
                reduced, keys, counts, reduction_size, BLOCK=prep_block
            )
        sorted_values, _ = _xpu_sort(keys, dim=-1, descending=False)
        block_q, block_n = _block_qn(q_size, n_rows)
        grid = (triton.cdiv(q_size, block_q), triton.cdiv(n_rows, block_n))
        with torch_device_fn.device(inp.device):
            _nanquantile_gather_kernel[grid](
                sorted_values,
                q_contiguous,
                counts,
                internal,
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
        copy_(out, result)
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
