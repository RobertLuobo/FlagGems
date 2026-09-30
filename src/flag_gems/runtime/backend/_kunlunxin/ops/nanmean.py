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

from flag_gems.ops.nanmean import (
    _complex_mean,
    _masked_input_parts,
    _mean_backward,
    _nanmean_global,
    _normalize_dims,
)
from flag_gems.ops.sum import sum_dim
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as ext

from .copy import copy_

logger = logging.getLogger(__name__)

_BLOCK_N_MAX = 8192
_BLOCK_M = 128
_SMALL_M = 4096
_HUGE_N = 32768
_SMALL_BLOCK_M = 8

@libentry()
@triton.jit
def nanmean_rows_kernel(
    inp,
    out,
    M,
    N,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    # Row-reduce with reduce-OUTSIDE: accumulate sum and non-NaN count into
    # persisted [BLOCK_M, BLOCK_N] tiles, single tl.sum(axis=1) after the loop
    # (exact for all N on XPU; full BLOCK_N column blocks unmasked, the tail
    # masked). NaN detected by integer bit compare (setuo unordered fp compare
    # crashes the XPU LLVM backend); NaN lanes zeroed before summing and
    # excluded from the count. An all-NaN row yields count 0 -> sum/0 = NaN.
    if tl.constexpr(out.dtype.element_ty == tl.float64):
        cdtype = tl.float64
    else:
        cdtype = tl.float32

    pid = ext.program_id(0)
    rows = pid * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]
    inp = inp + rows * N
    out = out + rows
    row_mask = rows < M

    _sum = tl.zeros([BLOCK_M, BLOCK_N], dtype=cdtype)
    _cnt = tl.zeros([BLOCK_M, BLOCK_N], dtype=cdtype)
    for off in range(0, N, BLOCK_N):
        cols = off + tl.arange(0, BLOCK_N)[None, :]
        mask = row_mask and (cols < N)
        val = tl.load(inp + cols, mask, other=0.0).to(cdtype)
        if tl.constexpr(cdtype == tl.float64):
            b = val.to(tl.int64, bitcast=True)
            is_nan = (b & 0x7FFFFFFFFFFFFFFF) > 0x7FF0000000000000
        else:
            b = val.to(tl.int32, bitcast=True)
            is_nan = (b & 0x7FFFFFFF) > 0x7F800000
        valid = mask and (not is_nan)
        val = tl.where(is_nan, 0.0, val)
        _sum += val
        _cnt += valid.to(cdtype)
    s = tl.sum(_sum, axis=1)
    c = tl.sum(_cnt, axis=1)
    tl.store(out, (s / c)[:, None], row_mask)


def _permute_last(inp, dims):
    """Bring reduced dims to the end via the gems triton strided copy."""
    ndim = inp.ndim
    keep = [d for d in range(ndim) if d not in dims]
    perm = keep + sorted(dims)
    shape = list(inp.shape)
    N = 1
    for d in dims:
        N *= shape[d]
    M = inp.numel() // N if N else 0
    view = inp.permute(*perm).reshape(M, N)
    buf = torch.empty(M * N, dtype=inp.dtype, device=inp.device).reshape(M, N)
    copy_(buf, view)
    return buf, M, N


def _launch_nanmean_rows(inp, out, M, N):
    block_n = min(triton.next_power_of_2(N), _BLOCK_N_MAX)
    if M <= _SMALL_M and N >= _HUGE_N:
        block_m = _SMALL_BLOCK_M
    else:
        block_m = _BLOCK_M
    grid = (triton.cdiv(M, block_m),)
    with torch_device_fn.device(inp.device):
        nanmean_rows_kernel[grid](
            inp, out, M, N, block_m, block_n, buffer_size_limit=2048
        )


def nanmean_dim(inp, dim=None, keepdim=False, *, dtype=None):
    logger.debug("GEMS_KUNLUNXIN NANMEAN DIM")
    if dtype is None:
        dtype = inp.dtype

    dims = _normalize_dims(dim, inp.ndim)

    if inp.ndim == 0:
        return _nanmean_global(inp, dtype=dtype)

    # dim=[] or full reduction -> delegate to the (working) global path.
    if len(dims) == 0 or len(dims) == inp.ndim:
        result = _nanmean_global(inp, dtype=dtype)
        if keepdim:
            result = result.reshape([1] * inp.ndim)
        return result

    shape = list(inp.shape)
    out_keep = list(shape)
    N = 1
    for d in dims:
        N *= shape[d]
        out_keep[d] = 1
    out_squeeze = [shape[d] for d in range(inp.ndim) if d not in dims]

    if N == 0:
        out = torch.full(out_keep, float("nan"), dtype=dtype, device=inp.device)
        return out if keepdim else out.reshape(out_squeeze)

    M = inp.numel() // N
    if M == 0:
        out = torch.empty(out_keep, dtype=dtype, device=inp.device)
        return out if keepdim else out.reshape(out_squeeze)

    if not inp.is_contiguous():
        inp = inp.contiguous()
    buf, M2, N2 = _permute_last(inp, dims)
    out_flat = torch.empty(M2, dtype=dtype, device=inp.device)
    _launch_nanmean_rows(buf, out_flat, M2, N2)

    if keepdim:
        return out_flat.reshape(out_keep)
    return out_flat.reshape(out_squeeze)


class _NanmeanAutograd(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inp, dim, keepdim, dtype):
        dtype = dtype or inp.dtype
        dims = _normalize_dims(dim, inp.ndim)
        if not dims:
            dims = list(range(inp.ndim))
        ctx.dims = dims
        ctx.keepdim = keepdim
        ctx.input_dtype = inp.dtype
        _, _, valid = _masked_input_parts(inp, dtype)
        count = sum_dim(valid, dim=dims, keepdim=True)
        ctx.save_for_backward(valid, count)
        if inp.is_complex() or dtype.is_complex:
            return _complex_mean(inp, dim, keepdim, dtype)
        return nanmean(inp.detach(), dim, keepdim, dtype=dtype)

    @staticmethod
    def backward(ctx, grad):
        from flag_gems.ops.copy import copy_

        valid, count = ctx.saved_tensors
        if not ctx.keepdim:
            for dim in sorted(ctx.dims):
                grad = grad.unsqueeze(dim)
        result = torch.empty_like(valid, dtype=ctx.input_dtype)
        _mean_backward(grad, valid, count, out0=result)
        return result, None, None, None


def nanmean(inp, dim=None, keepdim=False, *, dtype=None):
    logger.debug("GEMS_KUNLUNXIN NANMEAN")
    if not (inp.is_floating_point() or inp.is_complex()):
        raise NotImplementedError(
            "nanmean(): expected input to have floating point or complex dtype but got "
            f"{inp.dtype}"
        )
    if dtype is not None and not (dtype.is_floating_point or dtype.is_complex):
        raise RuntimeError(
            "nanmean(): could not infer output dtype. Optional dtype must be either "
            f"a floating point or complex dtype. Got: {dtype}"
        )
    complex_output = dtype is not None and dtype.is_complex
    if inp.requires_grad and torch.is_grad_enabled():
        return _NanmeanAutograd.apply(inp, dim, keepdim, dtype)
    if inp.is_complex() or complex_output:
        return _complex_mean(inp, dim, keepdim, dtype or inp.dtype)
    if dim is None:
        result = _nanmean_global(inp, dtype=dtype)
        if keepdim:
            result = result.reshape([1] * inp.ndim)
        return result
    return nanmean_dim(inp, dim=dim, keepdim=keepdim, dtype=dtype)


def nanmean_out(inp, dim=None, keepdim=False, *, dtype=None, out=None):
    logger.debug("GEMS_KUNLUNXIN NANMEAN_OUT")
    if out is None:
        raise RuntimeError("nanmean(): missing required out tensor")
    if torch.is_grad_enabled() and (inp.requires_grad or out.requires_grad):
        raise RuntimeError(
            "nanmean(): functions with out= arguments don't support automatic differentiation"
        )
    if out.device != inp.device:
        raise RuntimeError(
            "nanmean: expected result tensor to be on the same device as input"
        )
    if dtype is not None and dtype != out.dtype:
        raise RuntimeError(
            "nanmean: provided dtype must match dtype of result. Got "
            f"{out.dtype} and {dtype}."
        )
    result = nanmean(inp, dim=dim, keepdim=keepdim, dtype=dtype or out.dtype)
    if out.shape != result.shape:
        out.resize_(result.shape)
    if out.is_complex():
        out_parts = torch.view_as_real(out)
        result_parts = torch.view_as_real(result)
        copy_(out_parts, result_parts)
    else:
        copy_(out, result)
    return out
