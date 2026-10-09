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
import math

import torch
import triton
import triton.language as tl

from flag_gems.ops.nanmean import (
    _NanmeanAutograd,
    _complex_mean,
    _nanmean_global,
    _normalize_dims,
    _squeeze_dims,
)
from flag_gems.ops.copy import copy_
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import dim_compress, libentry
from flag_gems.utils import triton_lang_extension as ext

logger = logging.getLogger(__name__)

_TILE_BUDGET = 32768
_N_WIDE = 8192


def _block_n(N, dtype):
    if N > _N_WIDE:
        return builtins.min(triton.next_power_of_2(N), 2048)
    if dtype in (torch.float32, torch.float64):
        return builtins.min(triton.next_power_of_2(N), 512)
    return builtins.min(triton.next_power_of_2(N), 256)


def _block_m(M, dtype):
    cap = 64 if dtype in (torch.float32, torch.float64) else 128
    return builtins.min(triton.next_power_of_2(M), cap)


def heur_n_block_size(args):
    return _block_n(args["N"], args["X"].dtype)


def heur_m_block_size(args):
    block_n = _block_n(args["N"], args["X"].dtype)
    block_m = _block_m(args["M"], args["X"].dtype)
    return builtins.max(builtins.min(block_m, _TILE_BUDGET // block_n), 1)


@libentry()
@triton.heuristics(
    values={
        "BLOCK_M": heur_m_block_size,
        "BLOCK_N": heur_n_block_size,
    },
)
@triton.jit
def nanmean_dim_kernel(X, Out, M, N, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    if tl.constexpr(Out.dtype.element_ty == tl.float64):
        acc_dtype = tl.float64
    else:
        acc_dtype = tl.float32

    pid = ext.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]
    X = X + pid * N
    Out = Out + pid
    row_mask = pid < M

    # Persisted [BLOCK_M, BLOCK_N] accumulators + a SINGLE reduce after the
    # loop (slot j folds cols j, j+BLOCK_N, ...). We deliberately avoid an
    # in-loop axis=1 reduce and the generic 2-D [BLOCK_N, BLOCK_K] axis=0
    # kernel: both miscompile on XPU for fp16/bf16 (wrong results) and the
    # generic non-inner kernel also trips TritonXPUUnrollControl on wide K.
    # heur_m/heur_n cap BLOCK_M*BLOCK_N to _TILE_BUDGET so the tile is bounded.
    sum_acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=acc_dtype)
    cnt_acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=acc_dtype)
    for off in range(0, N, BLOCK_N):
        cols = off + tl.arange(0, BLOCK_N)[None, :]
        col_mask = cols < N
        mask = row_mask and col_mask
        a = tl.load(X + cols, mask, other=0.0).to(acc_dtype)
        valid = mask and (a == a)
        sum_acc += tl.where(valid, a, 0.0)
        cnt_acc += valid.to(acc_dtype)

    total = tl.sum(sum_acc, axis=1)[:, None]
    count = tl.sum(cnt_acc, axis=1)[:, None]
    result = total / count
    tl.store(Out, result, row_mask)


def nanmean_dim(inp, dim=None, keepdim=False, *, dtype=None):
    logger.debug("GEMS_KUNLUNXIN NANMEAN DIM")
    logger.debug("GEMS NANMEAN DIM")
    if dtype is None:
        dtype = inp.dtype

    dims = _normalize_dims(dim, inp.ndim)

    if inp.ndim == 0:
        return _nanmean_global(inp, dtype=dtype)

    # dim=[] -> reduce all
    if len(dims) == 0:
        result = _nanmean_global(inp, dtype=dtype)
        if keepdim:
            result = result.reshape([1] * inp.ndim)
        return result

    # full-dimensional reduction -> delegate to global
    if len(dims) == inp.ndim:
        result = _nanmean_global(inp, dtype=dtype)
        if keepdim:
            result = result.reshape([1] * inp.ndim)
        return result

    shape = list(inp.shape)
    N = 1
    for d in dims:
        N *= shape[d]

    reduced_shape = [1 if i in dims else s for i, s in enumerate(shape)]

    if N == 0:
        out = torch.full(reduced_shape, float("nan"), dtype=dtype, device=inp.device)
        return out if keepdim else _squeeze_dims(out, dims)

    if math.prod(reduced_shape) == 0:
        out = torch.empty(reduced_shape, dtype=dtype, device=inp.device)
        return out if keepdim else _squeeze_dims(out, dims)

    # Compress the reduced dims to the trailing axis so the reduction is always
    # over the inner (last) dimension: [M, N]. This routes every case through
    # the single XPU-safe inner-reduction kernel above and avoids the generic
    # non-inner (K>1) kernel that miscompiles / fails to tune on XPU.
    x = dim_compress(inp, dims)
    M = x.numel() // N

    out_shape = reduced_shape
    if not keepdim:
        out_shape = [s for i, s in enumerate(reduced_shape) if i not in dims]

    # Edge case: all non-reduced dims collapse -> global reduction over N.
    if M == 1:
        scalar_out = _nanmean_global(x, dtype=dtype)
        return scalar_out.reshape(out_shape)

    # Edge case: reducing a trivial size-1 dimension -> identity (cast only).
    if N == 1:
        out = torch.empty(out_shape, dtype=dtype, device=inp.device)
        copy_(out, x.reshape(out_shape))
        return out

    out = torch.empty(out_shape, dtype=dtype, device=inp.device)
    grid = lambda META: (triton.cdiv(M, META["BLOCK_M"]),)  # noqa: E731

    with torch_device_fn.device(inp.device):
        nanmean_dim_kernel[grid](x, out, M, N, buffer_size_limit=2048)
    return out


def nanmean(inp, dim=None, keepdim=False, *, dtype=None):
    logger.debug("GEMS_KUNLUNXIN NANMEAN")
    logger.debug("GEMS NANMEAN")
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
    logger.debug("GEMS NANMEAN_OUT")
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
