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
from flag_gems.utils import triton_lang_extension as ext
from flag_gems.utils.limits import get_dtype_max, get_dtype_min

from ..utils.tle_copy import tle_copy
from .copy import copy_

logger = logging.getLogger(__name__)

# aminmax mirrors the kunlunxin amax/amin override (2026-08-16 performance
# closure), computing both min and max in a single pass: every kernel below
# keeps a [BLOCK_M, 1] running min accumulator AND a [BLOCK_M, 1] running max
# accumulator, and reduces each [BLOCK_M, BLOCK_N] block along N inside the
# loop (reduce-INSIDE; the ONLY form numerically correct on this XPU for bf16
# tails, verified for amax/amin). NEED_MASK=False compiles to a mask-free
# kernel (all tiles full because M and N both divide by the block sizes),
# avoiding the XPU masked-memory slow path. NaN propagation semantics are
# identical to amax/amin / torch aminmax on this device.

_FULL_REDUCTION_BLOCK_SIZE = 8192
# Fast-path tile preferences for the DUAL-BUCKET kernel (min AND max live
# state), measured on this XPU (2026-08-17 sweep):
#   fp16:     [BLOCK_M=128, BLOCK_N=1024] is the sweet spot on every shape;
#   fp32/bf16: the two fp32 accumulator + two where-tile copies exhaust
#              uni_sram above ~8192 lanes per tile (BLOCK_M * BLOCK_N FAILS
#              to compile at (64,256)/(32,512)/(128,256)...). Verified-sound
#              picks keep BLOCK_M * BLOCK_N <= 8192, e.g. (32,256), (16,512),
#              (8,1024), (128,64), (64,128) -- all compile and match exactly.
# Small BLOCK_N (<=16) with a long loop is catastrophic (gather addressing), so
# the fast path only triggers when both M and N divide by the picked tiles
# (i.e. the whole reduction is mask-free); everything else keeps the HEAD
# masked path unchanged (HEAD tile bounds: BLOCK_M <= 8, BLOCK_N <= 512).
_FAST_BN_FP16 = (1024, 512, 256, 128, 64, 32, 16)
_FAST_BM_FP16 = (128, 64, 32, 16, 8, 4, 2)
_FAST_BN_FP32 = (256, 512, 128, 1024, 64, 32, 16)
_FAST_BM_FP32 = (32, 64, 16, 128, 8, 4, 2)
_MAX_TILE_LANES_FP32 = 8192  # BLOCK_M * BLOCK_N upper bound (uni_sram)


def _aminmax_block_n(n):
    # Restrict BLOCK_N to a power of two that does not exceed N or 512.
    # This avoids Triton XPU bugs triggered when BLOCK_N == 1024 with
    # small N (e.g. shape (1, 2), dim=0 produces N=1 in the dim path).
    return max(1, min(triton.next_power_of_2(n), 512))


def _aminmax_block_m(m):
    # Heuristic for the row-block size; choose the largest BLOCK_M that does
    # not exceed M (capped to one of the tune-space values).
    if m <= 1:
        return 1
    if m <= 2:
        return 2
    if m <= 4:
        return 4
    return 8


def _pick_fast_tile(M, N, big_acc):
    """Return (BLOCK_M, BLOCK_N) with M % BLOCK_M == 0 and N % BLOCK_N == 0,
    or None when no mask-free tile covers this shape. `big_acc` is True for
    fp32/bf16 (fp32 accumulator lanes), where the dual-bucket kernel only
    compiles up to 8192 lanes per tile."""
    bns = _FAST_BN_FP32 if big_acc else _FAST_BN_FP16
    bms = _FAST_BM_FP32 if big_acc else _FAST_BM_FP16
    max_lanes = _MAX_TILE_LANES_FP32 if big_acc else (1 << 30)
    bn = next((b for b in bns if N % b == 0), None)
    if bn is None:
        return None
    bm = next((m for m in bms if M % m == 0 and m * bn <= max_lanes), None)
    if bm is None:
        return None
    return bm, bn


@libentry()
@triton.jit
def aminmax_kernel_1(
    inp,
    min_out,
    max_out,
    M,
    BLOCK_SIZE: tl.constexpr,
    NEED_MASK: tl.constexpr,
):
    pid = ext.program_id(0)

    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    inp_ptrs = inp + offset

    dtype = inp.type.element_ty
    acc_type = tl.float32 if dtype == tl.bfloat16 else dtype
    min_fill = get_dtype_max(dtype)
    max_fill = get_dtype_min(dtype)

    if NEED_MASK:
        mask = offset < M
        value = tl.load(inp_ptrs, mask=mask, other=0.0).to(acc_type)
        min_val = tl.min(tl.where(mask, value, min_fill))
        max_val = tl.max(tl.where(mask, value, max_fill))
    else:
        value = tl.load(inp_ptrs).to(acc_type)
        min_val = tl.min(value)
        max_val = tl.max(value)

    tl.store(min_out + pid, min_val.to(dtype))
    tl.store(max_out + pid, max_val.to(dtype))


@libentry()
@triton.jit
def aminmax_kernel_2(
    min_inp,
    max_inp,
    min_out,
    max_out,
    M,
    BLOCK_SIZE: tl.constexpr,
    NEED_MASK: tl.constexpr,
):
    pid = ext.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    min_ptrs = min_inp + offset
    max_ptrs = max_inp + offset

    dtype = min_inp.type.element_ty
    acc_type = tl.float32 if dtype == tl.bfloat16 else dtype
    min_fill = get_dtype_max(dtype)
    max_fill = get_dtype_min(dtype)

    if NEED_MASK:
        mask = offset < M
        min_value = tl.load(min_ptrs, mask=mask, other=0.0).to(acc_type)
        max_value = tl.load(max_ptrs, mask=mask, other=0.0).to(acc_type)
        min_val = tl.min(tl.where(mask, min_value, min_fill))
        max_val = tl.max(tl.where(mask, max_value, max_fill))
    else:
        min_value = tl.load(min_ptrs).to(acc_type)
        max_value = tl.load(max_ptrs).to(acc_type)
        min_val = tl.min(min_value)
        max_val = tl.max(max_value)

    tl.store(min_out + pid, min_val.to(dtype))
    tl.store(max_out + pid, max_val.to(dtype))


@libentry()
@triton.jit
def aminmax_kernel_2d(
    inp,
    min_out,
    max_out,
    M,
    N,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    NEED_MASK: tl.constexpr,
):
    # NOTE: compute the element dtype BEFORE any pointer arithmetic: after
    # `inp = inp + rows * N` the tensor holds pointer<dtype> elements and
    # `.type.element_ty` would return the pointer type itself.
    dtype = inp.type.element_ty
    acc_type = tl.float32 if dtype == tl.bfloat16 else dtype
    max_fill = get_dtype_max(dtype)  # identity for the min accumulator
    min_fill = get_dtype_min(dtype)  # identity for the max accumulator

    # Map the program id to the row of inp it should compute.
    pid = ext.program_id(0)
    rows = pid * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]
    inp = inp + rows * N
    min_out = min_out + rows
    max_out = max_out + rows
    row_mask = rows < M

    # Reduce each tile before updating the row accumulators. The XPU backend
    # miscompiles the equivalent reduce-after-loop form for bf16 tails.
    min_identity = tl.full([], value=max_fill, dtype=acc_type)
    max_identity = tl.full([], value=min_fill, dtype=acc_type)
    _min = tl.full([BLOCK_M, 1], value=max_fill, dtype=acc_type)
    _max = tl.full([BLOCK_M, 1], value=min_fill, dtype=acc_type)
    for off in range(0, N, BLOCK_N):
        cols = off + tl.arange(0, BLOCK_N)[None, :]
        if NEED_MASK:
            col_mask = cols < N
            mask = row_mask & col_mask
            a = tl.load(inp + cols, mask=mask, other=0.0).to(acc_type)
            a_min = tl.where(mask, a, min_identity)
            a_max = tl.where(mask, a, max_identity)
            tile_min = tl.min(a_min, axis=1)[:, None].to(acc_type)
            tile_max = tl.max(a_max, axis=1)[:, None].to(acc_type)
        else:
            a = tl.load(inp + cols).to(acc_type)
            tile_min = tl.min(a, axis=1)[:, None].to(acc_type)
            tile_max = tl.max(a, axis=1)[:, None].to(acc_type)
        _min = tl.minimum(_min, tile_min)
        _max = tl.maximum(_max, tile_max)
    if NEED_MASK:
        tl.store(min_out, _min, row_mask)
        tl.store(max_out, _max, row_mask)
    else:
        tl.store(min_out, _min.to(dtype))
        tl.store(max_out, _max.to(dtype))


def _reduce_to_scalar(min_src, max_src, min_out, max_out):
    """Repeatedly reduce (min_src, max_src) with 8192-wide blocks until a
    scalar remains."""
    block = _FULL_REDUCTION_BLOCK_SIZE
    n = min_src.numel()
    min_data, max_data = min_src, max_src
    while n > block:
        mid_size = triton.cdiv(n, block)
        min_mid = torch.empty((mid_size,), dtype=min_data.dtype, device=min_data.device)
        max_mid = torch.empty((mid_size,), dtype=max_data.dtype, device=max_data.device)
        aminmax_kernel_2[(mid_size, 1)](
            min_data,
            max_data,
            min_mid,
            max_mid,
            n,
            block,
            n % block != 0,
            buffer_size_limit=2048,
        )
        min_data, max_data = min_mid, max_mid
        n = mid_size
    aminmax_kernel_2[(1, 1)](
        min_data,
        max_data,
        min_out,
        max_out,
        n,
        triton.next_power_of_2(n),
        n % triton.next_power_of_2(n) != 0,
        buffer_size_limit=2048,
    )


def _aminmax_flat(inp, min_out, max_out, device):
    """Full (dim=None) reduction over `inp` (any numel)."""
    # Flatten first: the staging below slices `inp[rows * block:]` as a flat
    # buffer, which for a rank>1 input would instead be a (out-of-range)
    # dim-0 slice and silently drop the residue. Non-contiguous inputs are
    # copied to a fresh contiguous buffer with the kunlunxin triton
    # strided-copy.
    if inp.is_contiguous():
        flat = inp.view(-1)
    else:
        flat = torch.empty((inp.numel(),), dtype=inp.dtype, device=inp.device)
        with torch_device_fn.device(inp.device):
            copy_(flat.view(inp.shape), inp)
    inp = flat
    block = _FULL_REDUCTION_BLOCK_SIZE
    numel = inp.numel()
    if numel <= block:
        # Pad only to the next power of two (never the full 8192-wide block)
        # so no masked lanes are read at all for exact powers of two.
        block_n = triton.next_power_of_2(numel)
        aminmax_kernel_1[(1, 1)](
            inp,
            min_out,
            max_out,
            numel,
            block_n,
            numel % block_n != 0,
            buffer_size_limit=2048,
        )
        return
    # Main pass: treat the flat buffer as rows of `block` elements and reduce
    # each row with the mask-free 2-D tile kernel (few wide programs instead of
    # the old one-program-per-row staging, which launch-bounds at big numel).
    # min and max are computed in the very same pass (dual-bucket reduction).
    rows = numel // block
    res = numel - rows * block
    big_acc = inp.dtype != torch.float16  # fp32/bf16 use fp32 accumulator lanes
    tile = _pick_fast_tile(rows, block, big_acc)
    if tile is None:
        # Fall back to a single-row tile (BLOCK_M=1 always divides `rows`);
        # 1x1024 lanes is within the fp32/bf16 uni_sram budget.
        bm, bn = (1, 1024)
    else:
        bm, bn = tile
    min_mid = torch.empty((rows + (1 if res else 0),), dtype=inp.dtype, device=device)
    max_mid = torch.empty((rows + (1 if res else 0),), dtype=inp.dtype, device=device)
    aminmax_kernel_2d[(rows // bm, 1)](
        inp,
        min_mid,
        max_mid,
        rows,
        block,
        bm,
        bn,
        False,
        buffer_size_limit=2048,
    )
    if res:
        # A single masked program handles the (< block) residue.
        aminmax_kernel_1[(1, 1)](
            inp[rows * block :],
            min_mid[rows:],
            max_mid[rows:],
            res,
            triton.next_power_of_2(res),
            res % triton.next_power_of_2(res) != 0,
            buffer_size_limit=2048,
        )
    _reduce_to_scalar(min_mid, max_mid, min_out, max_out)


def aminmax(inp, dim=None, keepdim=False, *, out=None):
    logger.debug("GEMS_KUNLUNXIN AMINMAX")

    if dim is None:
        dtype = inp.dtype

        if out is not None:
            min_out = out[0] if isinstance(out, tuple) else out
            max_out = out[1] if isinstance(out, tuple) else out
            if not keepdim:
                min_out = min_out.squeeze()
                max_out = max_out.squeeze()
        else:
            if not keepdim:
                min_out = torch.empty([], dtype=dtype, device=inp.device)
                max_out = torch.empty([], dtype=dtype, device=inp.device)
            else:
                shape = [1] * inp.dim()
                min_out = torch.empty(shape, dtype=dtype, device=inp.device)
                max_out = torch.empty(shape, dtype=dtype, device=inp.device)

        with torch_device_fn.device(inp.device):
            _aminmax_flat(inp, min_out, max_out, inp.device)
        return min_out, max_out
    else:
        if isinstance(dim, int):
            dim = [dim]
        assert ((i >= -inp.ndim and i < inp.ndim) for i in dim), "Invalid dim"
        dtype = inp.dtype

        shape = list(inp.shape)
        dim = [d % inp.ndim for d in dim]
        N = 1
        for i in dim:
            N *= shape[i]
            shape[i] = 1
        M = inp.numel() // N

        if out is not None:
            min_out = out[0] if isinstance(out, tuple) else out
            max_out = out[1] if isinstance(out, tuple) else out
        else:
            min_out = torch.empty(shape, dtype=dtype, device=inp.device)
            max_out = torch.empty(shape, dtype=dtype, device=inp.device)

        if N == 1:
            # Every reduced dim has size 1: aminmax over it is the identity.
            # Use the kunlunxin triton strided-copy instead of launching a
            # reduction kernel at all.
            with torch_device_fn.device(inp.device):
                copy_(min_out, inp)
                copy_(max_out, inp)
            if not keepdim:
                min_out = min_out.squeeze(dim=dim)
                max_out = max_out.squeeze(dim=dim)
            return min_out, max_out

        if M == 1:
            # A single batch row: the reduction covers the whole tensor, so
            # the mask-free flat path (rows-of-8192 2D pass + staged scalar
            # reduce) is both faster and identical in result. This is hit by
            # the official benchmark for the dim=-1 variant of 1-D shapes
            # (e.g. (2**30,) dim=-1 -> M=1, N=2**30), which would otherwise
            # run a single-program 131072-iteration masked row loop.
            with torch_device_fn.device(inp.device):
                _aminmax_flat(inp, min_out, max_out, inp.device)
            if not keepdim:
                min_out = min_out.squeeze(dim=dim)
                max_out = max_out.squeeze(dim=dim)
            return min_out, max_out

        # Reorder so the reduced dims are innermost (same order as
        # dim_compress), then make it contiguous with the native strided-copy
        # engine instead of the much slower gems `.contiguous()` override.
        dim_i = inp.dim()
        stride = inp.stride()
        batch_dim = [i for i in range(dim_i) if i not in dim]
        sorted_reduction_dim = sorted(dim, key=lambda x: stride[x], reverse=True)
        order = batch_dim + sorted_reduction_dim
        view = inp.permute(order)
        if view.is_contiguous():
            src = view
        else:
            src = torch.empty(list(view.shape), dtype=dtype, device=inp.device)
            with torch_device_fn.device(inp.device):
                copy_(src, view)

        big_acc = dtype != torch.float16  # fp32/bf16 use fp32 accumulator lanes
        tile = _pick_fast_tile(M, N, big_acc)
        with torch_device_fn.device(inp.device):
            if tile is not None:
                block_m, block_n = tile
                grid = (triton.cdiv(M, block_m),)
                aminmax_kernel_2d[grid](
                    src,
                    min_out,
                    max_out,
                    M,
                    N,
                    block_m,
                    block_n,
                    False,
                    buffer_size_limit=2048,
                )
            else:
                # Masked fallback: keep the HEAD tile bounds (BLOCK_M <= 8,
                # BLOCK_N <= 512). Wider masked tiles (e.g. BLOCK_M=64 /
                # BLOCK_N=8192) are both uni_sram-heavy with the dual min/max
                # bucket state and numerically unreliable on masked tails
                # (XPU backend family issue; verified miscompilation at
                # (600, 40999) with BLOCK_M=64, BLOCK_N=8192/1024). The fast
                # mask-free path covers every shape whose M and N divide by
                # the picked tiles, so the slow masked path is only a
                # correctness backstop for odd shapes.
                block_n = _aminmax_block_n(N)
                block_m = _aminmax_block_m(M)
                grid = (triton.cdiv(M, block_m),)
                aminmax_kernel_2d[grid](
                    src,
                    min_out,
                    max_out,
                    M,
                    N,
                    block_m,
                    block_n,
                    True,
                    buffer_size_limit=2048,
                )
        if not keepdim:
            min_out = min_out.squeeze(dim=dim)
            max_out = max_out.squeeze(dim=dim)
        return min_out, max_out


# ---------------------------------------------------------------------------
# aten::_aminmax / aten::_aminmax.out  (whole-tensor reduction, dim=None only)
#
# The generic KernelGen implementation loads OOB tail lanes with
# `tl.load(mask=..., other=fill)` and feeds them straight into `tl.max` /
# `tl.min`. On this XPU the load mask is NOT honored inside the reduction, so
# out-of-range memory (which may exceed the true max, e.g. +inf) leaks into the
# result. These backend kernels replace OOB lanes with the reduction IDENTITY
# via `tl.where` BEFORE reducing (identity-fill), which the device does honor.
# NaN propagation matches torch._aminmax (a NaN forces both outputs to NaN).
# ---------------------------------------------------------------------------
@libentry()
@triton.jit
def _aminmax_full_kernel_1(
    inp,
    min_out,
    max_out,
    M,
    BLOCK_SIZE: tl.constexpr,
    IS_FLOAT: tl.constexpr,
    NEED_MASK: tl.constexpr,
):
    pid = ext.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    inp_ptrs = inp + offset

    dtype = inp.type.element_ty
    min_fill = get_dtype_max(dtype)  # identity for the min reduction
    max_fill = get_dtype_min(dtype)  # identity for the max reduction

    if NEED_MASK:
        mask = offset < M
        min_val = tl.load(inp_ptrs, mask=mask, other=min_fill)
        max_val = tl.load(inp_ptrs, mask=mask, other=max_fill)
        min_in = tl.where(mask, min_val, min_fill)
        max_in = tl.where(mask, max_val, max_fill)
    else:
        # Full tiles: every lane is in-bounds so no identity-fill is needed.
        # Two separate masked loads (rather than one shared load) are used
        # because on this XPU a single-load-feeds-both form compiles to a
        # markedly slower memory path than the twin masked-load form.
        mask = offset < M
        min_val = tl.load(inp_ptrs, mask=mask, other=min_fill)
        max_val = tl.load(inp_ptrs, mask=mask, other=max_fill)
        min_in = min_val
        max_in = max_val

    if IS_FLOAT:
        has_nan = tl.sum((min_val != min_val).to(tl.int32)) > 0
        nan = float("nan")
        min_val = tl.where(has_nan, nan, tl.min(min_in))
        max_val = tl.where(has_nan, nan, tl.max(max_in))
    else:
        min_val = tl.min(min_in)
        max_val = tl.max(max_in)

    tl.store(min_out + pid, min_val)
    tl.store(max_out + pid, max_val)


@libentry()
@triton.jit
def _aminmax_full_kernel_2(
    min_inp,
    max_inp,
    min_out,
    max_out,
    mid_size,
    BLOCK_MID: tl.constexpr,
    IS_FLOAT: tl.constexpr,
    NEED_MASK: tl.constexpr,
):
    offset = tl.arange(0, BLOCK_MID)
    min_ptrs = min_inp + offset
    max_ptrs = max_inp + offset

    dtype = min_inp.type.element_ty
    min_fill = get_dtype_max(dtype)
    max_fill = get_dtype_min(dtype)

    if NEED_MASK:
        mask = offset < mid_size
        min_value = tl.load(min_ptrs, mask=mask, other=0)
        max_value = tl.load(max_ptrs, mask=mask, other=0)
        min_in = tl.where(mask, min_value, min_fill)
        max_in = tl.where(mask, max_value, max_fill)
    else:
        min_value = tl.load(min_ptrs)
        max_value = tl.load(max_ptrs)
        min_in = min_value
        max_in = max_value

    if IS_FLOAT:
        has_nan = (
            tl.sum((min_value != min_value).to(tl.int32))
            + tl.sum((max_value != max_value).to(tl.int32))
        ) > 0
        nan = float("nan")
        min_val = tl.where(has_nan, nan, tl.min(min_in))
        max_val = tl.where(has_nan, nan, tl.max(max_in))
    else:
        min_val = tl.min(min_in)
        max_val = tl.max(max_in)

    tl.store(min_out, min_val)
    tl.store(max_out, max_val)


def _aminmax_reduce(inp, min_out, max_out):
    """Two-stage whole-tensor reduction of a 1-D contiguous `inp`.

    The main pass runs over the full BLOCK_SIZE tiles: every lane is in-bounds
    so no identity-fill (`tl.where`) is applied, matching the cost of a plain
    contiguous reduction. Only the trailing residue program and the final
    reduce read out-of-range lanes, and those use the identity-fill path so no
    out-of-bounds value can leak into `tl.max`/`tl.min`.
    """
    M = inp.numel()
    is_float = inp.is_floating_point()
    block_size = triton.next_power_of_2(math.ceil(math.sqrt(M)))
    n_full = M // block_size
    res = M - n_full * block_size
    mid_size = n_full + (1 if res else 0)
    min_mid = torch.empty((mid_size,), dtype=inp.dtype, device=inp.device)
    max_mid = torch.empty((mid_size,), dtype=inp.dtype, device=inp.device)
    if n_full > 0:
        _aminmax_full_kernel_1[(n_full, 1)](
            inp,
            min_mid,
            max_mid,
            M,
            block_size,
            is_float,
            False,
            buffer_size_limit=2048,
        )
    if res:
        _aminmax_full_kernel_1[(1, 1)](
            inp[n_full * block_size :],
            min_mid[n_full:],
            max_mid[n_full:],
            res,
            triton.next_power_of_2(res),
            is_float,
            True,
            buffer_size_limit=2048,
        )
    block_mid = triton.next_power_of_2(mid_size)
    _aminmax_full_kernel_2[(1, 1)](
        min_mid,
        max_mid,
        min_out,
        max_out,
        mid_size,
        block_mid,
        is_float,
        mid_size % block_mid != 0,
        buffer_size_limit=2048,
    )


def _aminmax_run(inp, out0=None, out1=None):
    if inp.numel() == 0:
        # ATen refuses to reduce an empty input rather than returning +-inf.
        raise RuntimeError(
            "aminmax(): cannot compute aminmax over an empty dimension as the "
            "operation has no identity."
        )

    dtype = inp.dtype
    if out0 is None:
        min_out = torch.empty([], dtype=dtype, device=inp.device)
        max_out = torch.empty([], dtype=dtype, device=inp.device)
    else:
        # ATen validates dtype and device but resizes a mis-shaped output to a
        # scalar in place, returning the caller's objects (so they alias).
        for _name, o in (("out0", out0), ("out1", out1)):
            if o.dtype != dtype:
                raise RuntimeError(
                    f"Expected out tensor to have dtype {dtype}, but got "
                    f"{o.dtype} instead"
                )
            if o.device != inp.device:
                raise RuntimeError(
                    f"Expected out tensor to have device {inp.device}, but got "
                    f"{o.device} instead"
                )
        if out0.shape != torch.Size([]):
            out0.resize_(())
        if out1.shape != torch.Size([]):
            out1.resize_(())
        min_out, max_out = out0, out1

    # Flatten to a 1-D contiguous buffer without the slow gems `.contiguous()`
    # override: a sliced/transposed/expanded view would otherwise be read in the
    # wrong element order (or past its own storage for stride-0 expands). The
    # dense copy uses a shape-matched destination (the native strided-copy
    # engine requires matching shapes) and is then flattened.
    if inp.is_contiguous():
        flat = inp.view(-1)
    else:
        src = torch.empty(list(inp.shape), dtype=dtype, device=inp.device)
        with torch_device_fn.device(inp.device):
            if not tle_copy(inp, src):
                src.copy_(inp)
        flat = src.view(-1)

    with torch_device_fn.device(inp.device):
        _aminmax_reduce(flat, min_out, max_out)
    return min_out, max_out


def _aminmax(inp):
    logger.debug("GEMS_KUNLUNXIN _AMINMAX")
    return _aminmax_run(inp)


def _aminmax_out(inp, *, out0, out1):
    logger.debug("GEMS_KUNLUNXIN _AMINMAX_OUT")
    return _aminmax_run(inp, out0=out0, out1=out1)
