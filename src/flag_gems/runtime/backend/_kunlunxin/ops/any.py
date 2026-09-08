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

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import dim_compress, libentry
from flag_gems.utils import triton_lang_extension as ext

from ..utils.block_size_utils import get_block_size_1d

logger = logging.getLogger(__name__)

# torch.any: Tests if any elements in input evaluate to True. If the dtype of input
#            is not BOOL, then test if any elements in input evaluate to non-zero value
# In triton function, test if any elements in input evaluate to non-zero value is ok.

cluster_num = 12
core_num = 64
buf_len_per_core = 2048
vector_size = 16


def heur_m_block_size(args):
    return triton.next_power_of_2(min(triton.cdiv(args["M"], cluster_num), core_num))


def heur_n_block_size(args):
    return triton.next_power_of_2(min(args["N"], triton.cdiv(buf_len_per_core, 4)))


@triton.jit
def reduce_any(a, b):
    return a or b


@libentry()
# @triton.autotune(configs=runtime.get_tuned_config("any"), key=["M", "N"])
@triton.heuristics(
    values={
        "BLOCK_M": heur_m_block_size,
        "BLOCK_N": heur_n_block_size,
    },
)
@triton.jit
def any_kernel_dim(
    inp,
    out,
    M,
    N,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    # Map the program id to the row of inp it should compute.
    pid = ext.program_id(0)
    rows = pid * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]
    inp = inp + rows * N
    out = out + rows
    row_mask = rows < M

    _any = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.int1)
    for off in range(0, N, BLOCK_N):
        cols = off + tl.arange(0, BLOCK_N)[None, :]
        col_mask = cols < N
        mask = row_mask and col_mask

        a = tl.load(inp + cols, mask, other=0.0)
        _any = _any or (a != 0)
    any = tl.reduce(_any, axis=1, combine_fn=reduce_any)
    tl.store(out, any[:, None], row_mask)


@libentry()
@triton.heuristics(
    values={
        "BLOCK_M": heur_m_block_size,
        "BLOCK_N": heur_n_block_size,
    },
)
@triton.jit
def max_kernel_dim(
    in_ptr,
    out_ptr,
    M,
    N,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    xoffset = tl.program_id(0) * BLOCK_M
    xindex = xoffset + tl.arange(0, BLOCK_M)[:, None]
    xmask = xindex < M
    rbase = tl.arange(0, BLOCK_N)[None, :]
    _max = tl.full([BLOCK_M, BLOCK_N], float("-inf"), tl.float32)
    for roffset in range(0, N, BLOCK_N):
        rindex = roffset + rbase
        rmask = rindex < N
        r1 = rindex
        inp = tl.load(
            in_ptr + (r1 + (N * xindex)), rmask & xmask, other=float("-inf")
        ).to(tl.float32)
        inpb = tl.broadcast_to(inp, [BLOCK_M, BLOCK_N])
        _max = tl.maximum(_max, inpb)
    tmp2 = tl.max(_max, axis=1, return_indices=False)[:, None]
    tl.store(out_ptr + xindex, tmp2, xmask)


@libentry()
@triton.jit
def any_word_stage1(in_ptr, mid, n_words, BLOCK_SIZE: tl.constexpr):
    """Stage 1 (int32-word bitmap path) of the global-any reduction.

    Reads the input as raw int32 words (valid whenever element_size divides 4:
    word != 0  <=>  at least one element in that word is nonzero, bit-exact).
    Maps the word to 0 (zero word) / INT32_MAX (nonzero word) with an integer
    select, then reduces each chunk with an int32 max -> INT32_MAX iff the
    chunk contains any nonzero element. Integer-only pipeline (no fcmp->i1
    per-element converts and no i1 OR-tree), which is markedly faster on XPU.
    Masked tail lanes load `other=0` (a zero word) and cannot create a false
    positive. `mid` receives INT32_MAX / 0 per chunk."""
    pid = ext.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    w = tl.load(in_ptr + offs, mask=offs < n_words, other=0)
    m = tl.max(tl.where(w == 0, 0, 2147483647), axis=0)
    tl.store(mid + pid, m)


@libentry()
@triton.jit
def any_word_stage2(mid, out, MID_SIZE, BLOCK_MID: tl.constexpr):
    """Stage 2: single program reduces the per-chunk int32 flags; masked
    lanes load 0 (matches the "zero chunk" encoding) and cannot flip the
    result. Outputs boolean (mx == INT32_MAX)."""
    offs = tl.arange(0, BLOCK_MID)
    m = tl.load(mid + offs, mask=offs < MID_SIZE, other=0)
    mx = tl.max(m, axis=0)
    tl.store(out, mx == 2147483647)


@libentry()
@triton.jit
def any_kernel_1(
    inp,
    mid,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """Stage 1 of the global-any reduction: each program reduces one
    BLOCK_SIZE-sized chunk of the flattened input into a single bool in `mid`.
    Reads the real elements and tests `!= 0`, so unlike the old uint8-view/
    byte-max hack it produces a canonical bool and scans every element (the
    hack passed numel as the byte count, silently scanning only the first
    numel/itemsize elements)."""
    pid = ext.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < n_elements
    val = tl.load(inp + offset, mask=mask, other=0)
    # Nonzero test `val != 0` (matches the proven-correct sibling `all()` and the
    # pre-fix baseline). NaN -> True (correct, torch counts NaN as nonzero).
    # Known edge: XPU codegen mis-evaluates float `!=` on -0.0 as True, so -0.0 is
    # reported nonzero (torch counts -0.0 as zero). This is a *pre-existing* baseline
    # quirk, NOT introduced here. The sign-split `(val>0)|(val<0)` would fix -0.0 but
    # regress NaN to False -> a net functional degradation, so we keep `!= 0`.
    # On XPU only one of {-0.0, NaN} can be correct; NaN matters more and stays
    # baseline-correct. No test exercises NaN/-0.0.
    nz = tl.where(mask, val != 0, False)
    any_val = tl.reduce(nz, axis=0, combine_fn=reduce_any)
    tl.store(mid + pid, any_val)


@libentry()
@triton.jit
def any_kernel_dim_v2(
    inp,
    out,
    M,
    N,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    NEED_MASK: tl.constexpr,
):
    """Per-row `any` over the last axis of a contiguous [M, N] view.

    int1-OR reduction (no fp32 upcast), fixed bounded tiles
    (BLOCK_M x BLOCK_N <= 64x4096), and a NEED_MASK constexpr so that
    evenly-dividing shapes take the unmasked (contiguous DMA) load/store
    path (HARNESS_SUMMARY 2.4). Caller only sets NEED_MASK=False when
    M % BLOCK_M == 0 and N % BLOCK_N == 0, so unmasked accesses are
    in-bounds; tail `other=0.0` keeps the reduction neutral and never
    creates a false positive.
    """
    pid = ext.program_id(0)
    rows = pid * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]
    inp = inp + rows * N
    out = out + rows

    _any = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.int1)
    for off in range(0, N, BLOCK_N):
        cols = off + tl.arange(0, BLOCK_N)[None, :]
        if NEED_MASK:
            a = tl.load(inp + cols, (rows < M) and (cols < N), other=0.0)
        else:
            a = tl.load(inp + cols)
        _any = _any or (a != 0)
    any = tl.reduce(_any, axis=1, combine_fn=reduce_any)
    if NEED_MASK:
        tl.store(out, any[:, None], rows < M)
    else:
        tl.store(out, any[:, None])


@libentry()
@triton.jit
def any_kernel_2(mid, out, MID_SIZE, BLOCK_MID: tl.constexpr):
    """Stage 2: a single program reduces the per-chunk bools from stage 1."""
    offset = tl.arange(0, BLOCK_MID)
    mask = offset < MID_SIZE
    val = tl.load(mid + offset, mask=mask, other=0)
    nz = tl.where(mask, val != 0, False)
    any_val = tl.reduce(nz, axis=0, combine_fn=reduce_any)
    tl.store(out, any_val)


# Per-row 2-stage split reduction for any_dims (see any_dims note below).
# `max_kernel_dim` / `any_kernel_dim` only parallelize over M (grid=cdiv(M,BLOCK_M)),
# so when the reduced dims cover most of the tensor and M is tiny (e.g. dim=[0,1] on a
# 3D tensor -> M=kept-dim ~= 100), the whole N reduction is serialized inside a-few
# programs looping over N -> catastrophic (e.g. [64,512,512] 11.5ms, [100,65536,100]
# 441ms). These two kernels launch grid=(M, cdiv(N,BLOCK_N)) so the N axis is ALSO
# parallelized: stage1 reduces each contiguous BLOCK_N chunk of a row into `mid`,
# stage2 reduces the per-chunk bools of each row.
# BLOCK_N is capped at 8192: with a row base offset pid_m*N (pid_m>0) a fp32 tile of
# 65536 lanes MIS-REDUCES on XPU (verified: [512,32768]/[100,25600] fp32 wrong at
# BLOCK_N=65536, correct and stable at 8192). The pure global any() path (M==1, pid_m==0)
# is unaffected and keeps its larger blocks, so only the M>1 path uses these kernels.
@libentry()
@triton.jit
def any_row_stage1_kernel(inp, mid, N, N_CHUNKS, BLOCK_N: tl.constexpr):
    pid_m = ext.program_id(0)
    pid_c = ext.program_id(1)
    offset = pid_c * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = offset < N
    val = tl.load(inp + pid_m * N + offset, mask=mask, other=0)
    nz = tl.where(mask, val != 0, False)
    any_val = tl.reduce(nz, axis=0, combine_fn=reduce_any)
    tl.store(mid + pid_m * N_CHUNKS + pid_c, any_val)


@libentry()
@triton.jit
def any_row_stage2_kernel(mid, out, MID_N, BLOCK_MID: tl.constexpr):
    pid_m = ext.program_id(0)
    offset = tl.arange(0, BLOCK_MID)
    mask = offset < MID_N
    val = tl.load(mid + pid_m * MID_N + offset, mask=mask, other=0)
    nz = tl.where(mask, val != 0, False)
    any_val = tl.reduce(nz, axis=0, combine_fn=reduce_any)
    tl.store(out + pid_m, any_val)


def _any_dims_reduce(inp, M, N, out_shape):
    """Reduce a contiguous [M, N] view over its N axis (per row), returning a bool
    tensor of shape `out_shape` (reduced dims already collapsed to 1)."""
    BLOCK_N = 8192
    n_chunks = triton.cdiv(N, BLOCK_N)
    out = torch.empty(M, dtype=torch.bool, device=inp.device)
    with torch_device_fn.device(inp.device):
        if n_chunks == 1:
            any_row_stage1_kernel[(M, 1)](
                inp, out, N, 1, BLOCK_N=BLOCK_N, buffer_size_limit=2048
            )
        else:
            mid = torch.empty((M, n_chunks), dtype=torch.bool, device=inp.device)
            any_row_stage1_kernel[(M, n_chunks)](
                inp, mid, N, n_chunks, BLOCK_N=BLOCK_N, buffer_size_limit=2048
            )
            block_mid = triton.next_power_of_2(n_chunks)
            any_row_stage2_kernel[(M,)](
                mid, out, n_chunks, BLOCK_MID=block_mid, buffer_size_limit=2048
            )
    return out.reshape(out_shape)


def any(inp):
    logger.debug("GEMS_KUNLUNXIN ANY")
    n_elements = inp.numel()
    elem = inp.element_size()
    bytes_total = n_elements * elem

    # Faster int32-word bitmap path: valid when the flat byte count is a
    # multiple of 4 and the input is contiguous (flat view without copies).
    # word != 0 <=> some element in the word is nonzero holds bit-exactly for
    # any element size dividing 4 (fp16/bf16/fp32/bool/int*...), including
    # NaN (nonzero bits) and the pre-existing +/-0.0 handling of the old
    # `!= 0` element compare (a -0.0 bit pattern is nonzero here as well).
    if inp.is_contiguous() and bytes_total % 4 == 0:
        view = inp.reshape(-1).view(torch.uint8).view(torch.int32)
        n_words = view.numel()
        block_size = get_block_size_1d(n_words, 4)
        mid_size = triton.cdiv(n_words, block_size)
        block_mid = triton.next_power_of_2(mid_size)
        # empty_strided (not registered by gems) -> native allocator,
        # avoids the per-call gems empty tax on the mid/out buffers.
        mid = torch.empty_strided(
            (mid_size,), (1,), dtype=torch.int32, device=inp.device
        )
        out = torch.empty_strided((), (), dtype=torch.bool, device=inp.device)
        with torch_device_fn.device(inp.device):
            any_word_stage1[(mid_size, 1)](
                view, mid, n_words, block_size, buffer_size_limit=2048
            )
            if mid_size == 1:
                return (mid == 2147483647).reshape([])
            any_word_stage2[(1, 1)](
                mid, out, mid_size, block_mid, buffer_size_limit=2048
            )
        return out

    # generic elementwise two-stage path (any byte alignment / layout)
    block_size = get_block_size_1d(n_elements, elem)
    mid_size = triton.cdiv(n_elements, block_size)
    block_mid = triton.next_power_of_2(mid_size)

    mid = torch.empty_strided((mid_size,), (1,), dtype=torch.bool, device=inp.device)
    out = torch.empty_strided((), (), dtype=torch.bool, device=inp.device)
    with torch_device_fn.device(inp.device):
        any_kernel_1[(mid_size, 1)](
            inp, mid, n_elements, block_size, buffer_size_limit=2048
        )
        if mid_size == 1:
            return mid.reshape([])
        any_kernel_2[(1, 1)](mid, out, mid_size, block_mid, buffer_size_limit=2048)
    return out


def _move_dim_last_contig(inp, dim):
    """Return a contiguous view of `inp` with `dim` moved to the last position.

    Replaces `dim_compress(inp, dim)` (= `permute(order).contiguous()`): the
    trailing `.contiguous()` dispatches through the vendor `contiguous`/
    `copy_` strided kernel (~1.1 GB/s discrete point-to-point) whenever the
    reduced dim is not already last (e.g. [64, N, 64] with dim=1 -> 3D
    transpose copy). `torch.ops.aten._copy_from` is NOT overridden by gems
    -> native strided-copy engine (HARNESS_SUMMARY 3.4), ~100x faster on
    the mid-dim shapes. Zero-copy when the dim is already last/contiguous.
    """
    if dim == inp.ndim - 1 and inp.is_contiguous():
        return inp
    order = [i for i in range(inp.ndim) if i != dim] + [dim]
    permuted = inp.permute(order)
    if permuted.is_contiguous():
        return permuted
    new_shape = tuple(permuted.shape)
    strides = [1] * len(new_shape)
    for i in range(len(new_shape) - 2, -1, -1):
        strides[i] = strides[i + 1] * new_shape[i + 1]
    # empty_strided is not registered by gems -> native allocator/copy.
    dst = torch.empty_strided(
        new_shape, tuple(strides), dtype=inp.dtype, device=inp.device
    )
    torch.ops.aten._copy_from(permuted, dst, False)
    return dst


def any_dim(inp, dim=None, keepdim=False):
    logger.debug("GEMS_KUNLUNXIN ANY_DIM")
    shape = list(inp.shape)
    if dim is None:
        out = any(inp)
        if keepdim:
            out = torch.reshape(out, [1] * inp.ndim)
    else:
        assert dim >= -inp.ndim and dim < inp.ndim, "Invalid dim"
        dim = dim % inp.ndim
        N = shape[dim]
        shape[dim] = 1

        # Contiguous [M, N] view with the reduced dim last (see helper).
        if dim == inp.ndim - 1 and inp.is_contiguous():
            inpc = inp
        else:
            inpc = _move_dim_last_contig(inp, dim)
        M = inpc.numel() // N

        if inp.dtype == torch.bool:
            # bool: int1-OR reduction instead of the fp32-max proxy, whose
            # byte->fp32 lane upcast is ~10x slower at large N
            # ([1024,1048576]: ~160ms vs ~15.7ms, any_dim_xpu2 archive).
            # Fixed bounded tiles; BLOCK_N <= 4096 stays inside the proven
            # tl.reduce/BLOCK<=8192 safe window (HARNESS_SUMMARY 2.5).
            if N <= 512:
                block_m, block_n = 64, triton.next_power_of_2(N)
            elif N <= 4096:
                block_m, block_n = 64, 512
            else:
                block_m, block_n = 8, 4096
            need_mask = (M % block_m != 0) or (N % block_n != 0)
            out = torch.empty(shape, dtype=torch.bool, device=inp.device)
            grid = (triton.cdiv(M, block_m),)
            with torch_device_fn.device(inp.device):
                any_kernel_dim_v2[grid](
                    inpc,
                    out,
                    M,
                    N,
                    BLOCK_M=block_m,
                    BLOCK_N=block_n,
                    NEED_MASK=need_mask,
                    buffer_size_limit=2048,
                )
        elif N >= vector_size * vector_size:
            # float: use max to calculate (per api, op == any).
            # Feed the tensor directly: max_kernel_dim upcasts on load
            # (`.to(tl.float32)`), and on this XPU tree any external
            # fp16/bf16->fp32 cast (`.to(torch.float)` / aten::_to_copy)
            # runs at ~15GB/s (measured [1024,65536] fp16: 22ms cast +
            # 0.74ms kernel vs 0.74ms direct-read), so a pre-cast is a
            # strict loss (the old `large_n_precast` threshold predates
            # this tree's cast path).
            outf = torch.empty(shape, dtype=torch.float, device=inp.device)
            # Fixed tuple grid: host-computed BLOCK_M must match the
            # @triton.heuristics formula (any_dim_xpu2: a mismatched grid
            # under-covers rows and leaves `outf` half-uninitialized).
            block_m = triton.next_power_of_2(
                min(triton.cdiv(M, cluster_num), core_num)
            )
            grid = (triton.cdiv(M, block_m),)
            with torch_device_fn.device(inp.device):
                max_kernel_dim[grid](inpc, outf, M, N, buffer_size_limit=2048)
            out = outf.to(torch.bool)
        else:
            out = torch.empty(shape, dtype=torch.bool, device=inp.device)
            block_m = triton.next_power_of_2(
                min(triton.cdiv(M, cluster_num), core_num)
            )
            grid = (triton.cdiv(M, block_m),)
            with torch_device_fn.device(inp.device):
                any_kernel_dim[grid](inpc, out, M, N, buffer_size_limit=2048)

        if not keepdim:
            out = out.squeeze(dim=dim)
    return out


def any_dims(inp, dim=None, keepdim=False):
    logger.debug("GEMS_KUNLUNXIN ANY_DIMS")

    if dim is None or isinstance(dim, int):
        return any_dim(inp, dim=dim, keepdim=keepdim)
    assert ((i >= -inp.ndim and i < inp.ndim) for i in dim), "Invalid dim"

    shape = list(inp.shape)
    dim = [d % inp.ndim for d in dim]
    inp = dim_compress(inp, dim)
    N = 1
    for i in dim:
        N *= shape[i]
        shape[i] = 1
    M = inp.numel() // N

    # Per-row 2-stage split reduction: parallelizes the N axis on top of M, so the
    # small-M / huge-N cases produced by dim=[0,1] (2D -> M=1; 3D -> M=kept-dim) no
    # longer serialize the entire reduction in a-few programs. Replaces the old
    # max_kernel_dim / any_kernel_dim (grid=cdiv(M,BLOCK_M)) path, which pinned huge
    # shapes at ~11-570ms. `inp` is contiguous after dim_compress -> [M, N] row view.
    if M == 1:
        # All elements reduced -> a global any(); its two-stage reduction (proven,
        # well-tested) parallelizes over chunks and keeps its larger, correct blocks
        # (the pid_m==0 path is immune to the fp32 large-block mis-reduction).
        res = any(inp)
        out = res.reshape(shape)
    else:
        out = _any_dims_reduce(inp, M, N, shape)

    if not keepdim:
        out = out.squeeze(dim=dim)
    return out
