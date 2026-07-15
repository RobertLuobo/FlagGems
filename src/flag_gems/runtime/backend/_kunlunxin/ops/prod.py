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

# from flag_gems import runtime
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import dim_compress, libentry
from flag_gems.utils import triton_lang_extension as ext

from ..utils.block_size_utils import get_block_size_1d

logger = logging.getLogger(__name__)


@triton.jit
def reduce_mul(a, b):
    return a * b


@libentry()
@triton.jit
def prod_kernel_mid(
    inp,
    mid,
    M,
    BLOCK_SIZE: tl.constexpr,
):
    pid = ext.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    inp_ptrs = inp + offset
    mask = offset < M
    inp_val = tl.load(inp_ptrs, mask=mask, other=1.0).to(tl.float32)
    mid_value = tl.reduce(inp_val, axis=0, combine_fn=reduce_mul)
    mid_ptr = mid + pid
    tl.store(mid_ptr, mid_value.to(inp_val.dtype))


@libentry()
@triton.jit
def prod_kernel_result(mid, out, mid_size, BLOCK_MID: tl.constexpr):
    offset = tl.arange(0, BLOCK_MID)
    mid_ptrs = mid + offset
    mask = offset < mid_size
    mid_val = tl.load(mid_ptrs, mask=mask, other=1.0).to(tl.float32)
    prod_val = tl.reduce(mid_val, axis=0, combine_fn=reduce_mul)
    tl.store(out, prod_val)


def prod(inp, *, dtype=None):
    logger.debug("GEMS_KUNLUNXIN PROD")
    if dtype is None:
        dtype = inp.dtype

    M = inp.numel()
    # block_size = triton.next_power_of_2(math.ceil(math.sqrt(M)))
    block_size = get_block_size_1d(M, inp.element_size())
    mid_size = triton.cdiv(M, block_size)
    block_mid = triton.next_power_of_2(mid_size)

    mid = torch.empty((mid_size,), dtype=dtype, device=inp.device)
    out = torch.empty([], dtype=dtype, device=inp.device)

    with torch_device_fn.device(inp.device):
        prod_kernel_mid[(mid_size, 1, 1)](
            inp, mid, M, block_size, buffer_size_limit=2048
        )
        if mid_size == 1:
            return mid.reshape([])
        prod_kernel_result[(1, 1, 1)](
            mid, out, mid_size, block_mid, buffer_size_limit=2048
        )
    return out


def heur_m_block_size(args):
    # For large reduction dim N, assign one row per program (BLOCK_M=1): this
    # maximizes grid parallelism and keeps the accumulator tile [1, BLOCK_N]
    # small. For small N, pack many rows per program so the launch cost is
    # amortized. The old unbounded BLOCK_M = next_pow2(cdiv(M, 12)) built giant
    # [512, BLOCK_N] tiles with only ~8 programs -> catastrophic (see solution).
    if args["N"] >= 2048:
        return 1
    return triton.next_power_of_2(triton.cdiv(args["M"], 12))  # cluster_num


def heur_n_block_size(args):
    import builtins

    return builtins.min(triton.next_power_of_2(args["N"]), 8192)


@libentry()
@triton.heuristics(
    values={
        "BLOCK_M": heur_m_block_size,
        "BLOCK_N": heur_n_block_size,
    },
)
@triton.jit
def prod_kernel(
    inp,
    out,
    M,
    N,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    # Map program id to its rows and pre-offset the base pointer so the inner
    # `inp + cols` access is proven contiguous by OffsetAnalysis (block DMA).
    # Computing `m_offset[:, None] * N + n_offset` inline (old impl) blocks the
    # analysis -> discrete scalar gather.
    pid = ext.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]
    inp = inp + pid * N
    out = out + pid
    row_mask = pid < M

    acc = tl.full((BLOCK_M, BLOCK_N), value=1.0, dtype=tl.float32)
    for off in range(0, N, BLOCK_N):
        cols = off + tl.arange(0, BLOCK_N)[None, :]
        col_mask = cols < N
        inp_vals = tl.load(inp + cols, mask=row_mask & col_mask, other=1.0).to(
            tl.float32
        )
        acc *= inp_vals
    result = tl.reduce(acc, axis=1, combine_fn=reduce_mul)[:, None]
    tl.store(out, result, row_mask)


def prod_dim(inp, dim=None, keepdim=False, *, dtype=None):
    logger.debug("GEMS_KUNLUNXIN PROD_DIM")

    assert dim >= -inp.ndim and dim < inp.ndim, "Invalid dim"
    shape = list(inp.shape)
    dim = dim % inp.ndim
    inp = dim_compress(inp, dim)
    N = shape[dim]
    shape[dim] = 1
    M = inp.numel() // N

    if dtype is None:
        dtype = inp.dtype
    out = torch.empty(shape, dtype=dtype, device=inp.device)
    if not keepdim:
        out = torch.squeeze(out, dim)

    grid = lambda meta: (triton.cdiv(M, meta["BLOCK_M"]),)
    with torch_device_fn.device(inp.device):
        prod_kernel[grid](inp, out, M, N, buffer_size_limit=2048)

    return out
