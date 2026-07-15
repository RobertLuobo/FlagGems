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


# torch.all: Tests if all elements in input evaluate to True. If the dtype of input
#            is not BOOL, then test if all elements in input evaluate to non-zero value
# In triton function, test if all elements in input evaluate to non-zero value is ok.

cluster_num = 12
core_num = 64
buf_len_per_core = 2048
vector_size = 16


def heur_m_block_size(args):
    M = args["M"]
    # For very small M, use minimum BLOCK_M of 1
    block_m = min(triton.cdiv(M, cluster_num), core_num)
    return triton.next_power_of_2(max(block_m, 1))


def heur_n_block_size(args):
    N = args["N"]
    # For very small N, use minimum BLOCK_N of 1
    block_n = min(N, 512)
    return triton.next_power_of_2(max(block_n, 1))


@triton.jit
def reduce_all(a, b):
    return a and b


@libentry()
@triton.jit
def all_kernel_1(
    inp,
    mid,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """Stage 1 of the global-all reduction: each program reduces one
    BLOCK_SIZE-sized chunk of the flattened input into a single bool in `mid`.
    Splitting the work across `cdiv(n_elements, BLOCK_SIZE)` programs restores
    parallelism (the old single-program loop ran at ~7 GB/s on one core)."""
    pid = ext.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < n_elements
    val = tl.load(inp + offset, mask=mask, other=1)
    # masked-out lanes must be True (identity for AND); do not rely on `other`.
    nz = tl.where(mask, val != 0, True)
    result = tl.reduce(nz, axis=0, combine_fn=reduce_all)
    tl.store(mid + pid, result)


@libentry()
@triton.jit
def all_kernel_2(
    mid,
    out,
    mid_size,
    BLOCK_MID: tl.constexpr,
):
    """Stage 2: a single program reduces the per-chunk bools from stage 1."""
    offset = tl.arange(0, BLOCK_MID)
    mask = offset < mid_size
    val = tl.load(mid + offset, mask=mask, other=1)
    nz = tl.where(mask, val != 0, True)
    result = tl.reduce(nz, axis=0, combine_fn=reduce_all)
    tl.store(out, result)


@libentry()
@triton.heuristics(
    values={
        "BLOCK_M": heur_m_block_size,
        "BLOCK_N": heur_n_block_size,
    },
)
@triton.jit
def all_kernel_dim(
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

    _all = tl.full([BLOCK_M, BLOCK_N], value=1, dtype=tl.int1)
    for off in range(0, N, BLOCK_N):
        cols = off + tl.arange(0, BLOCK_N)[None, :]
        col_mask = cols < N
        mask = row_mask and col_mask

        a = tl.load(inp + cols, mask, other=1.0)
        _all = _all and (a != 0)
    all = tl.reduce(_all, axis=1, combine_fn=reduce_all)
    tl.store(out, all[:, None], row_mask)


def all(inp):
    logger.debug("GEMS_KUNLUNXIN ALL")
    n_elements = inp.numel()
    block_size = get_block_size_1d(n_elements, inp.element_size())
    mid_size = triton.cdiv(n_elements, block_size)
    block_mid = triton.next_power_of_2(mid_size)

    mid = torch.empty((mid_size,), dtype=torch.bool, device=inp.device)
    out = torch.empty([], dtype=torch.bool, device=inp.device)
    with torch_device_fn.device(inp.device):
        all_kernel_1[(mid_size, 1, 1)](
            inp, mid, n_elements, block_size, buffer_size_limit=2048
        )
        if mid_size == 1:
            return mid.reshape([])
        all_kernel_2[(1, 1, 1)](mid, out, mid_size, block_mid, buffer_size_limit=2048)
    return out


def all_dim(inp, dim=None, keepdim=False):
    logger.debug("GEMS_KUNLUNXIN ALL_DIM")
    shape = list(inp.shape)
    orig_ndim = inp.ndim

    if dim is None:
        out = all(inp)
        if keepdim:
            out = torch.reshape(out, [1] * orig_ndim)
        return out

    assert dim >= -orig_ndim and dim < orig_ndim, "Invalid dim"
    dim = dim % orig_ndim
    N = shape[dim]
    inp = dim_compress(inp, dim)
    shape[dim] = 1
    M = inp.numel() // N

    if inp.dtype != torch.bool and M * N <= 64:
        inp = inp != 0

    out = torch.empty(shape, dtype=torch.bool, device=inp.device)
    grid = lambda meta: (max(triton.cdiv(M, meta["BLOCK_M"]), 1),)
    with torch_device_fn.device(inp.device):
        all_kernel_dim[grid](inp, out, M, N, buffer_size_limit=2048)

    if not keepdim and out.ndim > 0:
        out = out.squeeze(dim) if dim < out.ndim else out
    return out


def all_dims(inp, dim=None, keepdim=False):
    logger.debug("GEMS_KUNLUNXIN ALL_DIMS")

    if dim is None or isinstance(dim, int):
        return all_dim(inp, dim=dim, keepdim=keepdim)
    orig_ndim = inp.ndim
    assert ((i >= -orig_ndim and i < orig_ndim) for i in dim), "Invalid dim"

    shape = list(inp.shape)
    dim = [d % orig_ndim for d in dim]
    inp = dim_compress(inp, dim)
    N = 1
    for i in dim:
        N *= shape[i]
        shape[i] = 1
    M = inp.numel() // N

    if inp.dtype != torch.bool and M * N <= 64:
        inp = inp != 0

    out = torch.empty(shape, dtype=torch.bool, device=inp.device)
    grid = lambda meta: (max(triton.cdiv(M, meta["BLOCK_M"]), 1),)
    with torch_device_fn.device(inp.device):
        all_kernel_dim[grid](inp, out, M, N, buffer_size_limit=2048)

    if not keepdim:
        for d in sorted(dim):
            if out.ndim > 0:
                out = out.squeeze(dim=d)
    return out
