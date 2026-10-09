# Copyright 2026 FlagOS Contributors.
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
from collections import namedtuple

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)

# The elimination column index ``J`` is passed as a *runtime* argument (not a
# ``tl.constexpr``). The shared vendor ``linalg_lu_factor`` kernels take ``J``
# as constexpr, which re-specializes and recompiles every kernel once per column
# -- for a 1024x1024 matrix that is ~4000 g++ compiles and blows past the test
# timeout. With ``J`` runtime each kernel compiles once and is reused across all
# columns; the math is identical. These kernels are local copies so the shared
# vendor path (used by linalg_lu_factor and friends) is left untouched.

# ruff: noqa: PLR0913


@triton.jit
def _lu_find_pivot_main_kernel(
    LU, PARTIAL_VALUES, PARTIAL_ROWS, M, N, K, J,
    BLOCKS: tl.constexpr, BLOCK_P: tl.constexpr,
):
    pid = tl.program_id(0)
    batch = pid // BLOCKS
    block = pid % BLOCKS
    rows = block * 64 + tl.arange(0, 64)
    values = tl.load(LU + batch * M * N + rows * N + J)
    candidates = tl.where(rows >= J, tl.abs(values), -1.0)
    local = tl.argmax(candidates, axis=0)
    tl.store(PARTIAL_VALUES + batch * BLOCK_P + block, tl.max(candidates, axis=0))
    tl.store(PARTIAL_ROWS + batch * BLOCK_P + block, (block * 64 + local).to(tl.int32))


@triton.jit
def _lu_find_pivot_tail_kernel(
    LU, PARTIAL_VALUES, PARTIAL_ROWS, M, N, K, J, TAIL_START,
    BLOCK_M: tl.constexpr, SLOT, BLOCK_P: tl.constexpr,
):
    batch = tl.program_id(0)
    rows = TAIL_START + tl.arange(0, BLOCK_M)
    values = tl.load(LU + batch * M * N + rows * N + J)
    candidates = tl.where(rows >= J, tl.abs(values), -1.0)
    local = tl.argmax(candidates, axis=0)
    tl.store(PARTIAL_VALUES + batch * BLOCK_P + SLOT, tl.max(candidates, axis=0))
    tl.store(PARTIAL_ROWS + batch * BLOCK_P + SLOT, (TAIL_START + local).to(tl.int32))


@triton.jit
def _lu_finish_pivot_kernel(PARTIAL_VALUES, PARTIAL_ROWS, PIVOTS, K, J, BLOCK_P: tl.constexpr):
    batch = tl.program_id(0)
    blocks = tl.arange(0, BLOCK_P)
    values = tl.load(PARTIAL_VALUES + batch * BLOCK_P + blocks)
    block = tl.argmax(values, axis=0)
    row = tl.load(PARTIAL_ROWS + batch * BLOCK_P + block)
    tl.store(PIVOTS + batch * K + J, row + 1)


@triton.jit
def _lu_swap_rows_kernel(LU, PIVOTS, M, N, K, J, BLOCKS: tl.constexpr, BLOCK_N: tl.constexpr):
    pid = tl.program_id(0)
    batch = pid // BLOCKS
    block = pid % BLOCKS
    columns = block * BLOCK_N + tl.arange(0, BLOCK_N)
    pivot_row = tl.load(PIVOTS + batch * K + J).to(tl.int64) - 1
    base = LU + batch * M * N
    current = tl.load(base + J * N + columns, mask=columns < N, other=0.0)
    pivot = tl.load(base + pivot_row * N + columns, mask=columns < N, other=0.0)
    tl.store(base + J * N + columns, pivot, mask=columns < N)
    tl.store(base + pivot_row * N + columns, current, mask=columns < N)


@triton.jit
def _lu_scale_column_kernel(LU, M, N, J, BLOCKS: tl.constexpr, BLOCK_M: tl.constexpr):
    pid = tl.program_id(0)
    batch = pid // BLOCKS
    block = pid % BLOCKS
    rows = J + 1 + block * BLOCK_M + tl.arange(0, BLOCK_M)
    base = LU + batch * M * N
    pivot = tl.load(base + J * N + J)
    values = tl.load(base + rows * N + J, mask=rows < M, other=0.0)
    tl.store(base + rows * N + J, values / pivot, mask=rows < M)


@triton.jit
def _lu_update_trailing_kernel(LU, M, N, J, ROWS, BLOCKS: tl.constexpr, BLOCK_N: tl.constexpr):
    pid = tl.program_id(0)
    batch = pid // (ROWS * BLOCKS)
    row = J + 1 + (pid // BLOCKS) % ROWS
    block = pid % BLOCKS
    columns = J + 1 + block * BLOCK_N + tl.arange(0, BLOCK_N)
    base = LU + batch * M * N
    mask = (row < M) & (columns < N)
    multiplier = tl.load(base + row * N + J, mask=row < M, other=0.0)
    pivot_row = tl.load(base + J * N + columns, mask=columns < N, other=0.0)
    offsets = base + row * N + columns
    values = tl.load(offsets, mask=mask, other=0.0)
    tl.store(offsets, values - multiplier * pivot_row, mask=mask)


@libentry()
@triton.jit
def _lu_info_kernel(LU, INFO, M, N, stride_batch, K_MAX: tl.constexpr, BLOCK: tl.constexpr):
    """Compute ``info`` for an LU factorization (XPU-safe, vectorized diagonal).

    ``info[b]`` is the 1-indexed position of the first exactly-zero pivot on the
    U diagonal of ``LU[b]``, or ``0`` when every pivot is nonzero. Only an exact
    zero counts: ``NaN``/``Inf`` pivots are not singular (ATen reports
    ``info == 0`` for them) and ``== 0.0`` already excludes them.

    The whole diagonal is read as one masked vector (``diag[j] = LU[b, j, j]`` at
    offset ``b*stride + j*(N+1)``) and the first zero is found with a single
    min-reduce, avoiding the per-iteration scalar ``tl.load(ptr + loopvar)``
    pattern that miscompiles on XPU3.
    """
    pid_b = tl.program_id(0)
    j = tl.arange(0, BLOCK)
    mask = j < K_MAX
    diag_off = pid_b.to(tl.int64) * stride_batch + j.to(tl.int64) * (N + 1)
    pivot = tl.load(LU + diag_off, mask=mask, other=1.0)
    is_singular = (pivot == 0.0) & mask
    idx = tl.where(is_singular, j + 1, K_MAX + 1)
    first = tl.min(idx, axis=0)
    info_val = tl.where(first <= K_MAX, first, 0).to(tl.int32)
    tl.store(INFO + pid_b, info_val)


def _linalg_lu_factor_xpu(input):
    input_contiguous = input.contiguous()
    m, n = input_contiguous.shape[-2:]
    k = min(m, n)
    batch = input_contiguous.numel() // (m * n)
    lu = torch.empty_like(input_contiguous)
    lu.copy_(input_contiguous)
    pivots = torch.empty(
        (*input_contiguous.shape[:-2], k), device=input.device, dtype=torch.int32
    )
    pivot_log = torch.empty_like(pivots)
    blocks_full = m // 64
    tail = m % 64
    slots = blocks_full + (1 if tail else 0)
    block_p = max(1, triton.next_power_of_2(slots))
    partial_values = torch.full(
        (batch, block_p), float("-inf"), device=input.device, dtype=torch.float32
    )
    partial_rows = torch.empty((batch, block_p), device=input.device, dtype=torch.int32)

    with torch_device_fn.device(input.device):
        for j in range(k):
            if blocks_full:
                _lu_find_pivot_main_kernel[(batch * blocks_full,)](
                    lu, partial_values, partial_rows, m, n, k, j,
                    BLOCKS=blocks_full, BLOCK_P=block_p, num_warps=4,
                )
            if tail:
                _lu_find_pivot_tail_kernel[(batch,)](
                    lu, partial_values, partial_rows, m, n, k, j,
                    blocks_full * 64, BLOCK_M=tail, SLOT=blocks_full,
                    BLOCK_P=block_p, num_warps=4,
                )
            _lu_finish_pivot_kernel[(batch,)](
                partial_values, partial_rows, pivot_log, k, j,
                BLOCK_P=block_p, num_warps=4,
            )
            swap_blocks = triton.cdiv(n, 64)
            _lu_swap_rows_kernel[(batch * swap_blocks,)](
                lu, pivot_log, m, n, k, j, BLOCKS=swap_blocks, BLOCK_N=64, num_warps=4,
            )
            if j + 1 < m:
                scale_blocks = triton.cdiv(m - j - 1, 64)
                _lu_scale_column_kernel[(batch * scale_blocks,)](
                    lu, m, n, j, BLOCKS=scale_blocks, BLOCK_M=64, num_warps=4,
                )
            if j + 1 < m and j + 1 < n:
                trailing_rows = m - j - 1
                trailing_blocks = triton.cdiv(n - j - 1, 128)
                _lu_update_trailing_kernel[(batch * trailing_rows * trailing_blocks,)](
                    lu, m, n, j, trailing_rows,
                    BLOCKS=trailing_blocks, BLOCK_N=128, num_warps=4,
                )
    pivots.copy_(pivot_log)
    return lu, pivots


def _lu_with_info_impl(input, pivot=True, check_errors=True):
    LuWithInfoResult = namedtuple("LuWithInfoResult", ["LU", "pivots", "info"])
    if input.dim() < 2:
        raise RuntimeError(
            "torch._lu_with_info: Expected input to have at least 2 "
            f"dimensions, got {input.dim()}"
        )
    if input.dtype not in (torch.float32, torch.float64):
        raise NotImplementedError(
            "FlagGems _lu_with_info currently supports float32 and float64 "
            f"only, got {input.dtype}"
        )
    if not pivot:
        raise NotImplementedError(
            "Kunlunxin _lu_with_info does not support pivot=False: the native "
            "accelerator baseline has no LU-without-pivoting kernel and no "
            "XPU-safe no-pivot factorization is available"
        )
    m, n = input.shape[-2], input.shape[-1]
    batch_shape = input.shape[:-2]
    k = min(m, n)
    if input.numel() == 0:
        lu = input.clone()
        pivots = torch.empty(batch_shape + (k,), device=input.device, dtype=torch.int32)
        info = torch.zeros(batch_shape, device=input.device, dtype=torch.int32)
        return LuWithInfoResult(lu, pivots, info)

    lu, pivots = _linalg_lu_factor_xpu(input)

    batch = lu.numel() // (m * n)
    info = torch.zeros(batch_shape, device=lu.device, dtype=torch.int32)

    if batch > 0:
        stride_batch = m * n
        block = triton.next_power_of_2(k)
        with torch_device_fn.device(lu.device):
            _lu_info_kernel[(batch,)](
                lu, info, m, n, stride_batch,
                K_MAX=k, BLOCK=block, num_warps=4,
            )
    return LuWithInfoResult(lu, pivots, info)


def _lu_with_info(input, pivot=True, check_errors=True):
    logger.debug("GEMS_KUNLUNXIN _LU_WITH_INFO")
    return _lu_with_info_impl(input, pivot, check_errors)
