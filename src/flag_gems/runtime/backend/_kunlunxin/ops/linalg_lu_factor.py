import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn

logger = logging.getLogger(__name__)


@triton.jit
def _lu_find_pivot_main_kernel(
    LU,
    PARTIAL_VALUES,
    PARTIAL_ROWS,
    M,
    N,
    K,
    J,
    BLOCKS: tl.constexpr,
    BLOCK_P: tl.constexpr,
):
    """Per-block pivot candidate search over 64-row aligned main blocks.

    Every block covers exactly 64 valid rows (blocks_full = M // 64), so no
    masked load / no tail garbage: XPU mis-compiles tl.argmax when the masked
    vector length is smaller than the block size (see solution notes), and the
    block-parallel partial results are merged by _lu_finish_pivot_kernel.

    NOTE: J is deliberately a runtime scalar (not constexpr): the step index
    changes every iteration of the elimination loop, so a constexpr J would
    force one Triton recompilation per step (and per kernel), which for a
    512x512 factorization means ~2-3k JIT compilations and minutes of dead
    time. With J as a plain argument the kernels compile once per shape.
    """
    pid = tl.program_id(0)
    batch = pid // BLOCKS
    block = pid % BLOCKS
    rows = block * 64 + tl.arange(0, 64)
    values = tl.load(LU + batch * M * N + rows * N + J)
    candidates = tl.where(rows >= J, tl.abs(values), -1.0)
    local = tl.argmax(candidates, axis=0)
    tl.store(PARTIAL_VALUES + batch * BLOCK_P + block, tl.max(candidates, axis=0))
    tl.store(
        PARTIAL_ROWS + batch * BLOCK_P + block,
        (block * 64 + local).to(tl.int32),
    )


@triton.jit
def _lu_find_pivot_tail_kernel(
    LU,
    PARTIAL_VALUES,
    PARTIAL_ROWS,
    M,
    N,
    K,
    J,
    TAIL_START: tl.constexpr,
    BLOCK_M: tl.constexpr,
    SLOT: tl.constexpr,
    BLOCK_P: tl.constexpr,
):
    # Tail segment: rows [TAIL_START, M), BLOCK_M == M - TAIL_START exactly.
    # J is a runtime scalar for the same reason as _lu_find_pivot_main_kernel:
    # one compilation per shape instead of one per elimination step.
    batch = tl.program_id(0)
    rows = TAIL_START + tl.arange(0, BLOCK_M)
    values = tl.load(LU + batch * M * N + rows * N + J)
    candidates = tl.where(rows >= J, tl.abs(values), -1.0)
    local = tl.argmax(candidates, axis=0)
    tl.store(PARTIAL_VALUES + batch * BLOCK_P + SLOT, tl.max(candidates, axis=0))
    tl.store(
        PARTIAL_ROWS + batch * BLOCK_P + SLOT,
        (TAIL_START + local).to(tl.int32),
    )


@triton.jit
def _lu_finish_pivot_kernel(
    PARTIAL_VALUES,
    PARTIAL_ROWS,
    PIVOTS,
    K,
    J,
    BLOCK_P: tl.constexpr,
):
    # BLOCK_P = next_pow2(slots); pad slots are pre-filled with -inf so the
    # full-block argmax below selects a real slot even when slots < BLOCK_P
    # (no masked reduce involved). J is a runtime scalar (see main kernel).
    batch = tl.program_id(0)
    blocks = tl.arange(0, BLOCK_P)
    values = tl.load(PARTIAL_VALUES + batch * BLOCK_P + blocks)
    block = tl.argmax(values, axis=0)
    row = tl.load(PARTIAL_ROWS + batch * BLOCK_P + block)
    tl.store(PIVOTS + batch * K + J, row + 1)


@triton.jit
def _lu_swap_rows_kernel(
    LU, PIVOTS, M, N, K, J, BLOCKS: tl.constexpr, BLOCK_N: tl.constexpr
):
    # J is a runtime scalar (see _lu_find_pivot_main_kernel): with a constexpr
    # J every elimination step would recompile the kernel.
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
def _lu_scale_column_kernel(
    LU, M, N, J, BLOCKS: tl.constexpr, BLOCK_M: tl.constexpr
):
    # J is a runtime scalar (see _lu_find_pivot_main_kernel).
    pid = tl.program_id(0)
    batch = pid // BLOCKS
    block = pid % BLOCKS
    rows = J + 1 + block * BLOCK_M + tl.arange(0, BLOCK_M)
    base = LU + batch * M * N
    pivot = tl.load(base + J * N + J)
    values = tl.load(base + rows * N + J, mask=rows < M, other=0.0)
    tl.store(base + rows * N + J, values / pivot, mask=rows < M)


@triton.jit
def _lu_update_trailing_kernel(
    LU,
    M,
    N,
    J,
    ROW_BLOCKS: tl.constexpr,
    COL_BLOCKS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Trailing rank-1 update A[J+1:, J+1:] -= L[/, J] * U[J, /] on a 2D tile.

    One program covers a [BLOCK_M, BLOCK_N] tile (4x128).  The 2D form with a
    [BLOCK_M, BLOCK_N] mask is REQUIRED on XPU: the 1-row form whose store mask
    is `(scalar < M) & (vector < N)` with J as a runtime scalar silently
    mis-compiles on this backend (rows of the trailing block are updated with
    garbage column-0 values; verified 2026-09-04).  Larger tiles (64x128) were
    measured slower here (large int64 offset-tile expansion in
    ConvertTritonXPUToLLVM), so 4x128 is the sweet spot: identical per-row work
    to the 1-row form but 4x fewer programs.

    J is a runtime scalar (see _lu_find_pivot_main_kernel): a constexpr J would
    force one recompilation per elimination step.
    """
    pid = tl.program_id(0)
    batch = pid // (ROW_BLOCKS * COL_BLOCKS)
    block = pid % (ROW_BLOCKS * COL_BLOCKS)
    bm = block // COL_BLOCKS
    bn = block % COL_BLOCKS
    rows = J + 1 + bm * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = J + 1 + bn * BLOCK_N + tl.arange(0, BLOCK_N)
    base = LU + batch * M * N
    row_mask = rows < M
    col_mask = cols < N
    mask = row_mask[:, None] & col_mask[None, :]
    multiplier = tl.load(base + rows * N + J, mask=row_mask, other=0.0)
    pivot_row = tl.load(base + J * N + cols, mask=col_mask, other=0.0)
    offsets = base + rows[:, None] * N + cols[None, :]
    values = tl.load(offsets, mask=mask, other=0.0)
    tl.store(offsets, values - multiplier[:, None] * pivot_row[None, :], mask=mask)


def _check_linalg_lu_factor(input, pivot):
    if input.dim() < 2:
        raise RuntimeError(
            "torch.linalg.lu_factor: Expected input to have at least 2 dimensions, "
            f"got {input.dim()}"
        )
    if input.dtype not in (torch.float32, torch.float64):
        raise NotImplementedError(
            "FlagGems linalg_lu_factor currently supports float32 and float64 only, "
            f"got {input.dtype}"
        )
    if input.shape[-2] == 0 or input.shape[-1] == 0:
        raise NotImplementedError(
            "FlagGems linalg_lu_factor currently does not support empty matrices"
        )
    if not isinstance(pivot, bool):
        raise TypeError(f"pivot must be a bool, got {type(pivot)}")


def _linalg_lu_factor(input, pivot):
    _check_linalg_lu_factor(input, pivot)
    if not pivot:
        raise NotImplementedError(
            "Kunlunxin linalg_lu_factor does not support pivot=False: "
            "the vendor lu_factor_ex primitive rejects it and no XPU-safe "
            "no-pivot kernel is available"
        )

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
    # Segment the pivot search into exact 64-row main blocks plus an exact
    # tail block (tail rows = M % 64).  XPU mis-compiles tl.argmax with masked
    # loads whose valid length is smaller than the block size, so every block
    # covers exactly its valid rows and blocks are padded to a power of two
    # with -inf for the final argmax merge.
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
                    lu,
                    partial_values,
                    partial_rows,
                    m,
                    n,
                    k,
                    j,
                    BLOCKS=blocks_full,
                    BLOCK_P=block_p,
                    num_warps=4,
                )
            if tail:
                _lu_find_pivot_tail_kernel[(batch,)](
                    lu,
                    partial_values,
                    partial_rows,
                    m,
                    n,
                    k,
                    j,
                    TAIL_START=blocks_full * 64,
                    BLOCK_M=tail,
                    SLOT=blocks_full,
                    BLOCK_P=block_p,
                    num_warps=4,
                )
            _lu_finish_pivot_kernel[(batch,)](
                partial_values,
                partial_rows,
                pivot_log,
                k,
                j,
                BLOCK_P=block_p,
                num_warps=4,
            )
            swap_blocks = triton.cdiv(n, 64)
            _lu_swap_rows_kernel[(batch * swap_blocks,)](
                lu,
                pivot_log,
                m,
                n,
                k,
                j,
                BLOCKS=swap_blocks,
                BLOCK_N=64,
                num_warps=4,
            )
            if j + 1 < m:
                scale_blocks = triton.cdiv(m - j - 1, 64)
                _lu_scale_column_kernel[(batch * scale_blocks,)](
                    lu,
                    m,
                    n,
                    j,
                    BLOCKS=scale_blocks,
                    BLOCK_M=64,
                    num_warps=4,
                )
            if j + 1 < m and j + 1 < n:
                row_blocks = triton.cdiv(m - j - 1, 4)
                col_blocks = triton.cdiv(n - j - 1, 128)
                _lu_update_trailing_kernel[(batch * row_blocks * col_blocks,)](
                    lu,
                    m,
                    n,
                    j,
                    ROW_BLOCKS=row_blocks,
                    COL_BLOCKS=col_blocks,
                    BLOCK_M=4,
                    BLOCK_N=128,
                    num_warps=4,
                )
    pivots.copy_(pivot_log)
    return lu, pivots


def linalg_lu_factor(input, *, pivot=True):
    logger.debug("GEMS_KUNLUNXIN LINALG_LU_FACTOR")
    return _linalg_lu_factor(input, pivot)


def _resolve_linalg_lu_factor_out_args(input, LU, pivots):
    if LU is None or pivots is None:
        raise TypeError(
            "linalg_lu_factor(): LU and pivots must both be provided " "for out variant"
        )
    if LU.device != input.device or pivots.device != input.device:
        raise RuntimeError("linalg_lu_factor(): out tensors must be on input's device")
    if LU.dtype != input.dtype:
        raise RuntimeError("linalg_lu_factor(): LU out tensor must match input dtype")
    if pivots.dtype != torch.int32:
        raise RuntimeError(
            "linalg_lu_factor(): pivots out tensor must have dtype int32"
        )
    return LU, pivots


def linalg_lu_factor_out(input, *, pivot=True, LU=None, pivots=None):
    logger.debug("GEMS_KUNLUNXIN LINALG_LU_FACTOR_OUT")
    lu_out, pivots_out = _resolve_linalg_lu_factor_out_args(input, LU, pivots)
    lu, pivots_result = _linalg_lu_factor(input, pivot)
    lu_out.resize_(lu.shape)
    pivots_out.resize_(pivots_result.shape)
    lu_out.copy_(lu)
    pivots_out.copy_(pivots_result)
    return lu_out, pivots_out
