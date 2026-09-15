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

logger = logging.getLogger("flag_gems").getChild(__name__.lstrip("."))

_WOS_BS = 8192


@libentry()
@triton.jit
def block_diag_wos_kernel(
    out_ptr,
    base_ptr,
    input_stride,
    total,
    LOGB: tl.constexpr,
    LOGT: tl.constexpr,
    BS: tl.constexpr,
):
    """Whole-output kernel for square pow2 blocks: unit-stride contiguous store.

    Output element ``q`` maps to (blk, r, c) with power-of-two shifts:
    ``row = q >> LOGT``, ``col = q & (T-1)``, ``blk = row >> LOGB``,
    ``r = row & (B-1)``, ``c = col & (B-1)``, and the source element is
    ``blk * input_stride + r * B + c`` (in-bounds for every lane after
    clamping: the row index is bounded by ``n*B`` and the block footprint by
    ``input_stride``). ``BS`` is a power of two that divides ``total`` (both
    are powers of two on this path), so every program's store executes and the
    whole output is written; the caller may pass an ``torch.empty`` buffer.
    """
    tile = ext.program_id(0)
    base = tile * BS
    q = base + tl.arange(0, BS)
    rq = tl.minimum(q, total - 1)

    b = 1 << LOGB
    row = rq >> LOGT
    col = rq & ((1 << LOGT) - 1)
    blk = row >> LOGB
    r = row & (b - 1)
    c = col & (b - 1)

    in_blk = (col >> LOGB) == blk
    src = blk * input_stride + r * b + c
    val = tl.load(base_ptr + src)
    val = tl.where(in_blk, val, 0.0)

    if base + BS <= total:
        tl.store(out_ptr + q, val)


@libentry()
@triton.jit
def block_diag_row_strided_kernel(
    out_ptr,
    base_ptr,
    input_stride,
    block_rows,
    block_cols,
    total_cols,
    BC: tl.constexpr,
):
    """Per-(block, row) kernel: contiguous load AND contiguous store.

    Grid is (n, block_rows) so the row index is always valid. The column index
    is clamped to the last valid column so the store stays in-bounds whenever
    BC > block_cols; duplicated lanes re-write the same last cell with its own
    value, which is invisible.
    """
    blk = ext.program_id(0)
    row = ext.program_id(1)
    cols = tl.arange(0, BC)
    cl = tl.minimum(cols, block_cols - 1)

    val = tl.load(base_ptr + blk * input_stride + row * block_cols + cl)
    tl.store(
        out_ptr + (blk * block_rows + row) * total_cols + blk * block_cols + cl,
        val,
    )


@libentry()
@triton.jit
def block_diag_varlen_general_kernel(
    out_ptr,
    ptrs_ptr,
    meta_ptr,
    total_cols,
    BC: tl.constexpr,
):
    """Per-(block, row) kernel for variable-sized blocks from a meta table.

    Clamps both indexes to the block's last valid (row, col): lanes past the
    block's extent re-write the same last-cell with its own value, which is
    invisible, and the addresses stay in bounds unconditionally.
    """
    block_id = ext.program_id(0)
    row = ext.program_id(1)

    base = block_id * 4
    row_off = tl.load(meta_ptr + base + 0)
    col_off = tl.load(meta_ptr + base + 1)
    block_rows = tl.load(meta_ptr + base + 2)
    block_cols = tl.load(meta_ptr + base + 3)

    rr = tl.minimum(row, block_rows - 1)
    cols = tl.arange(0, BC)
    cl = tl.minimum(cols, block_cols - 1)

    ptr_val = tl.load(ptrs_ptr + block_id)
    src_ptr = ptr_val.to(tl.pointer_type(out_ptr.dtype.element_ty))
    val = tl.load(src_ptr + rr * block_cols + cl)
    tl.store(out_ptr + (row_off + rr) * total_cols + col_off + cl, val)


@libentry()
@triton.jit
def _coalesce_blocks_kernel(out_ptr, ptrs_ptr, B2: tl.constexpr, CH: tl.constexpr):
    """Copy ``n`` same-shape blocks (arbitrary host addresses) into a contiguous
    scratch buffer of ``n * B2`` elements.

    One program per block; the block base pointer is a *scalar* load (per
    program, not per lane), so the load and store are affine unit-stride
    vectors. When ``B2`` is a power of two ``CH == B2`` and there is no
    clamping; otherwise the tail lanes clamp-replicate to the block's last
    element (invisible, keeps every address in bounds).

    ``torch.cat`` is intentionally not used here: this file runs inside
    ``flag_gems.use_gems`` when benchmarked, where ``cat`` is overridden by a
    per-block Triton implementation that costs ~60us/block on this backend.
    """
    blk = ext.program_id(0)
    idx = tl.arange(0, CH)
    if CH == B2:
        cl = idx
    else:
        cl = tl.minimum(idx, B2 - 1)
    base = tl.load(ptrs_ptr + blk)
    src_ptr = base.to(tl.pointer_type(out_ptr.dtype.element_ty))
    val = tl.load(src_ptr + cl)
    tl.store(out_ptr + blk * B2 + cl, val)


@libentry()
@triton.jit
def _coalesce16_kernel(
    out_ptr,
    p0,
    p1,
    p2,
    p3,
    p4,
    p5,
    p6,
    p7,
    p8,
    p9,
    p10,
    p11,
    p12,
    p13,
    p14,
    p15,
    B2: tl.constexpr,
    CH: tl.constexpr,
):
    """Single-launch copy of up to 16 same-shape blocks into a contiguous
    scratch buffer of ``n * B2`` elements.

    The block base pointers are passed as plain scalar kernel arguments (no
    device-side pointer table, hence no host-to-device copy) and selected by
    ``program_id`` with a chain of *scalar* ``tl.where`` selects; the dead
    branches are pure integer selects with no memory traffic. Both the load
    (``base + cl``) and the store (``out_ptr + blk * B2 + cl``) are affine
    unit-stride vectors, so they lower to block-DMA. ``CH`` is the power of
    two covering ``B2``; when equal there is no clamping, otherwise the tail
    lanes clamp-replicate to the block's last element (invisible, keeps every
    address in bounds).
    """
    blk = ext.program_id(0)
    idx = tl.arange(0, CH)
    if CH == B2:
        cl = idx
    else:
        cl = tl.minimum(idx, B2 - 1)
    base = tl.where(
        blk == 0,
        p0,
        tl.where(
            blk == 1,
            p1,
            tl.where(
                blk == 2,
                p2,
                tl.where(
                    blk == 3,
                    p3,
                    tl.where(
                        blk == 4,
                        p4,
                        tl.where(
                            blk == 5,
                            p5,
                            tl.where(
                                blk == 6,
                                p6,
                                tl.where(
                                    blk == 7,
                                    p7,
                                    tl.where(
                                        blk == 8,
                                        p8,
                                        tl.where(
                                            blk == 9,
                                            p9,
                                            tl.where(
                                                blk == 10,
                                                p10,
                                                tl.where(
                                                    blk == 11,
                                                    p11,
                                                    tl.where(
                                                        blk == 12,
                                                        p12,
                                                        tl.where(
                                                            blk == 13,
                                                            p13,
                                                            tl.where(
                                                                blk == 14,
                                                                p14,
                                                                p15,
                                                            ),
                                                        ),
                                                    ),
                                                ),
                                            ),
                                        ),
                                    ),
                                ),
                            ),
                        ),
                    ),
                ),
            ),
        ),
    )
    src_ptr = (base + cl).to(tl.pointer_type(out_ptr.dtype.element_ty))
    val = tl.load(src_ptr)
    tl.store(out_ptr + blk * B2 + cl, val)


def _is_pow2(v):
    return v > 0 and (v & (v - 1)) == 0


def _row_tile(block_cols):
    """Constexpr tile width for the per-row kernels (power of two)."""
    return max(1, min(65536, triton.next_power_of_2(max(1, int(block_cols)))))


def block_diag(*tensors):
    """Block diagonal matrix construction (Kunlunxin override)."""
    logger.debug("GEMS_KUNLUNXIN BLOCK_DIAG")

    # Handle case where tensors is passed as a single list/tuple
    if len(tensors) == 1 and isinstance(tensors[0], (list, tuple)):
        tensors = tuple(tensors[0])

    if len(tensors) == 0:
        return torch.tensor([])

    n = len(tensors)

    # Fast check: are all 2D, same shape, same dtype, contiguous?
    t0 = tensors[0]
    if t0.ndim == 2:
        shape0 = t0.shape
        dtype0 = t0.dtype
        fast_path = t0.is_contiguous() and (
            n == 1
            or all(
                t.ndim == 2
                and t.shape == shape0
                and t.dtype == dtype0
                and t.is_contiguous()
                for t in tensors[1:]
            )
        )
    else:
        fast_path = False

    if fast_path:
        block_rows, block_cols = shape0
        block_numel = block_rows * block_cols
        total_rows = n * block_rows
        total_cols = n * block_cols
        device = t0.device

        use_wos = (
            block_rows == block_cols
            and _is_pow2(block_cols)
            and _is_pow2(total_cols)
            and total_rows * total_cols < 1 << 31
        )
        elem_bytes = t0.element_size()
        base_ptr_val = t0.data_ptr()
        if n == 1:
            contiguous = True
        else:
            off1 = (tensors[1].data_ptr() - base_ptr_val) // elem_bytes
            contiguous = off1 == block_numel and (
                n == 2
                or all(
                    (tensors[i].data_ptr() - base_ptr_val) // elem_bytes
                    == i * block_numel
                    for i in range(2, n)
                )
            )
        use_empty = use_wos
        out = (
            torch.empty((total_rows, total_cols), dtype=dtype0, device=device)
            if use_empty
            else torch.zeros((total_rows, total_cols), dtype=dtype0, device=device)
        )

        if block_numel == 0:
            return out

        if n == 1:
            src = t0
            input_stride = block_numel
        elif contiguous:
            src = t0
            input_stride = block_numel
        else:
            src = torch.empty((n, block_numel), dtype=dtype0, device=device)
            if n <= 16:
                bases = [t.data_ptr() for t in tensors] + [0] * (16 - n)
                _coalesce16_kernel[(n,)](
                    src,
                    *bases,
                    B2=block_numel,
                    CH=triton.next_power_of_2(block_numel),
                )
            else:
                ptrs = torch.tensor(
                    [t.data_ptr() for t in tensors], dtype=torch.int64, device=device
                )
                _coalesce_blocks_kernel[(n,)](
                    src,
                    ptrs,
                    B2=block_numel,
                    CH=triton.next_power_of_2(block_numel),
                )
            input_stride = block_numel

        with torch_device_fn.device(device):
            if use_wos:
                # Whole-output kernel: contiguous store, shift-based index math.
                total = total_rows * total_cols
                bs = min(_WOS_BS, triton.next_power_of_2(max(1, total)))
                block_diag_wos_kernel[((total + bs - 1) // bs,)](
                    out,
                    src,
                    input_stride,
                    total,
                    LOGB=int(math.log2(block_cols)),
                    LOGT=int(math.log2(total_cols)),
                    BS=bs,
                )
            else:
                # Per-row kernel: both load and store contiguous.
                bc = _row_tile(block_cols)
                block_diag_row_strided_kernel[(n, block_rows)](
                    out,
                    src,
                    input_stride,
                    block_rows,
                    block_cols,
                    total_cols,
                    BC=bc,
                )
        return out

    # General path: normalize, compute dtype, handle mixed shapes
    tensors_2d = []
    for t in tensors:
        if t.ndim == 0:
            tensors_2d.append(t.unsqueeze(0).unsqueeze(0))
        elif t.ndim == 1:
            tensors_2d.append(t.unsqueeze(0))
        else:
            assert t.ndim == 2, f"Expected 0D, 1D, or 2D tensor, got {t.ndim}D"
            tensors_2d.append(t)

    total_rows = sum(t.shape[0] for t in tensors_2d)
    total_cols = sum(t.shape[1] for t in tensors_2d)

    out_dtype = tensors_2d[0].dtype
    for t in tensors_2d[1:]:
        out_dtype = torch.result_type(
            torch.empty(0, dtype=out_dtype), torch.empty(0, dtype=t.dtype)
        )
    device = tensors_2d[0].device

    out = torch.zeros((total_rows, total_cols), dtype=out_dtype, device=device)

    meta_list = []
    ptrs_list = []
    src_tensors = []  # Keep references to prevent GC
    cur_row = 0
    cur_col = 0
    max_rows = 0
    max_cols = 0
    max_numel = 0

    for t in tensors_2d:
        rows, cols = t.shape
        numel = rows * cols
        meta_list.extend([cur_row, cur_col, rows, cols])
        if numel > 0:
            src = (
                t
                if (t.is_contiguous() and t.dtype == out_dtype)
                else t.contiguous().to(out_dtype)
            )
            ptrs_list.append(src.data_ptr())
            src_tensors.append(src)
        else:
            ptrs_list.append(0)
        max_rows = max(max_rows, rows)
        max_cols = max(max_cols, cols)
        max_numel = max(max_numel, numel)
        cur_row += rows
        cur_col += cols

    if max_numel == 0:
        return out

    ptrs = torch.tensor(ptrs_list, dtype=torch.int64, device=device)
    meta = torch.tensor(meta_list, dtype=torch.int64, device=device)

    bc = _row_tile(max_cols)
    grid = (len(tensors_2d), max_rows)

    with torch_device_fn.device(device):
        block_diag_varlen_general_kernel[grid](
            out,
            ptrs,
            meta,
            total_cols,
            BC=bc,
        )

    return out
