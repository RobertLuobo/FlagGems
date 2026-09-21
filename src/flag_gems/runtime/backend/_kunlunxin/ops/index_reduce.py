import logging
import math

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)


@libentry()
@triton.jit
def _index_reduce_kernel(
    inp,
    index,
    source,
    output,
    output_dim_size,
    inner_size,
    SOURCE_DIM_SIZE: tl.constexpr,
    REDUCE: tl.constexpr,
    INCLUDE_SELF: tl.constexpr,
):
    output_offset = tl.program_id(0)
    inner_offset = output_offset % inner_size
    output_dim_offset = (output_offset // inner_size) % output_dim_size
    outer_offset = output_offset // (output_dim_size * inner_size)
    self_value = tl.load(inp + output_offset).to(tl.float32)

    if REDUCE == 0:
        accumulator = self_value if INCLUDE_SELF else 1.0
    elif REDUCE == 1:
        accumulator = self_value if INCLUDE_SELF else 0.0
    elif REDUCE == 2:
        accumulator = self_value if INCLUDE_SELF else -float("inf")
    else:
        accumulator = self_value if INCLUDE_SELF else float("inf")
    count = 1 if INCLUDE_SELF else 0

    for source_dim_offset in tl.range(0, SOURCE_DIM_SIZE):
        selected = tl.load(index + source_dim_offset) == output_dim_offset
        source_offset = (
            outer_offset * SOURCE_DIM_SIZE + source_dim_offset
        ) * inner_size + inner_offset
        value = tl.load(source + source_offset).to(tl.float32)
        if REDUCE == 0:
            accumulator = tl.where(selected, accumulator * value, accumulator)
        elif REDUCE == 1:
            accumulator = tl.where(selected, accumulator + value, accumulator)
        elif REDUCE == 2:
            accumulator = tl.where(
                selected, tl.maximum(accumulator, value), accumulator
            )
        else:
            accumulator = tl.where(
                selected, tl.minimum(accumulator, value), accumulator
            )
        count += selected.to(tl.int32)

    if REDUCE == 1:
        accumulator /= count
    if not INCLUDE_SELF:
        accumulator = tl.where(count == 0, self_value, accumulator)
    tl.store(output + output_offset, accumulator)


@libentry()
@triton.jit
def _index_reduce_fill_kernel(
    target,
    size,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    tl.store(target + offsets, tl.full([BLOCK], -1, tl.int32), mask=offsets < size)


@libentry()
@triton.jit
def _index_reduce_invert_kernel(
    index,
    inverse,
    out_dim_size,
    index_size,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < index_size
    dst = tl.load(index + offsets, mask=mask, other=0).to(tl.int64)
    dst = tl.minimum(tl.maximum(dst, 0), out_dim_size - 1)
    tl.store(inverse + dst, offsets, mask=mask)


@libentry()
@triton.jit
def _index_reduce_dup_kernel(
    index,
    inverse,
    dup_flag,
    index_size,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < index_size
    dst = tl.load(index + offsets, mask=mask, other=0).to(tl.int64)
    dst = tl.minimum(tl.maximum(dst, 0), 1073741823)
    src = tl.load(inverse + dst, mask=mask, other=-1)
    bad = ((src.to(tl.int64) != offsets) & mask).to(tl.int32)
    tl.atomic_add(dup_flag, tl.sum(bad, axis=0))


@libentry()
@triton.jit
def _index_reduce_unique_kernel(
    inp,
    inverse,
    source,
    output,
    out_dim_size,
    source_dim_size,
    inner_size,
    TOTAL,
    REDUCE: tl.constexpr,
    INCLUDE_SELF: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < TOTAL
    inner_offset = offsets % inner_size
    outer_offset = offsets // (out_dim_size * inner_size)
    output_dim_offset = (offsets // inner_size) % out_dim_size
    self_value = tl.load(inp + offsets, mask=mask, other=0.0).to(tl.float32)

    source_dim_offset = tl.load(inverse + output_dim_offset, mask=mask, other=-1)
    matched = (source_dim_offset >= 0) & mask
    clamped = tl.where(matched, source_dim_offset, 0).to(tl.int64)
    source_offset = (
        outer_offset.to(tl.int64) * source_dim_size + clamped
    ) * inner_size + inner_offset
    value = tl.load(source + source_offset, mask=mask, other=0.0).to(tl.float32)

    if REDUCE == 0:
        accumulator = self_value if INCLUDE_SELF else 1.0
    elif REDUCE == 1:
        accumulator = self_value if INCLUDE_SELF else 0.0
    elif REDUCE == 2:
        accumulator = self_value if INCLUDE_SELF else -float("inf")
    else:
        accumulator = self_value if INCLUDE_SELF else float("inf")
    count = 1 if INCLUDE_SELF else 0

    if REDUCE == 0:
        accumulator = tl.where(matched, accumulator * value, accumulator)
    elif REDUCE == 1:
        accumulator = tl.where(matched, accumulator + value, accumulator)
    elif REDUCE == 2:
        accumulator = tl.where(matched, tl.maximum(accumulator, value), accumulator)
    else:
        accumulator = tl.where(matched, tl.minimum(accumulator, value), accumulator)
    count += matched.to(tl.int32)

    if REDUCE == 1:
        accumulator /= count
    if not INCLUDE_SELF:
        accumulator = tl.where(count == 0, self_value, accumulator)
    tl.store(output + offsets, accumulator, mask=mask)


_REDUCTIONS = {"prod": 0, "mean": 1, "amax": 2, "amin": 3}

_BLOCK = 1024
_DUP_BLOCK = 256


def _can_use_unique_path(index, out_dim_size):
    return index.numel() > 0 and out_dim_size > 0


def index_reduce_(inp, dim, index, source, reduce, *, include_self=True):
    logger.debug("GEMS_KUNLUNXIN INDEX_REDUCE_")
    if reduce not in _REDUCTIONS:
        raise RuntimeError(
            f"index_reduce(): Expected reduce to be one of prod, mean, amax or amin but got {reduce}."
        )
    if inp.ndim == 0:
        raise IndexError(
            "index_reduce_(): Expected self to have non-zero dimensionality"
        )

    dim %= inp.ndim
    if index.ndim != 1 or index.numel() != source.shape[dim]:
        raise RuntimeError(
            "index_reduce_(): Expected index to be a vector matching source.size(dim)"
        )
    if any(
        source.shape[axis] != inp.shape[axis] for axis in range(inp.ndim) if axis != dim
    ):
        raise RuntimeError(
            "index_reduce_(): source must match self outside the reduced dimension"
        )

    input_contiguous = inp.contiguous()
    source = source.contiguous()
    index = index.contiguous()
    result = input_contiguous.clone()
    out_dim_size = inp.shape[dim]
    index_size = index.numel()
    inner_size = math.prod(inp.shape[dim + 1 :])
    reduce_id = _REDUCTIONS[reduce]

    dispatch_unique = (
        _can_use_unique_path(index, out_dim_size)
        and index.dtype in (torch.int32, torch.int64)
    )

    with torch_device_fn.device(inp.device):
        if dispatch_unique:
            inverse = torch.empty(out_dim_size, dtype=torch.int32, device=inp.device)
            dup_flag = torch.zeros(1, dtype=torch.int32, device=inp.device)
            _index_reduce_fill_kernel[(triton.cdiv(out_dim_size, _BLOCK),)](
                inverse, out_dim_size, BLOCK=_BLOCK
            )
            _index_reduce_invert_kernel[(triton.cdiv(index_size, _BLOCK),)](
                index, inverse, out_dim_size, index_size, BLOCK=_BLOCK
            )
            _index_reduce_dup_kernel[(triton.cdiv(index_size, _DUP_BLOCK),)](
                index, inverse, dup_flag, index_size, BLOCK=_DUP_BLOCK
            )
            dispatch_unique = int(dup_flag.item()) == 0

        if dispatch_unique:
            total = result.numel()
            _index_reduce_unique_kernel[(triton.cdiv(total, _BLOCK),)](
                input_contiguous,
                inverse,
                source,
                result,
                out_dim_size,
                index_size,
                inner_size,
                total,
                REDUCE=reduce_id,
                INCLUDE_SELF=include_self,
                BLOCK=_BLOCK,
            )
        else:
            _index_reduce_kernel[(result.numel(),)](
                input_contiguous,
                index,
                source,
                result,
                inp.shape[dim],
                inner_size,
                SOURCE_DIM_SIZE=index.numel(),
                REDUCE=reduce_id,
                INCLUDE_SELF=include_self,
                isCloseVectorization=True,
                buffer_size_limit=2048,
            )
    inp.copy_(result)
    return inp
