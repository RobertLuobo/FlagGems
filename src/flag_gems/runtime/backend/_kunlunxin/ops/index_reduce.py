import logging
import math

import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)

_REDUCTIONS = {"prod": 0, "mean": 1, "amax": 2, "amin": 3}


@libentry()
@triton.jit(do_not_specialize=["output_dim_size", "inner_size", "source_dim_size"])
def _index_reduce_kernel(
    inp,
    index,
    source,
    output,
    output_dim_size,
    inner_size,
    source_dim_size,
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

    source_dim_offset = 0
    while source_dim_offset < source_dim_size:
        selected = tl.load(index + source_dim_offset) == output_dim_offset
        source_offset = (
            outer_offset * source_dim_size + source_dim_offset
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
        source_dim_offset += 1

    if REDUCE == 1:
        accumulator /= count
    if not INCLUDE_SELF:
        accumulator = tl.where(count == 0, self_value, accumulator)
    tl.store(output + output_offset, accumulator)


def _validate(inp, dim, index, source, reduce):
    assert reduce in _REDUCTIONS, f"Unsupported reduce: {reduce}"
    assert inp.ndim > 0, "index_reduce_(): Expected self to have non-zero dimensionality"
    d = dim % inp.ndim
    assert (
        index.ndim == 1 and index.numel() == source.shape[d]
    ), "index_reduce_(): Expected index to be a vector matching source.size(dim)"
    assert not any(
        source.shape[axis] != inp.shape[axis]
        for axis in range(inp.ndim)
        if axis != d
    ), "index_reduce_(): source must match self outside the reduced dimension"
    return d


def _run(inp, dim, index, source, reduce, include_self):
    d = _validate(inp, dim, index, source, reduce)
    input_contiguous = inp.contiguous()
    source = source.contiguous()
    index = index.contiguous()
    result = input_contiguous.clone()
    inner_size = math.prod(inp.shape[d + 1 :])
    with torch_device_fn.device(inp.device):
        _index_reduce_kernel[(result.numel(),)](
            input_contiguous,
            index,
            source,
            result,
            inp.shape[d],
            inner_size,
            index.numel(),
            REDUCE=_REDUCTIONS[reduce],
            INCLUDE_SELF=include_self,
        )
    return result


def index_reduce_(inp, dim, index, source, reduce, *, include_self=True):
    logger.debug("GEMS_KUNLUNXIN INDEX_REDUCE_")
    result = _run(inp, dim, index, source, reduce, include_self)
    inp.copy_(result)
    return inp


def index_reduce(inp, dim, index, source, reduce, *, include_self=True):
    logger.debug("GEMS_KUNLUNXIN INDEX_REDUCE")
    return _run(inp, dim, index, source, reduce, include_self)


def index_reduce_out(inp, dim, index, source, reduce, *, include_self=True, out=None):
    logger.debug("GEMS_KUNLUNXIN INDEX_REDUCE_OUT")
    if out is None:
        return _run(inp, dim, index, source, reduce, include_self)
    if out.dtype != inp.dtype:
        raise RuntimeError(
            f"Expected out tensor to have dtype {inp.dtype}, but got {out.dtype} instead"
        )
    if out.device != inp.device:
        raise RuntimeError(
            f"Expected out tensor to be on device {inp.device}, but got {out.device} instead"
        )
    result = _run(inp, dim, index, source, reduce, include_self)
    if tuple(out.shape) != tuple(result.shape):
        out.resize_(result.shape)
    out.copy_(result)
    return out
