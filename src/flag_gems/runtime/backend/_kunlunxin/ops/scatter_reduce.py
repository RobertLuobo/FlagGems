import logging

import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)


@libentry()
@triton.jit
def _scatter_reduce_kernel(
    inp,
    index,
    src,
    out,
    input_size0,
    input_size1,
    input_size2,
    index_size0,
    index_size1,
    index_size2,
    src_size1,
    src_size2,
    DIM: tl.constexpr,
    REDUCE: tl.constexpr,
    INCLUDE_SELF: tl.constexpr,
    BLOCK: tl.constexpr,
):
    output_offset = tl.program_id(0)
    z = output_offset % input_size2
    y = (output_offset // input_size2) % input_size1
    x = output_offset // (input_size1 * input_size2)
    source_dim_offsets = tl.arange(0, BLOCK)

    if DIM == 0:
        valid_base = (y < index_size1) & (z < index_size2)
        dim_size = index_size0
        index_base = y * index_size2 + z
        src_base = y * src_size2 + z
        index_stride = index_size1 * index_size2
        src_stride = src_size1 * src_size2
        destination = x
    elif DIM == 1:
        valid_base = (x < index_size0) & (z < index_size2)
        dim_size = index_size1
        index_base = x * index_size1 * index_size2 + z
        src_base = x * src_size1 * src_size2 + z
        index_stride = index_size2
        src_stride = src_size2
        destination = y
    else:
        valid_base = (x < index_size0) & (y < index_size1)
        dim_size = index_size2
        index_base = x * index_size1 * index_size2 + y * index_size2
        src_base = x * src_size1 * src_size2 + y * src_size2
        index_stride = 1
        src_stride = 1
        destination = z

    valid = valid_base & (source_dim_offsets < dim_size)
    index_offsets = index_base + source_dim_offsets * index_stride
    src_offsets = src_base + source_dim_offsets * src_stride
    indices = tl.load(index + index_offsets, mask=valid, other=-1)
    selected = valid & (indices == destination)
    values = tl.load(src + src_offsets, mask=selected, other=0.0).to(tl.float32)
    self_value = tl.load(inp + output_offset).to(tl.float32)
    selected_count = tl.sum(selected.to(tl.int32), axis=0)

    if REDUCE == 0:
        reduced = tl.sum(tl.where(selected, values, 0.0), axis=0)
        if INCLUDE_SELF:
            reduced += self_value
    elif REDUCE == 1:
        reduced = 1.0
        for offset in tl.static_range(BLOCK):
            valid_offset = valid_base & (offset < dim_size)
            index_value = tl.load(
                index + index_base + offset * index_stride,
                mask=valid_offset,
                other=-1,
            )
            value = tl.load(
                src + src_base + offset * src_stride,
                mask=valid_offset,
                other=1.0,
            ).to(tl.float32)
            reduced *= tl.where(index_value == destination, value, 1.0)
        if INCLUDE_SELF:
            reduced *= self_value
    elif REDUCE == 2:
        reduced = tl.sum(tl.where(selected, values, 0.0), axis=0)
        count = selected_count
        if INCLUDE_SELF:
            reduced += self_value
            count += 1
        reduced /= count
    elif REDUCE == 3:
        reduced = tl.max(tl.where(selected, values, -float("inf")), axis=0)
        if INCLUDE_SELF:
            reduced = tl.maximum(reduced, self_value)
    else:
        reduced = tl.min(tl.where(selected, values, float("inf")), axis=0)
        if INCLUDE_SELF:
            reduced = tl.minimum(reduced, self_value)

    if not INCLUDE_SELF:
        reduced = tl.where(selected_count == 0, self_value, reduced)
    tl.store(out + output_offset, reduced)


@libentry()
@triton.jit(do_not_specialize=["input_dim_size", "source_dim_size", "inner_size"])
def _scatter_reduce_prod_kernel(
    inp,
    index,
    src,
    out,
    input_dim_size,
    source_dim_size,
    inner_size,
    INCLUDE_SELF: tl.constexpr,
):
    output_offset = tl.program_id(0)
    inner_offset = output_offset % inner_size
    output_dim_offset = (output_offset // inner_size) % input_dim_size
    outer_offset = output_offset // (input_dim_size * inner_size)
    self_value = tl.load(inp + output_offset).to(tl.float32)
    product = self_value if INCLUDE_SELF else 1.0
    selected_count = 0

    source_dim_offset = 0
    while source_dim_offset < source_dim_size:
        source_offset = (
            outer_offset * source_dim_size + source_dim_offset
        ) * inner_size + inner_offset
        selected = tl.load(index + source_offset) == output_dim_offset
        value = tl.load(src + source_offset).to(tl.float32)
        product = tl.where(selected, product * value, product)
        selected_count += selected.to(tl.int32)
        source_dim_offset += 1

    if not INCLUDE_SELF:
        product = tl.where(selected_count == 0, self_value, product)
    tl.store(out + output_offset, product)


@libentry()
@triton.jit
def _scatter_reduce_prod_3d_kernel(
    inp,
    index,
    src,
    out,
    input_size0,
    input_size1,
    input_size2,
    index_size0,
    index_size1,
    index_size2,
    src_size1,
    src_size2,
    DIM: tl.constexpr,
    INCLUDE_SELF: tl.constexpr,
):
    output_offset = tl.program_id(0)
    z = output_offset % input_size2
    y = (output_offset // input_size2) % input_size1
    x = output_offset // (input_size1 * input_size2)
    self_value = tl.load(inp + output_offset).to(tl.float32)

    if DIM == 0:
        valid_base = (y < index_size1) & (z < index_size2)
        dim_size = index_size0
        index_base = y * index_size2 + z
        src_base = y * src_size2 + z
        index_stride = index_size1 * index_size2
        src_stride = src_size1 * src_size2
        destination = x
    elif DIM == 1:
        valid_base = (x < index_size0) & (z < index_size2)
        dim_size = index_size1
        index_base = x * index_size1 * index_size2 + z
        src_base = x * src_size1 * src_size2 + z
        index_stride = index_size2
        src_stride = src_size2
        destination = y
    else:
        valid_base = (x < index_size0) & (y < index_size1)
        dim_size = index_size2
        index_base = x * index_size1 * index_size2 + y * index_size2
        src_base = x * src_size1 * src_size2 + y * src_size2
        index_stride = 1
        src_stride = 1
        destination = z

    product = self_value if INCLUDE_SELF else 1.0
    selected_count = 0
    neg = 0
    if INCLUDE_SELF:
        neg ^= (self_value.to(tl.int32, bitcast=True) >> 31) & 1
    offset = 0
    if valid_base:
        while offset < dim_size:
            index_value = tl.load(index + index_base + offset * index_stride)
            value = tl.load(src + src_base + offset * src_stride).to(tl.float32)
            matched = index_value == destination
            product = tl.where(matched, product * value, product)
            vneg = (value.to(tl.int32, bitcast=True) >> 31) & 1
            neg = tl.where(matched, neg ^ vneg, neg)
            selected_count += matched.to(tl.int32)
            offset += 1
    if not INCLUDE_SELF:
        product = tl.where(selected_count == 0, self_value, product)
        neg = tl.where(
            selected_count == 0,
            (self_value.to(tl.int32, bitcast=True) >> 31) & 1,
            neg,
        )
    zbits = tl.where(neg == 1, 0x80000000, 0).to(tl.uint32).to(tl.int32)
    negzero = zbits.to(tl.float32, bitcast=True)
    product = tl.where(product == 0.0, tl.where(neg == 1, negzero, 0.0), product)
    tl.store(out + output_offset, product)


@libentry()
@triton.jit
def _scatter_reduce_2d_kernel(
    inp,
    index,
    src,
    out,
    input_size0,
    input_size1,
    index_size0,
    index_size1,
    src_size1,
    DIM: tl.constexpr,
    REDUCE: tl.constexpr,
    INCLUDE_SELF: tl.constexpr,
    BLOCK: tl.constexpr,
):
    output_offset = tl.program_id(0)
    y = output_offset % input_size1
    x = output_offset // input_size1
    offsets = tl.arange(0, BLOCK)
    if DIM == 0:
        valid_base = y < index_size1
        dim_size, index_base, src_base = index_size0, y, y
        index_stride, src_stride, destination = index_size1, src_size1, x
    else:
        valid_base = x < index_size0
        dim_size, index_base, src_base = index_size1, x * index_size1, x * src_size1
        index_stride, src_stride, destination = 1, 1, y

    valid = valid_base & (offsets < dim_size)
    indices = tl.load(index + index_base + offsets * index_stride, mask=valid, other=-1)
    selected = valid & (indices == destination)
    values = tl.load(
        src + src_base + offsets * src_stride, mask=selected, other=0.0
    ).to(tl.float32)
    self_value = tl.load(inp + output_offset).to(tl.float32)
    selected_count = tl.sum(selected.to(tl.int32), axis=0)
    if REDUCE == 0:
        reduced = tl.sum(tl.where(selected, values, 0.0), axis=0)
        if INCLUDE_SELF:
            reduced += self_value
    elif REDUCE == 1:
        reduced = 1.0
        for offset in tl.static_range(BLOCK):
            valid_offset = valid_base & (offset < dim_size)
            index_value = tl.load(
                index + index_base + offset * index_stride, mask=valid_offset, other=-1
            )
            value = tl.load(
                src + src_base + offset * src_stride, mask=valid_offset, other=1.0
            ).to(tl.float32)
            reduced *= tl.where(valid_offset & (index_value == destination), value, 1.0)
        if INCLUDE_SELF:
            reduced *= self_value
    elif REDUCE == 2:
        reduced = tl.sum(tl.where(selected, values, 0.0), axis=0)
        count = selected_count
        if INCLUDE_SELF:
            reduced += self_value
            count += 1.0
        reduced /= count
    elif REDUCE == 3:
        reduced = tl.max(tl.where(selected, values, float("-inf")), axis=0)
        if INCLUDE_SELF:
            reduced = tl.maximum(reduced, self_value)
    else:
        reduced = tl.min(tl.where(selected, values, float("inf")), axis=0)
        if INCLUDE_SELF:
            reduced = tl.minimum(reduced, self_value)
    if not INCLUDE_SELF:
        reduced = tl.where(selected_count == 0, self_value, reduced)
    tl.store(out + output_offset, reduced)


@libentry()
@triton.jit
def _scatter_reduce_general_kernel(
    inp,
    index,
    src,
    out,
    s0,
    s1,
    s2,
    s3,
    s4,
    i0,
    i1,
    i2,
    i3,
    i4,
    r0,
    r1,
    r2,
    r3,
    r4,
    DIM: tl.constexpr,
    REDUCE: tl.constexpr,
    INCLUDE_SELF: tl.constexpr,
    BLOCK: tl.constexpr,
):
    output_offset = tl.program_id(0).to(tl.int64)
    rem = output_offset
    c0 = rem // (s1 * s2 * s3 * s4)
    rem = rem % (s1 * s2 * s3 * s4)
    c1 = rem // (s2 * s3 * s4)
    rem = rem % (s2 * s3 * s4)
    c2 = rem // (s3 * s4)
    rem = rem % (s3 * s4)
    c3 = rem // s4
    c4 = rem % s4

    istr0 = i1 * i2 * i3 * i4
    istr1 = i2 * i3 * i4
    istr2 = i3 * i4
    istr3 = i4
    istr4 = 1
    rstr0 = r1 * r2 * r3 * r4
    rstr1 = r2 * r3 * r4
    rstr2 = r3 * r4
    rstr3 = r4
    rstr4 = 1

    if DIM == 0:
        valid_fp = (c1 < i1) & (c2 < i2) & (c3 < i3) & (c4 < i4)
        index_base = c1 * istr1 + c2 * istr2 + c3 * istr3 + c4 * istr4
        src_base = c1 * rstr1 + c2 * rstr2 + c3 * rstr3 + c4 * rstr4
        index_stride = istr0
        src_stride = rstr0
        dim_size = i0
        destination = c0
    elif DIM == 1:
        valid_fp = (c0 < i0) & (c2 < i2) & (c3 < i3) & (c4 < i4)
        index_base = c0 * istr0 + c2 * istr2 + c3 * istr3 + c4 * istr4
        src_base = c0 * rstr0 + c2 * rstr2 + c3 * rstr3 + c4 * rstr4
        index_stride = istr1
        src_stride = rstr1
        dim_size = i1
        destination = c1
    elif DIM == 2:
        valid_fp = (c0 < i0) & (c1 < i1) & (c3 < i3) & (c4 < i4)
        index_base = c0 * istr0 + c1 * istr1 + c3 * istr3 + c4 * istr4
        src_base = c0 * rstr0 + c1 * rstr1 + c3 * rstr3 + c4 * rstr4
        index_stride = istr2
        src_stride = rstr2
        dim_size = i2
        destination = c2
    elif DIM == 3:
        valid_fp = (c0 < i0) & (c1 < i1) & (c2 < i2) & (c4 < i4)
        index_base = c0 * istr0 + c1 * istr1 + c2 * istr2 + c4 * istr4
        src_base = c0 * rstr0 + c1 * rstr1 + c2 * rstr2 + c4 * rstr4
        index_stride = istr3
        src_stride = rstr3
        dim_size = i3
        destination = c3
    else:
        valid_fp = (c0 < i0) & (c1 < i1) & (c2 < i2) & (c3 < i3)
        index_base = c0 * istr0 + c1 * istr1 + c2 * istr2 + c3 * istr3
        src_base = c0 * rstr0 + c1 * rstr1 + c2 * rstr2 + c3 * rstr3
        index_stride = istr4
        src_stride = rstr4
        dim_size = i4
        destination = c4

    offsets = tl.arange(0, BLOCK)
    valid = valid_fp & (offsets < dim_size)
    indices = tl.load(index + index_base + offsets * index_stride, mask=valid, other=-1)
    selected = valid & (indices == destination)
    values = tl.load(
        src + src_base + offsets * src_stride, mask=selected, other=0.0
    ).to(tl.float32)
    self_value = tl.load(inp + output_offset).to(tl.float32)
    selected_count = tl.sum(selected.to(tl.int32), axis=0)
    if REDUCE == 0:
        reduced = tl.sum(tl.where(selected, values, 0.0), axis=0)
        if INCLUDE_SELF:
            reduced += self_value
    elif REDUCE == 1:
        reduced = 1.0
        offset = 0
        while offset < dim_size:
            valid_offset = valid_fp & (offset < dim_size)
            index_value = tl.load(
                index + index_base + offset * index_stride,
                mask=valid_offset,
                other=-1,
            )
            value = tl.load(
                src + src_base + offset * src_stride,
                mask=valid_offset,
                other=1.0,
            ).to(tl.float32)
            reduced *= tl.where(valid_offset & (index_value == destination), value, 1.0)
            offset += 1
        if INCLUDE_SELF:
            reduced *= self_value
    elif REDUCE == 2:
        reduced = tl.sum(tl.where(selected, values, 0.0), axis=0)
        count = selected_count
        if INCLUDE_SELF:
            reduced += self_value
            count += 1
        reduced /= count
    elif REDUCE == 3:
        reduced = tl.max(tl.where(selected, values, float("-inf")), axis=0)
        if INCLUDE_SELF:
            reduced = tl.maximum(reduced, self_value)
    else:
        reduced = tl.min(tl.where(selected, values, float("inf")), axis=0)
        if INCLUDE_SELF:
            reduced = tl.minimum(reduced, self_value)
    if not INCLUDE_SELF:
        reduced = tl.where(selected_count == 0, self_value, reduced)
    tl.store(out + output_offset, reduced)


@libentry()
@triton.jit
def _scatter_reduce_prod_general_kernel(
    inp,
    index,
    src,
    out,
    s0,
    s1,
    s2,
    s3,
    s4,
    i0,
    i1,
    i2,
    i3,
    i4,
    r0,
    r1,
    r2,
    r3,
    r4,
    DIM: tl.constexpr,
    INCLUDE_SELF: tl.constexpr,
):
    output_offset = tl.program_id(0).to(tl.int64)
    rem = output_offset
    c0 = rem // (s1 * s2 * s3 * s4)
    rem = rem % (s1 * s2 * s3 * s4)
    c1 = rem // (s2 * s3 * s4)
    rem = rem % (s2 * s3 * s4)
    c2 = rem // (s3 * s4)
    rem = rem % (s3 * s4)
    c3 = rem // s4
    c4 = rem % s4

    istr0 = i1 * i2 * i3 * i4
    istr1 = i2 * i3 * i4
    istr2 = i3 * i4
    istr3 = i4
    istr4 = 1
    rstr0 = r1 * r2 * r3 * r4
    rstr1 = r2 * r3 * r4
    rstr2 = r3 * r4
    rstr3 = r4
    rstr4 = 1

    if DIM == 0:
        valid_fp = (c1 < i1) & (c2 < i2) & (c3 < i3) & (c4 < i4)
        index_base = c1 * istr1 + c2 * istr2 + c3 * istr3 + c4 * istr4
        src_base = c1 * rstr1 + c2 * rstr2 + c3 * rstr3 + c4 * rstr4
        index_stride = istr0
        src_stride = rstr0
        dim_size = i0
        destination = c0
    elif DIM == 1:
        valid_fp = (c0 < i0) & (c2 < i2) & (c3 < i3) & (c4 < i4)
        index_base = c0 * istr0 + c2 * istr2 + c3 * istr3 + c4 * istr4
        src_base = c0 * rstr0 + c2 * rstr2 + c3 * rstr3 + c4 * rstr4
        index_stride = istr1
        src_stride = rstr1
        dim_size = i1
        destination = c1
    elif DIM == 2:
        valid_fp = (c0 < i0) & (c1 < i1) & (c3 < i3) & (c4 < i4)
        index_base = c0 * istr0 + c1 * istr1 + c3 * istr3 + c4 * istr4
        src_base = c0 * rstr0 + c1 * rstr1 + c3 * rstr3 + c4 * rstr4
        index_stride = istr2
        src_stride = rstr2
        dim_size = i2
        destination = c2
    elif DIM == 3:
        valid_fp = (c0 < i0) & (c1 < i1) & (c2 < i2) & (c4 < i4)
        index_base = c0 * istr0 + c1 * istr1 + c2 * istr2 + c4 * istr4
        src_base = c0 * rstr0 + c1 * rstr1 + c2 * rstr2 + c4 * rstr4
        index_stride = istr3
        src_stride = rstr3
        dim_size = i3
        destination = c3
    else:
        valid_fp = (c0 < i0) & (c1 < i1) & (c2 < i2) & (c3 < i3)
        index_base = c0 * istr0 + c1 * istr1 + c2 * istr2 + c3 * istr3
        src_base = c0 * rstr0 + c1 * rstr1 + c2 * rstr2 + c3 * rstr3
        index_stride = istr4
        src_stride = rstr4
        dim_size = i4
        destination = c4

    self_value = tl.load(inp + output_offset).to(tl.float32)
    product = self_value if INCLUDE_SELF else 1.0
    selected_count = 0
    offset = 0
    while offset < dim_size:
        index_value = tl.load(
            index + index_base + offset * index_stride, mask=valid_fp, other=-1
        )
        value = tl.load(
            src + src_base + offset * src_stride, mask=valid_fp, other=1.0
        ).to(tl.float32)
        matched = valid_fp & (index_value == destination)
        product = tl.where(matched, product * value, product)
        selected_count += matched.to(tl.int32)
        offset += 1
    if not INCLUDE_SELF:
        product = tl.where(selected_count == 0, self_value, product)
    tl.store(out + output_offset, product)


@triton.jit
def _sr_mul(a, b):
    return a * b


@libentry()
@triton.jit
def _scatter_reduce_chunked_kernel(
    inp,
    index,
    src,
    out,
    input_size0,
    input_size1,
    input_size2,
    index_size0,
    index_size1,
    index_size2,
    src_size1,
    src_size2,
    REDUCE: tl.constexpr,
    INCLUDE_SELF: tl.constexpr,
    BLOCK: tl.constexpr,
):
    output_offset = tl.program_id(0)
    z = output_offset % input_size2
    y = (output_offset // input_size2) % input_size1
    x = output_offset // (input_size1 * input_size2)
    valid_base = (x < index_size0) & (z < index_size2)
    dim_size = index_size1
    index_base = x * index_size1 * index_size2 + z
    src_base = x * src_size1 * src_size2 + z
    index_stride = index_size2
    src_stride = src_size2
    destination = y

    self_value = tl.load(inp + output_offset).to(tl.float32)
    if REDUCE == 0 or REDUCE == 2:
        acc = 0.0
    elif REDUCE == 3:
        acc = -float("inf")
    else:
        acc = float("inf")
    cnt = 0
    for start in tl.range(0, dim_size, BLOCK):
        off = start + tl.arange(0, BLOCK)
        valid = valid_base & (off < dim_size)
        idx = tl.load(index + index_base + off * index_stride, mask=valid, other=-1)
        sel = valid & (idx == destination)
        val = tl.load(
            src + src_base + off * src_stride, mask=sel, other=0.0
        ).to(tl.float32)
        cnt += tl.sum(sel.to(tl.int32), axis=0)
        if REDUCE == 0 or REDUCE == 2:
            acc += tl.sum(tl.where(sel, val, 0.0), axis=0)
        elif REDUCE == 3:
            acc = tl.maximum(acc, tl.max(tl.where(sel, val, -float("inf")), axis=0))
        else:
            acc = tl.minimum(acc, tl.min(tl.where(sel, val, float("inf")), axis=0))

    if REDUCE == 2:
        count = cnt + (1 if INCLUDE_SELF else 0)
        if INCLUDE_SELF:
            acc += self_value
        acc = acc / count
    elif INCLUDE_SELF:
        if REDUCE == 0:
            acc += self_value
        elif REDUCE == 3:
            acc = tl.maximum(acc, self_value)
        else:
            acc = tl.minimum(acc, self_value)
    if not INCLUDE_SELF:
        acc = tl.where(cnt == 0, self_value, acc)
    tl.store(out + output_offset, acc)


@libentry()
@triton.jit
def _scatter_reduce_prod_chunked_kernel(
    inp,
    index,
    src,
    out,
    input_size0,
    input_size1,
    input_size2,
    index_size0,
    index_size1,
    index_size2,
    src_size1,
    src_size2,
    INCLUDE_SELF: tl.constexpr,
    BLOCK: tl.constexpr,
):
    output_offset = tl.program_id(0)
    z = output_offset % input_size2
    y = (output_offset // input_size2) % input_size1
    x = output_offset // (input_size1 * input_size2)
    valid_base = (x < index_size0) & (z < index_size2)
    dim_size = index_size1
    index_base = x * index_size1 * index_size2 + z
    src_base = x * src_size1 * src_size2 + z
    index_stride = index_size2
    src_stride = src_size2
    destination = y

    self_value = tl.load(inp + output_offset).to(tl.float32)
    product = self_value if INCLUDE_SELF else 1.0
    neg = 0
    if INCLUDE_SELF:
        neg ^= (self_value.to(tl.int32, bitcast=True) >> 31) & 1
    cnt = 0
    for start in tl.range(0, dim_size, BLOCK):
        off = start + tl.arange(0, BLOCK)
        valid = valid_base & (off < dim_size)
        idx = tl.load(index + index_base + off * index_stride, mask=valid, other=-1)
        sel = valid & (idx == destination)
        val = tl.load(
            src + src_base + off * src_stride, mask=sel, other=1.0
        ).to(tl.float32)
        product *= tl.reduce(tl.where(sel, val, 1.0), 0, _sr_mul)
        vneg = (val.to(tl.int32, bitcast=True) >> 31) & 1
        neg ^= tl.sum(tl.where(sel, vneg, 0), axis=0) & 1
        cnt += tl.sum(sel.to(tl.int32), axis=0)

    if not INCLUDE_SELF:
        product = tl.where(cnt == 0, self_value, product)
        neg = tl.where(
            cnt == 0, (self_value.to(tl.int32, bitcast=True) >> 31) & 1, neg
        )
    zbits = tl.where(neg == 1, 0x80000000, 0).to(tl.uint32).to(tl.int32)
    negzero = zbits.to(tl.float32, bitcast=True)
    product = tl.where(product == 0.0, tl.where(neg == 1, negzero, 0.0), product)
    tl.store(out + output_offset, product)


_REDUCTIONS = {"sum": 0, "prod": 1, "mean": 2, "amax": 3, "amin": 4}
_SINGLE_TILE_LIMIT = 65536
_CHUNK_BLOCK = 4096


def _pad5(shape, fill=1):
    """Rank-pad a shape tuple to 5 entries (scatter_reduce supports <= 5D)."""
    return tuple(shape) + (fill,) * (5 - len(shape))


def _scatter_reduce_high_rank(inp, dim, index, src, reduce, include_self):
    """Collapse an arbitrary-rank scatter into an equivalent 3D problem.

    ``index`` defines the active prefix domain. Only that domain is read from
    ``src`` and updated in ``inp``; everything outside it is copied through
    unchanged. After narrowing the active prefixes every valid scatter maps to
    ``(outer, scatter_dim, inner)`` and is handled by the existing aligned 3D
    kernels (no atomics, deterministic).
    """
    index_shape = tuple(int(size) for size in index.shape)
    active_inp = inp
    active_src = src
    for axis, index_size in enumerate(index_shape):
        active_src = active_src.narrow(axis, 0, index_size)
        if axis != dim:
            active_inp = active_inp.narrow(axis, 0, index_size)

    outer = 1
    for s in index_shape[:dim]:
        outer *= s
    inner = 1
    for s in index_shape[dim + 1 :]:
        inner *= s
    active_shape = tuple(active_inp.shape)
    inp_3d = active_inp.contiguous().reshape(outer, inp.size(dim), inner)
    index_3d = index.contiguous().reshape(outer, index_shape[dim], inner)
    src_3d = active_src.contiguous().reshape(outer, index_shape[dim], inner)

    active_result = scatter_reduce(
        inp_3d, 1, index_3d, src_3d, reduce, include_self=include_self
    )

    active_domain_is_full = all(
        axis == dim or index_size == inp.size(axis)
        for axis, index_size in enumerate(index_shape)
    )
    if active_domain_is_full:
        return active_result.reshape(active_shape)

    result = inp.contiguous().clone()
    result_active = result
    for axis, index_size in enumerate(index_shape):
        if axis != dim:
            result_active = result_active.narrow(axis, 0, index_size)
    result_active.copy_(active_result.reshape(active_shape))
    return result


def scatter_reduce(inp, dim, index, src, reduce, *, include_self=True):
    logger.debug("GEMS_KUNLUNXIN SCATTER_REDUCE")
    if reduce not in _REDUCTIONS:
        raise RuntimeError(
            f"reduce argument must be either sum, prod, mean, amax or amin, got {reduce}"
        )
    if inp.ndim == 0:
        raise RuntimeError(
            "scatter_reduce(): Expected self to have non-zero dimensionality"
        )
    if dim < -inp.ndim or dim >= inp.ndim:
        raise IndexError(
            "Dimension out of range (expected to be in range of "
            f"[{-inp.ndim}, {inp.ndim - 1}], but got {dim})"
        )

    dim %= inp.ndim
    if index.ndim != inp.ndim or src.ndim != inp.ndim:
        raise RuntimeError(
            "index and src must have the same number of dimensions as self"
        )
    for axis, size in enumerate(index.shape):
        if size > src.shape[axis] or (axis != dim and size > inp.shape[axis]):
            raise RuntimeError(
                "index must not be larger than src or self outside the scatter dimension"
            )

    if inp.ndim > 5:
        return _scatter_reduce_high_rank(inp, dim, index, src, reduce, include_self)

    result = inp.contiguous().clone()
    if index.numel() == 0 or result.numel() == 0:
        return result

    index = index.contiguous()
    src = src.contiguous()
    idx_dim_size = index.shape[dim]
    use_chunked = idx_dim_size > _SINGLE_TILE_LIMIT
    block = triton.next_power_of_2(idx_dim_size) if not use_chunked else _CHUNK_BLOCK

    outer_d = 1
    for s in index.shape[:dim]:
        outer_d *= s
    idx_dim = index.shape[dim]
    inner_d = 1
    for s in index.shape[dim + 1 :]:
        inner_d *= s
    outer_i = 1
    for s in inp.shape[:dim]:
        outer_i *= s
    in_dim = inp.shape[dim]
    inner_i = 1
    for s in inp.shape[dim + 1 :]:
        inner_i *= s
    src_dim = src.shape[dim]
    src_inner = 1
    for s in src.shape[dim + 1 :]:
        src_inner *= s

    if inp.ndim == 2 and reduce != "prod" and not use_chunked:
        with torch_device_fn.device(inp.device):
            _scatter_reduce_2d_kernel[(result.numel(),)](
                inp.contiguous(),
                index,
                src,
                result,
                *inp.shape,
                *index.shape,
                src.shape[1],
                DIM=dim,
                REDUCE=_REDUCTIONS[reduce],
                INCLUDE_SELF=include_self,
                BLOCK=block,
            )
        return result

    aligned = all(
        index.shape[d] == src.shape[d] == inp.shape[d]
        for d in range(inp.ndim)
        if d != dim
    )

    input_contiguous = inp.contiguous()
    with torch_device_fn.device(inp.device):
        if use_chunked:
            if not aligned:
                raise RuntimeError(
                    "Kunlunxin scatter_reduce chunked path requires aligned "
                    "non-scatter dimensions"
                )
            if reduce == "prod":
                _scatter_reduce_prod_chunked_kernel[(result.numel(),)](
                    input_contiguous,
                    index,
                    src,
                    result,
                    outer_i,
                    in_dim,
                    inner_i,
                    outer_d,
                    idx_dim,
                    inner_d,
                    src_dim,
                    src_inner,
                    INCLUDE_SELF=include_self,
                    BLOCK=_CHUNK_BLOCK,
                    isCloseVectorization=True,
                    buffer_size_limit=2048,
                )
            else:
                _scatter_reduce_chunked_kernel[(result.numel(),)](
                    input_contiguous,
                    index,
                    src,
                    result,
                    outer_i,
                    in_dim,
                    inner_i,
                    outer_d,
                    idx_dim,
                    inner_d,
                    src_dim,
                    src_inner,
                    REDUCE=_REDUCTIONS[reduce],
                    INCLUDE_SELF=include_self,
                    BLOCK=_CHUNK_BLOCK,
                    isCloseVectorization=True,
                    buffer_size_limit=2048,
                )
        elif not aligned:
            sp = _pad5(inp.shape, 1)
            ip = _pad5(index.shape, 1)
            rp = _pad5(src.shape, 1)
            if reduce == "prod":
                _scatter_reduce_prod_general_kernel[(result.numel(),)](
                    input_contiguous,
                    index,
                    src,
                    result,
                    *sp,
                    *ip,
                    *rp,
                    DIM=dim,
                    INCLUDE_SELF=include_self,
                    isCloseVectorization=True,
                    buffer_size_limit=2048,
                )
            else:
                _scatter_reduce_general_kernel[(result.numel(),)](
                    input_contiguous,
                    index,
                    src,
                    result,
                    *sp,
                    *ip,
                    *rp,
                    DIM=dim,
                    REDUCE=_REDUCTIONS[reduce],
                    INCLUDE_SELF=include_self,
                    BLOCK=block,
                    isCloseVectorization=True,
                    buffer_size_limit=2048,
                )
        elif reduce == "prod":
            _scatter_reduce_prod_3d_kernel[(result.numel(),)](
                input_contiguous,
                index,
                src,
                result,
                outer_i,
                in_dim,
                inner_i,
                outer_d,
                idx_dim,
                inner_d,
                src_dim,
                src_inner,
                DIM=1,
                INCLUDE_SELF=include_self,
                isCloseVectorization=True,
                buffer_size_limit=2048,
            )
        else:
            _scatter_reduce_kernel[(result.numel(),)](
                input_contiguous,
                index,
                src,
                result,
                outer_i,
                in_dim,
                inner_i,
                outer_d,
                idx_dim,
                inner_d,
                src_dim,
                src_inner,
                DIM=1,
                REDUCE=_REDUCTIONS[reduce],
                INCLUDE_SELF=include_self,
                BLOCK=block,
                isCloseVectorization=True,
                buffer_size_limit=2048,
            )
    return result


def scatter_reduce_(inp, dim, index, src, reduce, *, include_self=True):
    logger.debug("GEMS_KUNLUNXIN SCATTER_REDUCE_TWO_")
    result = scatter_reduce(inp, dim, index, src, reduce, include_self=include_self)
    inp.copy_(result)
    return inp


def scatter_reduce_out(inp, dim, index, src, reduce, *, include_self=True, out=None):
    logger.debug("GEMS_KUNLUNXIN SCATTER_REDUCE_TWO_OUT")
    if out is not None and out.dtype != inp.dtype:
        raise RuntimeError(
            f"Expected out tensor to have dtype {inp.dtype}, but got {out.dtype} instead"
        )
    result = scatter_reduce(inp, dim, index, src, reduce, include_self=include_self)
    if tuple(out.shape) != tuple(result.shape):
        out.resize_(result.shape)
    out.copy_(result)
    return out
