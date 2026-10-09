# Copyright 2026, The FlagOS Contributors.
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

from flag_gems.ops.put import (
    _NARROW_ACC_DTYPES,
    _check_same_device,
    _flat_copy,
    _scalar_type_name,
    _type_name,
    _validate_put_args,
    put_narrow_cast_kernel,
)
from flag_gems.utils import libentry
from flag_gems.utils.shape_utils import MemOverlap, has_internal_overlapping

logger = logging.getLogger(__name__)

@libentry()
@triton.jit(do_not_specialize=["N", "out_numel"])
def put_scatter_kernel(
    data_ptr,
    index_ptr,
    source_ptr,
    out_numel,
    N,
    ELEMS_PER_SLOT: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Scatter ``source`` into the contiguous ``data`` buffer at the flat indices.

    ``data`` is a row-major staging buffer with one extra scratch slot at
    ``out_numel`` (``ELEMS_PER_SLOT`` scalars wide). Masked lanes (``offsets >= N``)
    and any out-of-range index are redirected to that scratch slot, so the store
    needs no mask: every real element lands in ``[0, out_numel)`` and the throwaway
    writes pile up on the scratch slot. This sidesteps two XPU behaviours that the
    generic CUDA kernel relied on but that do not hold here: masked scatter stores
    are not honoured for discrete offsets, and ``other=`` is unreliable for masked
    loads. This is the non-accumulate path only; ``accumulate`` uses the serial
    kernel below because cross-program atomics drop colliding updates on xpu3.
    """
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N

    raw_index = tl.load(index_ptr + offsets, mask=mask, other=0).to(tl.int64)
    cur_index = tl.where(raw_index < 0, raw_index + out_numel, raw_index)
    valid = mask & (cur_index >= 0) & (cur_index < out_numel)
    dst = tl.where(valid, cur_index, out_numel)

    if ELEMS_PER_SLOT == 1:
        value = tl.load(source_ptr + offsets, mask=valid, other=0)
        tl.store(data_ptr + dst, value)
    else:
        real = tl.load(source_ptr + 2 * offsets, mask=valid, other=0)
        imag = tl.load(source_ptr + 2 * offsets + 1, mask=valid, other=0)
        real_off = 2 * dst
        tl.store(data_ptr + real_off, real)
        tl.store(data_ptr + real_off + 1, imag)


@libentry()
@triton.jit(do_not_specialize=["N", "out_numel", "N_CHUNKS"])
def put_accumulate_kernel(
    data_ptr,
    index_ptr,
    source_ptr,
    out_numel,
    N,
    N_CHUNKS,
    ELEMS_PER_SLOT: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Accumulate ``source`` into ``data`` at the flat indices, launched with a
    single program.

    On xpu3 ``tl.atomic_add`` only serialises colliding updates *within* one
    program instance; two different programs that hit the same address race and
    silently drop all but one of the adds (measured: 100k lanes adding 1.0 to one
    slot land ~2k-9k depending on dtype). Duplicate indices are the whole point of
    ``accumulate`` so that cross-program race corrupts the result. Running the
    scatter inside a single program (``grid=(1,)``) that loops over the input in
    ``BLOCK_SIZE`` chunks keeps every atomic add within that one program, where it
    is honoured, so colliding adds accumulate correctly. Invalid / masked lanes are
    redirected to the ``out_numel`` scratch slot exactly as in the store path.
    """
    base = tl.arange(0, BLOCK_SIZE)
    for i in range(N_CHUNKS):
        offsets = i * BLOCK_SIZE + base
        mask = offsets < N
        raw_index = tl.load(index_ptr + offsets, mask=mask, other=0).to(tl.int64)
        cur_index = tl.where(raw_index < 0, raw_index + out_numel, raw_index)
        valid = mask & (cur_index >= 0) & (cur_index < out_numel)
        dst = tl.where(valid, cur_index, out_numel)
        if ELEMS_PER_SLOT == 1:
            value = tl.load(source_ptr + offsets, mask=valid, other=0)
            tl.atomic_add(data_ptr + dst, value)
        else:
            real = tl.load(source_ptr + 2 * offsets, mask=valid, other=0)
            imag = tl.load(source_ptr + 2 * offsets + 1, mask=valid, other=0)
            real_off = 2 * dst
            tl.atomic_add(data_ptr + real_off, real)
            tl.atomic_add(data_ptr + real_off + 1, imag)


def _put_block_config(N):
    if N <= 1024:
        return 1024, 1
    elif N <= 8192:
        return 1024, 4
    else:
        return 2048, 8


def _launch_scatter(data, index, source, out_numel, elems_per_slot):
    """Launch the parallel store scatter into the padded staging buffer."""
    N = index.numel()
    BLOCK_SIZE, num_warps = _put_block_config(N)
    grid = (triton.cdiv(N, BLOCK_SIZE),)
    put_scatter_kernel[grid](
        data,
        index,
        source,
        out_numel,
        N,
        ELEMS_PER_SLOT=elems_per_slot,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps,
    )


def _launch_accumulate(data, index, source, out_numel, elems_per_slot):
    """Launch the single-program accumulate into the padded staging buffer."""
    N = index.numel()
    BLOCK_SIZE = 1024
    n_chunks = triton.cdiv(N, BLOCK_SIZE)
    put_accumulate_kernel[(1,)](
        data,
        index,
        source,
        out_numel,
        N,
        n_chunks,
        ELEMS_PER_SLOT=elems_per_slot,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=1,
    )


def _put_scatter(out, index, source, accumulate):
    """Scatter ``source`` into ``out`` at the flat (row-major) positions in ``index``.

    The scatter always runs on a contiguous row-major staging buffer padded with
    one scratch slot; ``out``'s own strides are honoured only by the surrounding
    ``_flat_copy`` in/out, so a non-contiguous ``out`` stays correct without the
    kernel decoding its layout.
    """
    N = index.numel()
    if N == 0:
        return out
    n = out.numel()
    if n == 0:
        raise IndexError("put_(): Tried to put elements into an empty tensor")

    if out.dtype == torch.complex32:
        raise NotImplementedError(
            '"put_cuda" not implemented for ' f"'{_scalar_type_name(torch.complex32)}'"
        )

    if out.is_complex():
        staging = torch.empty(n + 1, dtype=out.dtype, device=out.device)
        _flat_copy(staging[:n].view(out.shape), out)
        staging_real = torch.view_as_real(staging)
        source_real = torch.view_as_real(source)
        # xpu3 miscompiles the two interleaved scatter writes (to 2*dst and
        # 2*dst+1) that a single complex kernel emits: the imaginary store lands
        # on the wrong lane and the result is non-deterministic across launches.
        # Scatter the real and imaginary planes independently through the proven
        # single-slot path, where every store addresses a unique slot.
        real_plane = staging_real[:, 0].contiguous()
        imag_plane = staging_real[:, 1].contiguous()
        src_real = source_real[:, 0].contiguous()
        src_imag = source_real[:, 1].contiguous()
        if accumulate:
            _launch_accumulate(real_plane, index, src_real, n, elems_per_slot=1)
            _launch_accumulate(imag_plane, index, src_imag, n, elems_per_slot=1)
        else:
            _launch_scatter(real_plane, index, src_real, n, elems_per_slot=1)
            _launch_scatter(imag_plane, index, src_imag, n, elems_per_slot=1)
        combined = torch.stack([real_plane[:n], imag_plane[:n]], dim=-1).contiguous()
        _flat_copy(out, torch.view_as_complex(combined).view(out.shape))
        return out

    if not accumulate:
        staging = torch.empty(n + 1, dtype=out.dtype, device=out.device)
        _flat_copy(staging[:n].view(out.shape), out)
        _launch_scatter(staging, index, source, n, elems_per_slot=1)
        _flat_copy(out, staging[:n].view(out.shape))
        return out

    # accumulate == True from here down.
    if out.dtype in _NARROW_ACC_DTYPES:
        staging = torch.empty(n + 1, dtype=torch.int32, device=out.device)
        _flat_copy(staging[:n].view(out.shape), out.to(torch.int32))
        _launch_accumulate(staging, index, source.to(torch.int32), n, elems_per_slot=1)
        narrowed = torch.empty(n, dtype=out.dtype, device=out.device)
        BLOCK_SIZE = 1024
        put_narrow_cast_kernel[(triton.cdiv(n, BLOCK_SIZE),)](
            narrowed,
            staging[:n],
            n,
            TO_BOOL=out.dtype == torch.bool,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=4,
        )
        _flat_copy(out, narrowed.view(out.shape))
        return out

    if out.dtype in (torch.float16, torch.bfloat16):
        # xpu3 half/bfloat16 atomics round every partial sum, which drifts past the
        # test tolerance on the longer duplicate runs. Accumulate in fp32 and narrow
        # back so only a single rounding step remains.
        staging = torch.empty(n + 1, dtype=torch.float32, device=out.device)
        _flat_copy(staging[:n].view(out.shape), out.to(torch.float32))
        _launch_accumulate(
            staging, index, source.to(torch.float32), n, elems_per_slot=1
        )
        narrowed = staging[:n].to(out.dtype)
        _flat_copy(out, narrowed.view(out.shape))
        return out

    staging = torch.empty(n + 1, dtype=out.dtype, device=out.device)
    _flat_copy(staging[:n].view(out.shape), out)
    _launch_accumulate(staging, index, source, n, elems_per_slot=1)
    _flat_copy(out, staging[:n].view(out.shape))
    return out


def put_impl(out, index, source, accumulate):
    _validate_put_args(out, index, source)
    return _put_scatter(
        out, index.contiguous().reshape(-1), source.contiguous().reshape(-1), accumulate
    )


def put(self, index, source, accumulate=False):
    logger.debug("GEMS_KUNLUNXIN PUT")
    _check_same_device(self, index, source)
    out = torch.empty_like(self)
    _flat_copy(out, self)
    return put_impl(out, index, source, accumulate)


def put_(self, index, source, accumulate=False):
    logger.debug("GEMS_KUNLUNXIN PUT_")
    _check_same_device(self, index, source)
    return put_impl(self, index, source, accumulate)


def put_out(self, index, source, accumulate=False, *, out=None):
    logger.debug("GEMS_KUNLUNXIN PUT_OUT")

    if out is None:
        return put(self, index, source, accumulate)

    _check_same_device(self, index, source, out)

    _validate_put_args(self, index, source)
    if out.dtype != self.dtype:
        raise RuntimeError(
            f"Expected out tensor to have dtype {_type_name(self.dtype)}, "
            f"but got {_type_name(out.dtype)} instead"
        )

    assert (
        has_internal_overlapping(out) != MemOverlap.Yes
    ), "Unsupported operation: trying to inplace write to an internally overlapping tensor."

    if out is not self:
        if out.shape != self.shape:
            out = out.resize_as_(self)
        _flat_copy(out, self)

    put_impl(out, index, source, accumulate)
    return out
