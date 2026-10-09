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

from flag_gems.ops.contiguous import contiguous
from flag_gems.ops.copy import copy_
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

from .sort import sort_stable

logger = logging.getLogger(__name__)


@libentry()
@triton.jit
def canonicalize_nan_kernel(inp, out, n_elements, BLOCK_SIZE: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    values = tl.load(inp + offsets, mask=mask)
    values = tl.where(values != values, float("nan"), values)
    tl.store(out + offsets, values, mask=mask)


@libentry()
@triton.jit
def gather_rowwise_kernel(src, indices, out, n_elements, N, BLOCK_SIZE: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    row_base = (offsets // N) * N
    src_idx = tl.load(indices + offsets, mask=mask, other=0)
    values = tl.load(src + row_base + src_idx, mask=mask)
    tl.store(out + offsets, values, mask=mask)


def _msort_contiguous(inp, out):
    if inp.ndim == 0:
        copy_(out, inp)
        return

    n_rows = inp.shape[0]
    if inp.numel() == 0 or n_rows == 1:
        copy_(out, inp)
        return

    n_cols = inp.numel() // n_rows
    # msort sorts along dim 0.  The tuned kunlunxin radix sort is only correct
    # on its contiguous last-dim path, so transpose the sort axis to the end
    # ourselves instead of relying on sort_stable's strided dim!=-1 permute.
    src2d = inp.reshape(n_rows, n_cols)
    srcT = contiguous(src2d.transpose(0, 1))  # (n_cols, n_rows), sort axis last

    if inp.dtype.is_floating_point:
        # Map every NaN (including negative-signed ones) to a canonical
        # positive NaN so they all sort last like CPU ATen; the sort only
        # drives the ordering while the original payloads are gathered back.
        keyT = torch.empty_like(srcT)
        block_size = 1024
        with torch_device_fn.device(inp.device):
            canonicalize_nan_kernel[(triton.cdiv(keyT.numel(), block_size),)](
                srcT, keyT, keyT.numel(), BLOCK_SIZE=block_size
            )
        _, indices = sort_stable(keyT, stable=True, dim=-1, descending=False)
        indices = contiguous(indices)
        outT = torch.empty_like(srcT)
        with torch_device_fn.device(inp.device):
            gather_rowwise_kernel[(triton.cdiv(srcT.numel(), 1024),)](
                srcT, indices, outT, srcT.numel(), n_rows, BLOCK_SIZE=1024
            )
    else:
        outT, _ = sort_stable(srcT, stable=True, dim=-1, descending=False)

    out2d = contiguous(outT.transpose(0, 1)).reshape(inp.shape)
    copy_(out, out2d)


def msort(inp):
    logger.debug("GEMS_KUNLUNXIN MSORT")
    if inp.is_complex():
        raise RuntimeError('"msort" not implemented for complex dtypes')
    out = torch.empty_like(inp, memory_format=torch.preserve_format)
    msort_out(inp, out=out)
    return out


def msort_out(inp, *, out):
    logger.debug("GEMS_KUNLUNXIN MSORT.OUT")
    if inp.is_complex():
        raise RuntimeError('"msort" not implemented for complex dtypes')
    if out.dtype != inp.dtype:
        raise RuntimeError(
            f"Expected out tensor to have dtype {inp.dtype}, but got {out.dtype} instead"
        )
    if out.device != inp.device:
        raise RuntimeError(
            f"Expected out tensor to have device {inp.device}, but got {out.device} instead"
        )
    if (
        inp.numel()
        and inp.untyped_storage().data_ptr() == out.untyped_storage().data_ptr()
    ):
        source = torch.empty(inp.shape, dtype=inp.dtype, device=inp.device)
        copy_(source, inp)
        inp = source
    if out.shape != inp.shape:
        out.resize_(inp.shape)

    if inp.is_contiguous() and out.is_contiguous():
        _msort_contiguous(inp, out)
    else:
        contiguous_inp = contiguous(inp)
        contiguous_out = torch.empty_like(contiguous_inp)
        _msort_contiguous(contiguous_inp, contiguous_out)
        copy_(out, contiguous_out)
    return out
