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

from .copy import copy_

logger = logging.getLogger("flag_gems").getChild(__name__.lstrip("."))

# Fixed bounded tile for the broadcast (stride-0) gather path. 4096 with 4
# warps is the same width as the proven strided kernels in as_strided_copy.py
# (larger tiles were observed to regress the strided access pattern on XPU).
_BCAST_BLOCK = 4096


@triton.jit
def _expand_bcast_kernel(
    src_ptr,
    dst_ptr,
    n_elements,
    s0,
    s1,
    s2,
    s3,
    s4,
    s5,
    st0,
    st1,
    st2,
    st3,
    st4,
    st5,
    NDIM: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Gather+store kernel for an expanded (broadcast) view into a contiguous
    output.  ``dst`` is contiguous, ``src`` is the expand view whose strides
    are 0 on the expanded (size-1) dims.  The flat output index is decomposed
    back into per-dim indices from the innermost dim outwards and multiplied by
    the (possibly 0) source strides; the store side stays contiguous so it
    keeps the block-DMA lowering.
    """
    pid = tl.program_id(0)
    out_offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = out_offs < n_elements
    offs = out_offs.to(tl.int64)
    src_off = tl.zeros([BLOCK], dtype=tl.int64)
    if NDIM >= 6:
        c = offs % s5
        offs = offs // s5
        src_off += c * st5
    if NDIM >= 5:
        c = offs % s4
        offs = offs // s4
        src_off += c * st4
    if NDIM >= 4:
        c = offs % s3
        offs = offs // s3
        src_off += c * st3
    if NDIM >= 3:
        c = offs % s2
        offs = offs // s2
        src_off += c * st2
    if NDIM >= 2:
        c = offs % s1
        offs = offs // s1
        src_off += c * st1
    if NDIM >= 1:
        c = offs % s0
        src_off += c * st0
    vals = tl.load(src_ptr + src_off, mask=mask, other=0.0)
    tl.store(dst_ptr + out_offs, vals, mask=mask)


def _launch_bcast(shape, strides, src, dst, n):
    ndim = len(shape)
    shapes = (tuple(shape) + (1,) * (6 - ndim))
    strided = (tuple(strides) + (0,) * (6 - ndim))
    grid = (triton.cdiv(n, _BCAST_BLOCK),)
    _expand_bcast_kernel[grid](
        src,
        dst,
        n,
        *shapes,
        *strided,
        NDIM=ndim,
        BLOCK=_BCAST_BLOCK,
        num_warps=4,
    )


def expand_copy(x: torch.Tensor, size) -> torch.Tensor:
    """Kunlunxin override for aten::expand_copy.

    The generic ``flag_gems.ops.expand_copy`` calls the triton ``copy_`` from
    ``flag_gems.ops.copy``, whose pointwise kernel (default KUNLUNXIN config:
    no ``buffer_size_limit``/``kunlunAutoGrid``/``unroll``, per-lane
    ``offset//stride%size`` index math) defeats the XPU OffsetAnalysis pass and
    measures ~39ms on a 16M-element same-shape copy (0.00093x torch).  This
    override routes contiguous sources through the Kunlunxin ``copy_``
    (bounded-tile flat block-DMA kernel) and broadcast (stride-0) sources
    through a fixed-tile leaf gather kernel, avoiding both the generic slow
    path and the per-launch Python overhead of the generated pointwise wrapper.
    """
    logger.debug("GEMS_KUNLUNXIN EXPAND_COPY")

    # Convert size to tuple and handle -1 (meaning keep original size)
    size_tuple = tuple(-1 if s is None else s for s in size)

    # Ensure input is on the correct device
    device = x.device

    # Create output tensor with target shape on the same device
    out = torch.empty(size_tuple, dtype=x.dtype, device=device)

    # Handle empty tensors
    if out.numel() == 0:
        return out

    # Use torch.expand to get a broadcasted view with correct strides
    view = x.expand(size_tuple)

    # Ensure view is on the right device (expand preserves device)
    if view.device != device:
        view = view.to(device)

    if view.is_contiguous():
        # Same-shape (or full) copy: flat bounded-tile block DMA.
        return copy_(out, view)

    # Broadcast (stride-0 dims): single leaf gather kernel.
    _launch_bcast(view.shape, view.stride(), view, out, out.numel())
    return out