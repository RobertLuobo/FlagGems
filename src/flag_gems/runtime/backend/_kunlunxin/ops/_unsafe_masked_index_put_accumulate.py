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
from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as ext

logger = logging.getLogger(__name__)

# Small masks use the single-launch gather-reduce kernel below (no atomics: the
# TritonXPU backend does not provide a correct atomic_add in any form - see the
# README for the probe evidence - so a scoped scatter-accumulate is not
# expressible there).  The kernel is O(input.numel() * mask.numel()) with a
# small constant, so it is only used while the scan stays cheap; larger masks
# switch to the sort-based segmented-sum path below, which is O(M log M) and
# never re-reads the whole mask for every output element.
#
# Why not bincount / scatter_reduce / index_reduce / histc / cumsum / put /
# index_add / scatter_add / tl.histogram?
#  - bincount / histc / index_reduce_ / scatter_reduce / segment_reduce are
#    overridden by _kunlunxin kernels of the O(output*M) scalar-sequential
#    shape (15s+ at M=1024 for bincount, 86ms @ 65k for scatter_add_).
#  - index_add_ (the vendor's own duplicate-safe segmented sum, measured
#    21.5ms @ 65k) and every custom per-segment kernel with an in-kernel
#    scalar loop (21ms) are secondary to the ~0.1us-per-scalar-load cost of
#    this backend: any implementation that reduces one element per lane with
#    per-lane (gather) addressing lands at the same ~20ms wall.
#  - cumsum / tl.cumsum / tl.histogram are broken or unsupported on this
#    backend (cumsum 99.9% wrong, tl.cumsum ~50% wrong, tl.histogram fails
#    PassManager::run at make_llir).
#  - scatter / native put-accumulate are wrong (maxerr 3-6.5).
#
# So the selected-lane histogram is assembled from verified-correct
# primitives (argsort, gather, nonzero) and ONE custom store-only kernel.
# The segment sums themselves are computed as a dense (S x U) padded-window
# gather + sum on the sorted array (U = next_pow2(max segment length), which
# is ~5 for the benchmark's 50% density): every value is loaded by a fast
# vectorized torc.gather (the vendor gather is 0.1ms-class) and no kernel
# touch per-lane scalar loads at all.  If some segment is longer than
# _UNROLL (adversarial input), the scalar-sequential fallback kernel is used.
_GATHER_LIMIT = 4096
_UNROLL = 32


@libentry()
@triton.jit(do_not_specialize=["mask_numel"])
def _unsafe_masked_index_put_accumulate_kernel(
    input,
    mask,
    index0,
    index1,
    index2,
    values,
    mask_numel,
    SHAPE0: tl.constexpr,
    SHAPE1: tl.constexpr,
    SHAPE2: tl.constexpr,
    STRIDE0: tl.constexpr,
    STRIDE1: tl.constexpr,
    STRIDE2: tl.constexpr,
    RANK: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    output_offset = ext.program_id(0)

    if RANK == 1:
        coordinate0 = output_offset
        coordinate1 = 0
        coordinate2 = 0
        input_offset = coordinate0 * STRIDE0
    elif RANK == 2:
        coordinate0 = output_offset // SHAPE1
        coordinate1 = output_offset % SHAPE1
        coordinate2 = 0
        input_offset = coordinate0 * STRIDE0 + coordinate1 * STRIDE1
    else:
        coordinate0 = output_offset // (SHAPE1 * SHAPE2)
        remainder = output_offset % (SHAPE1 * SHAPE2)
        coordinate1 = remainder // SHAPE2
        coordinate2 = remainder % SHAPE2
        input_offset = (
            coordinate0 * STRIDE0 + coordinate1 * STRIDE1 + coordinate2 * STRIDE2
        )

    offsets = tl.arange(0, BLOCK_SIZE)
    active = offsets < mask_numel
    selected = tl.load(mask + offsets, mask=active, other=0) != 0
    selected &= (
        tl.load(index0 + offsets, mask=active, other=0).to(tl.int32) == coordinate0
    )
    if RANK >= 2:
        selected &= (
            tl.load(index1 + offsets, mask=active, other=0).to(tl.int32) == coordinate1
        )
    if RANK == 3:
        selected &= (
            tl.load(index2 + offsets, mask=active, other=0).to(tl.int32) == coordinate2
        )

    update_values = tl.load(values + offsets, mask=active, other=0.0).to(tl.float32)
    update = tl.sum(tl.where(selected, update_values, 0.0), axis=0)
    original = tl.load(input + input_offset).to(tl.float32)
    tl.store(input + input_offset, original + update)


def _unsafe_masked_index_put_accumulate_gather(input, mask, indices, values):
    # O(M) launch, O(N*M) work scan-free per output element; fastest option
    # while the scan stays small (single kernel launch, no torch dispatch).
    mask_contiguous = mask.contiguous()
    values_contiguous = values.contiguous()
    contiguous_indices = [index.contiguous() for index in indices]
    while len(contiguous_indices) < 3:
        contiguous_indices.append(contiguous_indices[0])

    shape = list(input.shape) + [1] * (3 - input.ndim)
    strides = list(input.stride()) + [0] * (3 - input.ndim)
    block_size = triton.next_power_of_2(mask.numel())

    with torch_device_fn.device(input.device):
        _unsafe_masked_index_put_accumulate_kernel[(input.numel(),)](
            input,
            mask_contiguous,
            contiguous_indices[0],
            contiguous_indices[1],
            contiguous_indices[2],
            values_contiguous,
            mask.numel(),
            SHAPE0=shape[0],
            SHAPE1=shape[1],
            SHAPE2=shape[2],
            STRIDE0=strides[0],
            STRIDE1=strides[1],
            STRIDE2=strides[2],
            RANK=input.ndim,
            BLOCK_SIZE=block_size,
            isCloseVectorization=True,
            buffer_size_limit=2048,
        )
    return input


def _xsr_edges(sf32, k):
    """Segment-start / segment-length of the equal-key runs of a sorted int32
    key array (boundaries already computed).  Returns (heads, lens), both
    int64 (S,)."""
    boundaries = torch.nonzero(sf32[1:] != sf32[:-1]).flatten().to(torch.int64)
    heads = torch.cat(
        [
            torch.zeros(1, dtype=torch.int64, device=sf32.device),
            boundaries + 1,
        ]
    )
    tails = torch.cat(
        [boundaries, torch.tensor([k - 1], dtype=torch.int64, device=sf32.device)]
    )
    return heads, tails - heads + 1


@libentry()
@triton.jit
def _segsum_store_kernel(
    sums,
    keys,
    out,
    S,
    BLOCK: tl.constexpr,
):
    # Store-only: out[keys[i]] = sums[i] for i < S.  keys are pairwise
    # distinct (one per segment), so no two programs race on a slot.  Masked
    # stores are not honoured on this backend, so tail lanes (offs >= S)
    # replicate lane 0's exact (address, value) pair, making their extra
    # stores idempotent.
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    safe = tl.where(offs < S, offs, 0)
    v = tl.load(sums + safe)
    key = tl.load(keys + safe).to(tl.int64)
    tl.store(out + key, v)


@libentry()
@triton.jit(do_not_specialize=["K", "S"])
def _segsum_scatter_kernel(
    sw,
    sf,
    boundaries,
    out,
    K,
    S,
):
    # Adversarial fallback (a segment longer than _UNROLL): one program per
    # contiguous run of equal keys in the sorted arrays, scalar-sequential
    # accumulation over [start, end] (the dynamic-bound while loop is the
    # vendor-proven pattern on this backend, cf. _scatter_reduce_prod_kernel),
    # then one conflict-free store: every key belongs to exactly one segment,
    # so no two programs write the same slot.
    seg = tl.program_id(0)
    if seg == 0:
        start = 0
    else:
        start = tl.load(boundaries + seg - 1).to(tl.int32) + 1
    if seg == S - 1:
        end = K - 1
    else:
        end = tl.load(boundaries + seg).to(tl.int32)
    acc = 0.0
    i = start
    while i <= end:
        acc += tl.load(sw + i).to(tl.float32)
        i += 1
    key = tl.load(sf + end).to(tl.int64)
    tl.store(out + key, acc)


def _unsafe_masked_index_put_accumulate_histogram(input, mask, indices, values):
    # O(M log M) exact duplicate accumulation, no atomics, no per-lane scalar
    # kernels.  Plan (every piece was verified correct on this backend):
    #   1. compact the selected lanes (a lane with value * mask == 0 adds 0)
    #   2. argsort the flat destination and gather the sorted keys/weights
    #   3. segment boundaries are the positions where the sorted key changes
    #   4. segment sums are computed as a dense (S x U) padded-window gather +
    #      a row reduction (U = next_pow2(max segment length) <= _UNROLL; this
    #      keeps every value on the vendor's fast vectorized gather path and
    #      avoids the ~20ms per-lane-scalar-load wall of any in-kernel segment
    #      sum; the vendor's own index_add_ duplicate path hits that same
    #      wall: 21.5ms @ 65k).
    #   5. one store-only kernel scatters the S sums at their (distinct) keys.
    numel = input.numel()
    flat = indices[0].to(torch.int64)
    for d in range(1, len(indices)):
        flat = flat * input.shape[d] + indices[d].to(torch.int64)
    flat = flat.reshape(-1)

    weights = (values * (mask != 0)).to(torch.float32).reshape(-1)
    selected = torch.nonzero(weights != 0.0).flatten()
    k = selected.numel()
    if k == 0:
        return input.to(torch.float32).to(input.dtype)
    f = torch.gather(flat, 0, selected)
    w = torch.gather(weights, 0, selected)

    # NOTE: argsort/gather must stay int64 (the vendor int32 argsort is
    # non-deterministic: it intermittently returns a numel-sized permutation,
    # corrupting every later stage).  int32 is used only for the `!=` boundary
    # test, whose int64 form crashes the XPU legalizer (triton_xpu.cmpf).
    order = torch.argsort(f)
    sf = torch.gather(f, 0, order)
    sw = torch.gather(w, 0, order)
    sf32 = sf.to(torch.int32)
    heads, lens = _xsr_edges(sf32, k)
    len_max = int(lens.max().item())

    out = torch.zeros(numel, dtype=torch.float32, device=input.device)
    if len_max <= _UNROLL:
        # Dense (S x U) window: sums[i] = sum_{t < lens[i]} sw[heads[i] + t].
        # sw is padded with u zeros so every index is in-bounds; the row sums
        # go through the vendor einsum (1.7ms @ 51k x 8) because this
        # backend's 2-D reduce (sum(dim=1), 4-9ms and WRONG for this shape)
        # and torch.mm (error ~11 for (S x 8) @ (8 x 1)) are both broken for
        # the 2-D window, and every in-kernel per-lane gather is ~20ms.
        s = lens.shape[0]
        u = triton.next_power_of_2(max(len_max, 1))
        offs = torch.arange(u, dtype=torch.int32, device=input.device)
        sw_pad = torch.cat(
            [sw, torch.zeros(u, dtype=torch.float32, device=input.device)]
        )
        idx2 = (heads[:, None] + offs[None, :].to(torch.int64)).reshape(-1)
        vals = torch.gather(sw_pad, 0, idx2).reshape(s, u)
        m2 = (offs[None, :] < lens.to(torch.int32)[:, None]).to(torch.float32)
        sums = torch.einsum("su,su->s", vals, m2)
        keys = torch.gather(sf, 0, heads)
        with torch_device_fn.device(input.device):
            _segsum_store_kernel[(triton.cdiv(s, 1024),)](
                sums,
                keys,
                out,
                s,
                BLOCK=1024,
                isCloseVectorization=True,
                buffer_size_limit=2048,
            )
    else:
        # Adversarial long segments: scalar-sequential per-segment sum.
        boundaries = torch.nonzero(sf32[1:] != sf32[:-1]).flatten()
        n_segments = boundaries.numel() + 1
        with torch_device_fn.device(input.device):
            _segsum_scatter_kernel[(n_segments,)](
                sw,
                sf,
                boundaries,
                out,
                K=k,
                S=n_segments,
                isCloseVectorization=True,
                buffer_size_limit=2048,
            )
    return (input.to(torch.float32) + out.view(input.shape)).to(input.dtype)


def _unsafe_masked_index_put_accumulate(input, mask, indices, values):
    logger.debug("GEMS_KUNLUNXIN _UNSAFE_MASKED_INDEX_PUT_ACCUMULATE")
    rank = input.ndim
    if rank < 1 or rank > 3 or len(indices) != rank:
        raise RuntimeError(
            "Kunlunxin _unsafe_masked_index_put_accumulate supports ranks 1 to 3"
        )
    if input.numel() == 0 or mask.numel() == 0:
        return input

    if mask.numel() <= _GATHER_LIMIT:
        return _unsafe_masked_index_put_accumulate_gather(input, mask, indices, values)
    return _unsafe_masked_index_put_accumulate_histogram(input, mask, indices, values)
