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
import builtins as _builtins
import logging
import math

import torch
import triton
import triton.language as tl

from flag_gems.ops._histogramdd_from_bin_tensors import (
    _validate,
    histogramdd_density_kernel,
    histogramdd_total_kernel,
)
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)


@libentry()
@triton.jit(do_not_specialize=["M", "EDGES_STRIDE", "SEARCH_STEPS", "PARTIAL_STRIDE"])
def histogramdd_bin_tensors_kernel_xpu(
    inpT_ptr,  # (N, M) contiguous input (transposed): dim d's coords are row d
    weight_ptr,  # (M,) contiguous weights, or a 1-element dummy
    out_ptr,  # (N_PROGRAMS, PARTIAL_STRIDE) per-program partial accumulators
    M,
    N,
    edges_ptr,  # (N, EDGES_STRIDE) materialised per-dim bin edges (acc dtype)
    nedges_ptr,  # (N,) number of edges per dimension (= bins + 1)
    strides_ptr,  # (N,) row-major strides in bins
    num_bins,
    PARTIAL_STRIDE,  # num_bins + 1; slot num_bins is a per-row throwaway
    EDGES_STRIDE,
    SEARCH_STEPS,
    N_CONST: tl.constexpr,
    HAS_WEIGHT: tl.constexpr,
    ACC_DTYPE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Multi-dimensional histogram over explicit per-dim bin-EDGE tensors (xpu3).

    Cross-program ``tl.atomic_add`` silently drops colliding updates on xpu3, so
    each program owns a private partial row ``out[pid]`` and only ever scatters
    into that row; the ``histogramdd_partial_reduce_kernel_xpu`` pass then sums
    the rows. Within a program the ``BLOCK_SIZE`` lanes serialise their colliding
    atomic adds. Each program handles one disjoint block of points, so the point
    axis is data-parallel over the grid and there is no per-program chunk loop to
    unroll.

    Bins are explicit (possibly non-uniform) edge tensors, so the generic
    ``(points x edges)`` broadcast compare plus ``tl.sum`` reduction (which fails
    to legalise on xpu3) is replaced by a per-lane binary search over the
    materialised edge row: ``le_count`` = the number of edges ``<= coord`` is
    found in ``SEARCH_STEPS`` runtime iterations, each a plain vector gather at a
    clamped midpoint. The bin index is ``le_count - 1``; a coordinate exactly on
    the top edge is pulled into the rightmost (right-inclusive) bin, matching
    ATen. Invalid / out-of-range / NaN points are redirected to the per-row
    ``num_bins`` scratch slot and contribute 0.
    """
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    row_mask = offs < M
    valid = row_mask
    linear_idx = tl.zeros((BLOCK_SIZE,), dtype=tl.int64)
    for d in tl.static_range(N_CONST):
        coord = tl.load(
            inpT_ptr + d * M + offs, mask=row_mask, other=float("nan")
        ).to(ACC_DTYPE)
        n_edges = tl.load(nedges_ptr + d).to(tl.int32)
        stride = tl.load(strides_ptr + d).to(tl.int64)
        edge_base = edges_ptr + d * EDGES_STRIDE
        n_edges_v = tl.zeros((BLOCK_SIZE,), dtype=tl.int32) + n_edges
        lo = tl.zeros((BLOCK_SIZE,), dtype=tl.int32)
        hi = n_edges_v
        one = tl.full((BLOCK_SIZE,), 1, dtype=tl.int32)
        for _ in range(SEARCH_STEPS):
            mid = (lo + hi) >> one
            mid_c = tl.where(mid >= n_edges_v, n_edges_v - one, mid)
            e = tl.load(edge_base + mid_c).to(ACC_DTYPE)
            go_right = e <= coord
            lo = tl.where(go_right, mid + one, lo)
            hi = tl.where(go_right, hi, mid)
        le_count = lo

        bin_idx = le_count - one
        top_edge = tl.load(edge_base + (n_edges - 1)).to(ACC_DTYPE)
        at_top = coord == top_edge
        bin_idx = tl.where(at_top, n_edges_v - one - one, bin_idx)
        in_range = (
            (bin_idx >= 0) & (bin_idx < (n_edges_v - one)) & (coord <= top_edge)
        )
        valid = valid & in_range
        linear_idx = linear_idx + bin_idx.to(tl.int64) * stride

    dst = tl.where(valid, linear_idx, num_bins)
    row_base = pid.to(tl.int64) * PARTIAL_STRIDE
    if HAS_WEIGHT:
        contrib = tl.load(weight_ptr + offs, mask=valid, other=0.0).to(ACC_DTYPE)
    else:
        contrib = tl.where(valid, 1.0, 0.0).to(ACC_DTYPE)
    tl.atomic_add(out_ptr + row_base + dst, contrib)


@libentry()
@triton.jit(do_not_specialize=["N_PROGRAMS", "PARTIAL_STRIDE"])
def histogramdd_partial_reduce_kernel_xpu(
    partials_ptr,  # (N_PROGRAMS, PARTIAL_STRIDE) per-program partial histograms
    out_ptr,  # (num_bins,) summed histogram
    N_PROGRAMS,
    PARTIAL_STRIDE,
    num_bins,
    ACC_DTYPE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Sum the per-program partial rows into the final flat histogram (xpu3).

    A single runtime loop over the ``N_PROGRAMS`` rows (one gather per row) keeps
    the IR bounded and avoids any cross-program atomics. Only the first
    ``num_bins`` columns are read; the per-row ``num_bins`` scratch slot is
    dropped.
    """
    offs = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    m = offs < num_bins
    acc = tl.zeros((BLOCK_SIZE,), dtype=ACC_DTYPE)
    for p in range(N_PROGRAMS):
        acc += tl.load(
            partials_ptr + p * PARTIAL_STRIDE + offs, mask=m, other=0.0
        ).to(ACC_DTYPE)
    tl.store(out_ptr + offs, acc, mask=m)


@libentry()
@triton.jit
def histogramdd_tensors_volume_kernel_xpu(
    edges_ptr,  # (D, EDGES_STRIDE) materialised per-dim bin edges (acc dtype)
    strides_ptr,  # (D,) int64 row-major strides in bins
    sizes_ptr,  # (D,) int64 bins per dimension
    vol_ptr,  # (num_bins,) out: per-bin volume
    num_bins,
    EDGES_STRIDE: tl.constexpr,
    D_CONST: tl.constexpr,
    ACC_DTYPE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Per-bin volume for the density path, XPU3-legal.

    The generic ``histogramdd_volume_kernel`` takes the per-dimension edge
    tensors as a ``tl.constexpr`` tuple of pointers; the xpu3 launcher rejects
    that with ``unexpected nested constant key`` because it cannot encode a
    tuple-valued constant argument. The edges are instead passed as the single
    materialised ``(D, EDGES_STRIDE)`` block (one contiguous row per dimension)
    that the accumulation kernel already uses, and the per-dimension strides and
    sizes as plain int64 tensors. Each flat bin index is decomposed back into
    per-dimension indices with the row-major strides and the volume is the
    product of the per-dimension edge widths ``edges[d, idx+1] - edges[d, idx]``.
    The gather only ever touches indices in ``[0, bins[d]]`` (``bins[d] <=
    EDGES_STRIDE - 1``), so the per-row +inf padding is never read.
    """
    offs = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    m = offs < num_bins
    offs64 = offs.to(tl.int64)
    vol = tl.full((BLOCK_SIZE,), 1.0, dtype=ACC_DTYPE)
    for d in tl.static_range(D_CONST):
        stride_d = tl.load(strides_ptr + d)
        size_d = tl.load(sizes_ptr + d)
        idx_d = (offs64 // stride_d) % size_d
        edge_base = edges_ptr + d * EDGES_STRIDE
        lo = tl.load(edge_base + idx_d, mask=m, other=0.0).to(ACC_DTYPE)
        hi = tl.load(edge_base + idx_d + 1, mask=m, other=1.0).to(ACC_DTYPE)
        vol = vol * (hi - lo)
    tl.store(vol_ptr + offs, vol, mask=m)


def _apply_density_xpu(
    hist, edges_mat, strides, sizes, D, num_bins, acc_dtype, acc_triton, edges_stride
):
    """Normalise ``hist`` in place by total weight and per-bin volume (xpu3).

    Mirrors the generic ``_apply_density`` (``hist[i] /= total * volume[i]`` with
    no guard on the total, matching ATen's unconditional division so empty /
    out-of-range inputs give NaN and cancelling weights give +-inf), but swaps
    the tuple-of-pointers volume kernel for the single-block
    ``histogramdd_tensors_volume_kernel_xpu``. The flat total/divide kernels are
    reused from the generic module unchanged -- they take plain tensor arguments
    and legalise on xpu3.
    """
    if num_bins == 0:
        return hist
    dev = hist.device
    total = torch.empty((), dtype=acc_dtype, device=dev)
    vol = torch.empty(num_bins, dtype=acc_dtype, device=dev)
    flat = hist if hist.is_contiguous() else hist.contiguous()
    BLOCK = 1024
    grid = (triton.cdiv(num_bins, BLOCK),)
    with torch_device_fn.device(dev):
        histogramdd_total_kernel[(1,)](
            flat, total, num_bins, ACC_DTYPE=acc_triton, BLOCK_SIZE=1024
        )
        histogramdd_tensors_volume_kernel_xpu[grid](
            edges_mat,
            strides,
            sizes,
            vol,
            num_bins,
            EDGES_STRIDE=edges_stride,
            D_CONST=D,
            ACC_DTYPE=acc_triton,
            BLOCK_SIZE=BLOCK,
        )
        histogramdd_density_kernel[grid](
            flat.view(num_bins),
            total,
            vol,
            num_bins,
            ACC_DTYPE=acc_triton,
            BLOCK_SIZE=BLOCK,
        )
    if flat.data_ptr() != hist.data_ptr():
        hist.copy_(flat)
    return hist


def _run_histogram_xpu(self, bins, weight, density, out):
    """Shared driver for the base and ``.out`` variants on xpu3."""
    _validate(self, bins, weight)
    D = len(bins)

    inp = self.reshape(-1, D).contiguous()
    M = inp.shape[0]

    edges_list = [b.contiguous() for b in bins]
    bin_counts = [b.numel() for b in edges_list]  # edges per dim (= bins + 1)
    out_shape = tuple(bc - 1 for bc in bin_counts)
    out_dtype = self.dtype

    acc_dtype = (
        torch.float32 if out_dtype in (torch.float16, torch.bfloat16) else out_dtype
    )
    acc_triton = tl.float64 if acc_dtype == torch.float64 else tl.float32

    has_weight = weight is not None
    if has_weight:
        w = weight.reshape(-1).contiguous()
        if w.dtype != acc_dtype:
            w = w.to(acc_dtype)
        weight_ptr = w
    else:
        weight_ptr = torch.empty(1, dtype=acc_dtype, device=self.device)

    if out is not None:
        if out.dtype != out_dtype:
            raise RuntimeError(
                f"Expected out tensor to have dtype {out_dtype}, but got "
                f"{out.dtype} instead"
            )
        if out.device != self.device:
            raise RuntimeError(
                f"Expected out tensor to have device {self.device}, but got "
                f"{out.device} instead"
            )
        if tuple(out.shape) != out_shape:
            out.resize_(out_shape)

    num_bins = 1
    for bc in out_shape:
        num_bins *= bc

    if num_bins == 0:
        if out is None:
            res = torch.empty(out_shape, dtype=out_dtype, device=self.device)
            return res
        return out

    # Row-major strides in bins.
    strides_host = [1] * D
    for d in _builtins.range(D - 2, -1, -1):
        strides_host[d] = strides_host[d + 1] * out_shape[d + 1]
    strides = torch.tensor(strides_host, dtype=torch.int64, device=inp.device)
    nedges = torch.tensor(bin_counts, dtype=torch.int64, device=inp.device)

    # Materialise the per-dim edge tensors into a padded (D, EDGES_STRIDE) block
    # so the kernel's binary-search gather stays within one contiguous row. The
    # search only ever touches indices in ``[0, bin_counts[d] - 1]`` so the
    # +inf padding is never read.
    edges_stride = max(bin_counts)
    edges_mat = torch.full(
        (D, edges_stride), float("inf"), dtype=acc_dtype, device=inp.device
    )
    for d in _builtins.range(D):
        nb = bin_counts[d]
        edges_mat[d, :nb] = edges_list[d].to(acc_dtype)

    search_steps = max(1, math.ceil(math.log2(edges_stride + 1)))

    partial_stride = num_bins + 1
    hist_flat = torch.zeros(num_bins, dtype=acc_dtype, device=inp.device)

    if M > 0:
        inpT = inp.t().contiguous()
        BLOCK_SIZE = 256
        n_programs = triton.cdiv(M, BLOCK_SIZE)
        partials = torch.zeros(
            n_programs * partial_stride, dtype=acc_dtype, device=inp.device
        )
        with torch_device_fn.device(self.device):
            histogramdd_bin_tensors_kernel_xpu[(n_programs,)](
                inpT,
                weight_ptr,
                partials,
                M,
                D,
                edges_mat,
                nedges,
                strides,
                num_bins,
                partial_stride,
                edges_stride,
                search_steps,
                N_CONST=D,
                HAS_WEIGHT=has_weight,
                ACC_DTYPE=acc_triton,
                BLOCK_SIZE=BLOCK_SIZE,
                num_warps=1,
            )
            RBLOCK = 1024
            histogramdd_partial_reduce_kernel_xpu[(triton.cdiv(num_bins, RBLOCK),)](
                partials,
                hist_flat,
                n_programs,
                partial_stride,
                num_bins,
                ACC_DTYPE=acc_triton,
                BLOCK_SIZE=RBLOCK,
                num_warps=1,
            )

    hist = hist_flat.view(out_shape).contiguous()

    if density:
        sizes = torch.tensor(out_shape, dtype=torch.int64, device=inp.device)
        hist = _apply_density_xpu(
            hist,
            edges_mat,
            strides,
            sizes,
            D,
            num_bins,
            acc_dtype,
            acc_triton,
            edges_stride,
        )

    if out is None:
        return hist if hist.dtype == out_dtype else hist.to(out_dtype)
    out.copy_(hist)
    return out


def _histogramdd_from_bin_tensors(self, bins, *, weight=None, density=False):
    logger.debug("GEMS_KUNLUNXIN _HISTOGRAMDD_FROM_BIN_TENSORS")
    return _run_histogram_xpu(self, bins, weight, density, None)


def _histogramdd_from_bin_tensors_out(
    self, bins, *, weight=None, density=False, out=None
):
    logger.debug("GEMS_KUNLUNXIN _HISTOGRAMDD_FROM_BIN_TENSORS_OUT")
    if out is None:
        raise RuntimeError(
            "_histogramdd_from_bin_tensors.out: 'out' argument is required"
        )
    return _run_histogram_xpu(self, bins, weight, density, out)
