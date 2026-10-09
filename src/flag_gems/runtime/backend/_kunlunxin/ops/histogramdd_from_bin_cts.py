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

from flag_gems.ops._histogramdd_from_bin_cts import (
    _format_bound,
    histogramdd_bin_geometry_kernel,
    histogramdd_density_kernel,
    histogramdd_log_volume_kernel,
    histogramdd_total_scale_kernel,
)
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as ext

logger = logging.getLogger(__name__)


@libentry()
@triton.jit
def histogramdd_col_range_kernel_xpu(
    inpT_ptr,  # (N, M) contiguous: each dimension's coords in a contiguous row
    left_ptr,  # (N,) out: per-dimension left edge
    right_ptr,  # (N,) out: per-dimension right edge
    flag_ptr,  # (N,) out: d if dim d is non-finite else N (sentinel)
    M,
    N,
    ACC_DTYPE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Per-dimension min/max + non-finite flag, one program per dimension.

    xpu3 rejects ``tl.min/tl.max`` with ``axis=0`` on a 2-D tile ("axis must not
    be 0 for 2-D shapes") and mis-compiles a reduce fed by a run-time-strided
    discrete load, which is what the generic (M, N) tiled range kernel relies on.
    Transposing to (N, M) on the host turns each dimension's column into a
    contiguous row, so this program reduces a plain 1-D contiguous load -- the
    only reduce shape xpu3 legalises here.

    NaN / +-Inf detection is folded into the same two reductions instead of
    separate ``tl.sum``/``tl.max`` reduces (those fail to legalise on xpu3): a NaN
    is mapped to -inf for the min feed and +inf for the max feed, so a column that
    is non-finite ends with ``amin == -inf`` or ``amax == +inf``. The exact ATen
    bounds for the error message are recomputed on the host error path only.
    """
    d = ext.program_id(0)
    amin = tl.full((), float("inf"), ACC_DTYPE)
    amax = tl.full((), float("-inf"), ACC_DTYPE)
    row = inpT_ptr + d * M
    for base in range(0, M, BLOCK_SIZE):
        offs = base + tl.arange(0, BLOCK_SIZE)
        m = offs < M
        vals = tl.load(row + offs, mask=m, other=float("nan")).to(ACC_DTYPE)
        isnan = vals != vals
        minf = tl.where(m, tl.where(isnan, float("-inf"), vals), float("inf"))
        maxf = tl.where(m, tl.where(isnan, float("inf"), vals), float("-inf"))
        amin = tl.minimum(amin, tl.min(minf))
        amax = tl.maximum(amax, tl.max(maxf))

    nonfinite = (amax == float("inf")) | (amin == float("-inf"))
    same = amin == amax
    lf = tl.where(same, amin - 0.5, amin)
    rt = tl.where(same, amax + 0.5, amax)
    tl.store(left_ptr + d, lf)
    tl.store(right_ptr + d, rt)
    tl.store(flag_ptr + d, tl.where(nonfinite, d, N))


@libentry()
@triton.jit(do_not_specialize=["M", "num_bins"])
def histogramdd_bin_cts_kernel_xpu(
    inpT_ptr,  # (N, M) contiguous input (transposed)
    weight_ptr,  # (M,) contiguous weights, or a 1-element dummy
    out_ptr,  # (num_bins + 1,) accumulator; slot num_bins is a throwaway
    M,
    N,
    left_ptr,  # (N,) left edges (fp64)
    right_ptr,  # (N,) right edges (fp64)
    edges_ptr,  # (N, EDGES_STRIDE) materialised per-dim bin edges (input dtype)
    strides_ptr,  # (N,) row-major strides in bins
    bins_ptr,  # (N,) bins per dimension
    num_bins,
    EDGES_STRIDE: tl.constexpr,
    N_CONST: tl.constexpr,
    N_CHUNKS: tl.constexpr,
    HAS_WEIGHT: tl.constexpr,
    ACC_DTYPE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Multi-dimensional histogram accumulation launched with a single program.

    Two xpu3 behaviours drive this kernel's shape. First, cross-program
    ``tl.atomic_add`` silently drops colliding updates on xpu3, and a histogram is
    a collision-heavy scatter, so the whole accumulation runs inside one program
    (``grid=(1,)``) that loops over the points in ``BLOCK_SIZE`` chunks -- every
    atomic add then lands in the same program, where collisions serialise. This is
    the same single-program pattern ``put_accumulate_kernel`` uses.

    Second, the per-dimension unrolled loop drives bin placement with ATen's own
    linear formula ``bin = (coord - left) / (right - left) * nbins`` computed in
    the input dtype, then floored and clamped. ATen's bin_cts path computes bins
    this way (not by searching materialised edges), then applies a one-step
    rounding correction against the two neighbouring edges to absorb the float
    error in that division: if ``edge[bin + 1] <= coord`` the point belongs one
    bin higher, and if ``edge[bin] > coord`` one bin lower. Reproducing that
    correction against the materialised ``edges`` (gathered at the computed bin,
    a tensor index, so it legalises on xpu3 unlike the scalar per-edge scan) is
    what makes points sitting exactly on an edge land in ATen's bin bit-for-bit.
    Invalid / out-of-range / NaN points are redirected to the ``num_bins``
    scratch slot and contribute 0, so the atomic add needs no mask (masked
    discrete scatter writes masked lanes to offset 0 on xpu3).
    """
    base_ar = tl.arange(0, BLOCK_SIZE)
    for i in range(N_CHUNKS):
        offs = i * BLOCK_SIZE + base_ar
        row_mask = offs < M
        valid = row_mask
        linear_idx = tl.zeros((BLOCK_SIZE,), dtype=tl.int64)
        for d in range(N_CONST):
            coord = tl.load(inpT_ptr + d * M + offs, mask=row_mask, other=float("nan")).to(
                ACC_DTYPE
            )
            left = tl.load(left_ptr + d).to(ACC_DTYPE)
            right = tl.load(right_ptr + d).to(ACC_DTYPE)
            nbins = tl.load(bins_ptr + d)
            stride = tl.load(strides_ptr + d)

            in_range = (coord >= left) & (coord <= right)
            valid = valid & in_range

            pos = (coord - left) / (right - left) * nbins.to(ACC_DTYPE)
            bin_idx = pos.to(tl.int64)
            bin_idx = tl.where(bin_idx >= nbins, nbins - 1, bin_idx)
            bin_idx = tl.where(bin_idx < 0, 0, bin_idx)

            edge_base = edges_ptr + d * EDGES_STRIDE
            e_hi = tl.load(edge_base + (bin_idx + 1)).to(ACC_DTYPE)
            e_lo = tl.load(edge_base + bin_idx).to(ACC_DTYPE)
            bump_up = (bin_idx != nbins - 1) & (e_hi <= coord)
            bump_dn = (bin_idx != 0) & (e_lo > coord)
            bin_idx = tl.where(bump_up, bin_idx + 1, tl.where(bump_dn, bin_idx - 1, bin_idx))

            linear_idx = linear_idx + bin_idx * stride

        dst = tl.where(valid, linear_idx, num_bins)
        if HAS_WEIGHT:
            contrib = tl.load(weight_ptr + offs, mask=valid, other=0.0).to(ACC_DTYPE)
        else:
            contrib = tl.where(valid, 1.0, 0.0).to(ACC_DTYPE)
        tl.atomic_add(out_ptr + dst, contrib)


def _resolve_range(inp, range_, N, acc_dtype):
    """Return (lefts, rights) fp64 tensors of shape (N,) on the input device.

    Mirrors the generic host logic for an explicit ``range`` (validation, zero
    width widening). For an auto-range it reduces the data on device with the
    transpose-based ``histogramdd_col_range_kernel_xpu`` and raises on a
    non-finite dimension, matching ATen's message. The exact ATen bounds for the
    error string are recomputed on the host error path only (``torch.aminmax``
    reproduces NaN propagation and inf exactly).
    """
    if range_ is not None:
        rng = [float(v) for v in range_]
        if len(rng) != 2 * N:
            raise RuntimeError(
                f"torch.histogramdd: for a {N}-dimensional histogram range should "
                f"have {2 * N} elements, but got {len(rng)}"
            )
        for d in range(N):
            lo, hi = rng[2 * d], rng[2 * d + 1]
            if not (math.isfinite(lo) and math.isfinite(hi)):
                raise RuntimeError(
                    f"torch.histogramdd: dimension {d}'s range "
                    f"[{_format_bound(lo)}, {_format_bound(hi)}] is not finite"
                )
            if lo > hi:
                raise RuntimeError(
                    f"torch.histogramdd: min should not exceed max, but got "
                    f"min {_format_bound(lo)} max {_format_bound(hi)} for "
                    f"dimension {d}"
                )
        for d in range(N):
            if rng[2 * d] == rng[2 * d + 1]:
                rng[2 * d] -= 0.5
                rng[2 * d + 1] += 0.5
        lefts = torch.tensor(rng[0::2], dtype=torch.float64, device=inp.device)
        rights = torch.tensor(rng[1::2], dtype=torch.float64, device=inp.device)
        return lefts, rights

    lefts = torch.empty(N, dtype=torch.float64, device=inp.device)
    rights = torch.empty(N, dtype=torch.float64, device=inp.device)
    M = inp.shape[0]
    if M == 0:
        lefts.fill_(0.0)
        rights.fill_(1.0)
        return lefts, rights

    # (N, M) contiguous so each dimension's coords are a contiguous row. The
    # reduction runs in the input dtype so an fp64 input keeps full precision in
    # its auto-range (narrowing to fp32 would shift points sitting within one
    # fp32 ulp of a bin boundary).
    acc_triton_dtype = tl.float64 if acc_dtype == torch.float64 else tl.float32
    inpT = inp.t().contiguous()
    lefts_acc = torch.empty(N, dtype=acc_dtype, device=inp.device)
    rights_acc = torch.empty(N, dtype=acc_dtype, device=inp.device)
    flags = torch.empty(N, dtype=torch.int32, device=inp.device)
    with torch_device_fn.device(inp.device):
        histogramdd_col_range_kernel_xpu[(N,)](
            inpT,
            lefts_acc,
            rights_acc,
            flags,
            M,
            N,
            ACC_DTYPE=acc_triton_dtype,
            BLOCK_SIZE=1024,
        )
    dim = int(flags.min().item())
    if dim < N:
        # Error path only: reproduce ATen's exact bounds for the non-finite
        # dimension. ATen's amin/amax propagate NaN (a column with any NaN
        # reports [nan, nan]); XPU's aminmax silently drops NaN, so detect it
        # explicitly. Inf participates in aminmax correctly on both.
        col = inp[:, dim]
        if torch.isnan(col).any().item():
            mn_v = mx_v = float("nan")
        else:
            mn, mx = torch.aminmax(col)
            mn_v, mx_v = mn.item(), mx.item()
        raise RuntimeError(
            f"torch.histogramdd: dimension {dim}'s range "
            f"[{_format_bound(mn_v)}, {_format_bound(mx_v)}] is not finite"
        )
    lefts.copy_(lefts_acc.to(torch.float64))
    rights.copy_(rights_acc.to(torch.float64))
    return lefts, rights


def _histogramdd_from_bin_cts_impl(
    self, bins, *, range=None, weight=None, density=False, out=None
):
    """Shared implementation for the base and ``.out`` ATen variants (xpu3)."""
    logger.debug("GEMS_KUNLUNXIN _HISTOGRAMDD_FROM_BIN_CTS")

    if self.ndim < 2:
        raise RuntimeError(
            "torch.histogramdd: input tensor should have at least 2 dimensions"
        )
    if self.shape[-1] != len(bins):
        raise RuntimeError(
            "histogramdd: The size of bins must be equal to the innermost "
            "dimension of the input."
        )
    if not self.is_floating_point():
        raise NotImplementedError('"histogramdd" not implemented for this dtype')
    if weight is not None and weight.dtype != self.dtype:
        raise RuntimeError(
            "torch.histogramdd: if weight tensor is provided, input tensor and "
            "weight tensor should have the same dtype, but got "
            f"input({self.dtype}), and weight({weight.dtype})"
        )

    N = self.shape[-1]
    bins_list = [int(b) for b in bins]
    out_shape = tuple(bins_list)

    inp = self.reshape(-1, N).contiguous()
    M = inp.shape[0]
    out_dtype = self.dtype

    acc_dtype = self.dtype
    acc_triton_dtype = tl.float64 if acc_dtype == torch.float64 else tl.float32

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
    else:
        out = torch.empty(out_shape, dtype=out_dtype, device=self.device)

    if out.numel() == 0:
        out.zero_()
        return out

    lefts, rights = _resolve_range(inp, range, N, acc_dtype)

    strides_host = [1] * N
    for d in _builtins.range(N - 2, -1, -1):
        strides_host[d] = strides_host[d + 1] * bins_list[d + 1]
    strides = torch.tensor(strides_host, dtype=torch.int64, device=inp.device)
    bins_tensor = torch.tensor(bins_list, dtype=torch.int64, device=inp.device)

    # Materialise ATen's per-dimension bin edges (linspace in the input dtype)
    # so the kernel can reproduce ATen's one-step rounding correction exactly.
    # Padded to a common width; the gather only ever touches indices in
    # ``[0, bins[d]]`` so the padding is never read.
    edges_stride = max(bins_list) + 1
    edges = torch.zeros(N, edges_stride, dtype=acc_dtype, device=inp.device)
    lefts_host = lefts.tolist()
    rights_host = rights.tolist()
    for d in _builtins.range(N):
        nb = bins_list[d]
        edges[d, : nb + 1] = torch.linspace(
            lefts_host[d], rights_host[d], nb + 1, dtype=acc_dtype, device=inp.device
        )

    widths = torch.empty(N, dtype=torch.float64, device=inp.device)
    bin_widths = torch.empty(N, dtype=torch.float64, device=inp.device)
    BLOCK_N = triton.next_power_of_2(N)
    with torch_device_fn.device(self.device):
        histogramdd_bin_geometry_kernel[(triton.cdiv(N, BLOCK_N),)](
            lefts,
            rights,
            bins_tensor,
            widths,
            bin_widths,
            N,
            ACC_DTYPE=tl.float64,
            BLOCK_N=BLOCK_N,
        )

    num_bins = 1
    for b in bins_list:
        num_bins *= b

    # One extra scratch slot absorbs out-of-range / masked points so the
    # single-program accumulate needs no masked discrete scatter.
    hist_acc = torch.zeros(num_bins + 1, dtype=acc_dtype, device=inp.device)

    if M > 0:
        inpT = inp.t().contiguous()
        BLOCK_SIZE = 1024
        n_chunks = triton.cdiv(M, BLOCK_SIZE)

        has_weight = weight is not None
        if has_weight:
            weight = weight.reshape(-1).contiguous()
            if weight.numel() != M:
                raise RuntimeError(
                    "_histogramdd_from_bin_cts: weight must have the same number "
                    "of rows as the input"
                )
            weight_ptr = weight
        else:
            weight_ptr = torch.empty(1, dtype=inp.dtype, device=inp.device)

        with torch_device_fn.device(self.device):
            histogramdd_bin_cts_kernel_xpu[(1,)](
                inpT,
                weight_ptr,
                hist_acc,
                M,
                N,
                lefts,
                rights,
                edges,
                strides,
                bins_tensor,
                num_bins,
                EDGES_STRIDE=edges_stride,
                N_CONST=N,
                N_CHUNKS=n_chunks,
                HAS_WEIGHT=has_weight,
                ACC_DTYPE=acc_triton_dtype,
                BLOCK_SIZE=BLOCK_SIZE,
                num_warps=1,
            )

    hist_counts = hist_acc[:num_bins]

    if density:
        log_vol = torch.empty((), dtype=acc_dtype, device=self.device)
        vol_sign = torch.empty((), dtype=acc_dtype, device=self.device)
        scale_t = torch.empty((), dtype=acc_dtype, device=self.device)
        with torch_device_fn.device(self.device):
            histogramdd_log_volume_kernel[(1,)](
                widths,
                bins_tensor,
                log_vol,
                vol_sign,
                N,
                ACC_DTYPE=acc_triton_dtype,
                BLOCK_N=BLOCK_N,
            )
            histogramdd_total_scale_kernel[(1,)](
                hist_counts,
                log_vol,
                vol_sign,
                scale_t,
                num_bins,
                ACC_DTYPE=acc_triton_dtype,
                BLOCK_SIZE=1024,
            )
        dense = out if out.is_contiguous() else torch.empty_like(out)
        out_flat = dense.view(num_bins)
        BLOCK = 1024
        grid = (triton.cdiv(num_bins, BLOCK),)
        with torch_device_fn.device(self.device):
            histogramdd_density_kernel[grid](
                hist_counts,
                out_flat,
                scale_t,
                num_bins,
                ACC_DTYPE=acc_triton_dtype,
                BLOCK_SIZE=BLOCK,
            )
        if dense is not out:
            out.copy_(dense)
    else:
        out.copy_(hist_counts.view(out_shape))

    return out


def _histogramdd_from_bin_cts(self, bins, *, range=None, weight=None, density=False):
    return _histogramdd_from_bin_cts_impl(
        self, bins, range=range, weight=weight, density=density, out=None
    )


def _histogramdd_from_bin_cts_out(
    self, bins, *, range=None, weight=None, density=False, out
):
    return _histogramdd_from_bin_cts_impl(
        self, bins, range=range, weight=weight, density=density, out=out
    )
