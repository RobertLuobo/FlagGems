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

logger = logging.getLogger(__name__)


# Kunlunxin (XPU) override of reflection_pad2d_backward.
#
# The previous implementation composed `narrow` / `flip` / `add_` / `to`
# (~10 launches, XDNN + slow in-place add_) and the first single-kernel
# attempt (one 9-load gather program per (row, w-chunk)) both scored
# ~0.002-0.077 Gems Speedup (100-300x SLOWER than torch native / XDNN) on
# this backend: XPU triton is launch-bound (~75-120ns/program), so one
# program per row with BLOCK<=256 leaves the device under-occupied.
#
# Math: the forward op is output[oh, ow] = input[R(oh - pt), R(ow - pl)]
# with the reflection R(t) = -t (t < 0), 2n-2-t (t >= n).  Each input
# element (ih, iw) receives contributions from at most 3 output rows x 3
# output columns:
#   h-positions: {ih+pt} U {pt-ih : 1<=ih<=pt} U {2H-2-ih+pt : H-1-pb<=ih<=H-2}
#   (mirrored for w with pl/pr), i.e. a 3x3 gather.
#
# Design (atomic-free, single-writer, 3 kernels, all 1D-flattened
# BLOCK=1024 so the launch-bound XPU backend stays under-occupied):
#   1. bulk: gi[nc, ih, iw] = go[nc, ih+pt, iw+pl] for EVERY input cell --
#      one strided copy.  Flattened 1D, BLOCK=1024, so triton reaches its
#      ~165GB/s DMA ceiling; tail lanes are clamped and the store masked.
#   2. row-edge: REWRITES complete values (9-term gather, clamped addresses,
#      register-level tl.where) for the reflection rows E = [1,pt] U
#      [H-1-pb, H-2] (union; may overlap for small H, deduped on host).
#      One lane per (nc, edge row, w) flattened; the row ids come from a
#      host-built int32 tensor.
#   3. col-edge: REWRITES complete values (center + 2 w-reflections, 3-term
#      gather) for the reflection columns C = [1,pl] U [W-1-pr, W-2] on
#      non-E rows (union; overlap handled by host-side dedupe).
#   The three kernels write disjoint cell sets (the bulk's writes to edge
#   cells are dead; edge kernels are single-writer), so no atomics and no
#   ordering hazard between them.
#
# XPU safety rules (see also replication_pad2d_backward, same file set):
#   - All load ADDRESSES are pre-clamped into [0, dim) and masks applied at
#     the register level with tl.where; no masked-load result feeds a sum
#     (masked loads may read real memory on masked lanes).
#   - All accumulation is fp32 in registers; stores explicitly cast to the
#     output element type so fp16/bf16 results round once from fp32.
#   - Address arithmetic: per-program scalar base + lane offset.
#   - 1D blocks only: BLOCK=1024 for all three kernels (launch-bound at
#     smaller blocks; each of these kernels runs in at most a few
#     hundred programs for the whole benchmark matrix, versus ~10^5-10^6
#     for the row-chunked variants).  NOTE: the UNMASKED vector store
#     miscompiles at small block sizes (silent wrong values); the correct
#     fast path is a masked store at BLOCK=1024.
@triton.jit
def _reflection_pad2d_backward_bulk_kernel(
    go_ptr,
    gi_ptr,
    H,
    W,
    OW,
    OHW,
    pt,
    pl,
    total,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    idx = pid * BLOCK + tl.arange(0, BLOCK)
    idxc = tl.minimum(idx, total - 1)
    row = idxc // W
    col = idxc - row * W
    nc = row // H
    ih = row - nc * H
    src = nc * OHW + (ih + pt) * OW + col + pl
    v = tl.load(go_ptr + src).to(tl.float32)
    tl.store(gi_ptr + idx, v, mask=idx < total)


@triton.jit
def _reflection_pad2d_backward_row_edge_kernel(
    go_ptr,
    gi_ptr,
    row_ids_ptr,
    H,
    W,
    OH,
    OW,
    HW,
    OHW,
    pt,
    pl,
    pb,
    pr,
    N_EDGE_ROWS: tl.constexpr,
    total,
    BLOCK: tl.constexpr,
):
    # One lane per (nc, edge row, w); every lane is a distinct grad_input
    # cell (single writer); only the tail lanes are masked.
    pid = tl.program_id(0)
    o = pid * BLOCK + tl.arange(0, BLOCK)
    oc = tl.minimum(o, total - 1)
    m = o < total

    iw = oc % W
    rest = oc // W
    e = rest % N_EDGE_ROWS
    nc = rest // N_EDGE_ROWS
    ih = tl.load(row_ids_ptr + e)

    h_c = ih + pt  # always in [0, OH)
    h_t = tl.minimum(tl.maximum(pt - ih, 0), OH - 1)
    h_b = tl.minimum(tl.maximum(2 * H - 2 - ih + pt, 0), OH - 1)
    v_t = (ih > 0) & (ih <= pt)
    v_b = (ih >= H - 1 - pb) & (ih < H - 1)

    w_c = iw + pl  # always in [0, OW)
    w_l = tl.minimum(tl.maximum(pl - iw, 0), OW - 1)
    w_r = tl.minimum(tl.maximum(2 * W - 2 - iw + pl, 0), OW - 1)
    m_l = (iw > 0) & (iw <= pl)
    m_r = (iw >= W - 1 - pr) & (iw < W - 1)

    out_base = nc * OHW + h_c * OW
    in_base = nc * HW + ih * W

    acc = tl.zeros((BLOCK,), dtype=tl.float32)

    v = tl.load(go_ptr + out_base + w_c).to(tl.float32)
    acc += tl.where(m, v, 0.0)
    v = tl.load(go_ptr + out_base + w_l).to(tl.float32)
    acc += tl.where(m & m_l, v, 0.0)
    v = tl.load(go_ptr + out_base + w_r).to(tl.float32)
    acc += tl.where(m & m_r, v, 0.0)

    top_base = nc * OHW + h_t * OW
    v = tl.load(go_ptr + top_base + w_c).to(tl.float32)
    acc += tl.where(v_t & m, v, 0.0)
    v = tl.load(go_ptr + top_base + w_l).to(tl.float32)
    acc += tl.where(v_t & m & m_l, v, 0.0)
    v = tl.load(go_ptr + top_base + w_r).to(tl.float32)
    acc += tl.where(v_t & m & m_r, v, 0.0)

    bot_base = nc * OHW + h_b * OW
    v = tl.load(go_ptr + bot_base + w_c).to(tl.float32)
    acc += tl.where(v_b & m, v, 0.0)
    v = tl.load(go_ptr + bot_base + w_l).to(tl.float32)
    acc += tl.where(v_b & m & m_l, v, 0.0)
    v = tl.load(go_ptr + bot_base + w_r).to(tl.float32)
    acc += tl.where(v_b & m & m_r, v, 0.0)

    tl.store(gi_ptr + in_base + iw, acc.to(gi_ptr.type.element_ty), mask=m)


@triton.jit
def _reflection_pad2d_backward_col_edge_kernel(
    go_ptr,
    gi_ptr,
    row_ids_ptr,
    col_ids_ptr,
    H,
    W,
    OW,
    HW,
    OHW,
    pt,
    pl,
    pr,
    N_PLAIN_ROWS: tl.constexpr,
    N_EDGE_COLS: tl.constexpr,
    total,
    BLOCK: tl.constexpr,
):
    # One lane per (nc, plain row, edge column).  Every lane is a distinct
    # grad_input cell (single writer); only the tail lanes are masked.
    pid = tl.program_id(0)
    o = pid * BLOCK + tl.arange(0, BLOCK)
    oc = tl.minimum(o, total - 1)
    m = o < total

    ci = oc % N_EDGE_COLS
    rest = oc // N_EDGE_COLS
    rp = rest % N_PLAIN_ROWS
    nc = rest // N_PLAIN_ROWS

    ih = tl.load(row_ids_ptr + rp)
    w_c = tl.load(col_ids_ptr + ci)
    # w_c is an input column: the center output column is w_c + pl
    # (always in [0, OW)), the reflected ones are pl - w_c and
    # 2W-2-w_c+pl (clamped; masked by m_l/m_r).
    w_ctr = w_c + pl
    w_l = tl.minimum(tl.maximum(pl - w_c, 0), OW - 1)
    w_r = tl.minimum(tl.maximum(2 * W - 2 - w_c + pl, 0), OW - 1)
    m_l = (w_c > 0) & (w_c <= pl)
    m_r = (w_c >= W - 1 - pr) & (w_c < W - 1)

    out_base = nc * OHW + (ih + pt) * OW
    in_base = nc * HW + ih * W

    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    v = tl.load(go_ptr + out_base + w_ctr).to(tl.float32)
    acc += tl.where(m, v, 0.0)
    v = tl.load(go_ptr + out_base + w_l).to(tl.float32)
    acc += tl.where(m & m_l, v, 0.0)
    v = tl.load(go_ptr + out_base + w_r).to(tl.float32)
    acc += tl.where(m & m_r, v, 0.0)

    tl.store(gi_ptr + in_base + w_c, acc.to(gi_ptr.type.element_ty), mask=m)


def _reflection_pad2d_backward_impl(grad_output, self, padding):
    if hasattr(padding, "tolist"):
        padding = padding.tolist()
    if len(padding) != 4:
        raise ValueError("padding must be a sequence of 4 elements")
    pad_left, pad_right, pad_top, pad_bottom = (int(p) for p in padding)

    if self.dim() not in (3, 4):
        raise ValueError("input must be a 3D or 4D tensor")

    input_height, input_width = self.shape[-2:]
    output_height = input_height + pad_top + pad_bottom
    output_width = input_width + pad_left + pad_right
    if tuple(grad_output.shape[-2:]) != (output_height, output_width):
        raise ValueError(
            "grad_output spatial shape "
            f"{tuple(grad_output.shape[-2:])}, expected {(output_height, output_width)}"
        )
    # Match ATen: reflection padding requires pad < input size on each axis.
    if (
        pad_left >= input_width
        or pad_right >= input_width
        or pad_top >= input_height
        or pad_bottom >= input_height
    ):
        raise RuntimeError(
            "Padding size should be less than the corresponding input dimension, "
            f"but got padding ({pad_left}, {pad_right}, {pad_top}, {pad_bottom}) "
            f"of input {tuple(self.shape)}"
        )

    if not any((pad_left, pad_right, pad_top, pad_bottom)):
        return grad_output.clone()

    is_3d = self.dim() == 3
    go = grad_output.contiguous()
    if is_3d:
        go = go.unsqueeze(0)
    N, C, OH, OW = go.shape
    H, W = input_height, input_width

    gi = torch.empty((N, C, H, W), device=go.device, dtype=go.dtype)

    if gi.numel() != 0:
        NC = N * C
        HW = H * W
        OHW = OH * OW
        total = NC * HW
        with torch_device_fn.device(go.device):
            # 1) bulk copy: center term for every cell
            _reflection_pad2d_backward_bulk_kernel[(triton.cdiv(total, 1024),)](
                go,
                gi,
                H,
                W,
                OW,
                OHW,
                pad_top,
                pad_left,
                total,
                BLOCK=1024,
                num_warps=4,
            )

            # 2) reflection rows: E = [1, pt] U [H-1-pb, H-2] (dedup, sorted)
            all_rows = set(range(H))
            edge_rows = sorted(
                (set(range(1, pad_top + 1)) | set(range(H - 1 - pad_bottom, H - 1)))
                & all_rows
            )
            if edge_rows:
                rows_t = torch.tensor(edge_rows, dtype=torch.int32, device=go.device)
                n_edge = len(edge_rows)
                row_total = n_edge * NC * W
                _reflection_pad2d_backward_row_edge_kernel[
                    (triton.cdiv(row_total, 1024),)
                ](
                    go,
                    gi,
                    rows_t,
                    H,
                    W,
                    OH,
                    OW,
                    HW,
                    OHW,
                    pad_top,
                    pad_left,
                    pad_bottom,
                    pad_right,
                    n_edge,
                    row_total,
                    BLOCK=1024,
                    num_warps=4,
                )

                # 3) reflection columns on non-edge rows:
                #    C = [1, pl] U [W-1-pr, W-2] (dedup, sorted)
                edge_cols = sorted(
                    (set(range(1, pad_left + 1))
                     | set(range(W - 1 - pad_right, W - 1)))
                    & set(range(W))
                )
                plain_rows = sorted(all_rows - set(edge_rows))
                if edge_cols and plain_rows:
                    cols_t = torch.tensor(
                        edge_cols, dtype=torch.int32, device=go.device
                    )
                    plain_t = torch.tensor(
                        plain_rows, dtype=torch.int32, device=go.device
                    )
                    n_plain = len(plain_rows)
                    n_cols = len(edge_cols)
                    col_total = n_plain * NC * n_cols
                    _reflection_pad2d_backward_col_edge_kernel[
                        (triton.cdiv(col_total, 1024),)
                    ](
                        go,
                        gi,
                        plain_t,
                        cols_t,
                        H,
                        W,
                        OW,
                        HW,
                        OHW,
                        pad_top,
                        pad_left,
                        pad_right,
                        n_plain,
                        n_cols,
                        col_total,
                        BLOCK=1024,
                        num_warps=4,
                    )

    if is_3d:
        return gi.squeeze(0)
    return gi


def reflection_pad2d_backward(grad_output, self, padding):
    logger.debug("GEMS_KUNLUNXIN REFLECTION_PAD2D_BACKWARD")
    return _reflection_pad2d_backward_impl(grad_output, self, padding)
