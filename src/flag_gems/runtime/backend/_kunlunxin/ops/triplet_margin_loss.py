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

from flag_gems import runtime
from flag_gems.ops.mean import mean as gems_mean
from flag_gems.ops.sum import sum as gems_sum
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry, libtuner, tl_extra_shim

exp2 = tl_extra_shim.exp2
log2 = tl_extra_shim.log2
logger = logging.getLogger(__name__)


TRIPLET_MARGIN_LOSS_CONFIGS = runtime.get_tuned_config("triplet_margin_loss") or [
    triton.Config({"BLOCK_D": 256}, num_warps=1),
    triton.Config({"BLOCK_D": 512}, num_warps=1),
    triton.Config({"BLOCK_D": 1024}, num_warps=1),
]


# On XPU3 a 2-D tile [BLOCK_M, BLOCK_D] reduced along the middle feature axis
# (tl.sum/tl.max/tl.min with axis=1) crashes TritonXPUCoreTiling. Each kernel
# below instead maps ONE row per program and reduces the trailing feature axis
# as a 1-D reduction (axis=0), which legalizes cleanly.


@libentry()
@libtuner(configs=TRIPLET_MARGIN_LOSS_CONFIGS, key=["D"])
@triton.jit
def triplet_margin_loss_p1_kernel(
    anchor_ptr,
    positive_ptr,
    negative_ptr,
    out_ptr,
    N,
    D,
    eps,
    margin,
    SWAP: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    acc_dtype = (
        tl.float64 if anchor_ptr.type.element_ty == tl.float64 else tl.float32
    )
    row = tl.program_id(0)
    base = row * D

    acc_ap = tl.zeros([BLOCK_D], dtype=acc_dtype)
    acc_an = tl.zeros([BLOCK_D], dtype=acc_dtype)
    acc_pn = tl.zeros([BLOCK_D], dtype=acc_dtype)

    for start in range(0, D, BLOCK_D):
        cols = start + tl.arange(0, BLOCK_D)
        mask = cols < D
        a = tl.load(anchor_ptr + base + cols, mask=mask, other=0).to(acc_dtype)
        pos = tl.load(positive_ptr + base + cols, mask=mask, other=0).to(acc_dtype)
        neg = tl.load(negative_ptr + base + cols, mask=mask, other=0).to(acc_dtype)

        diff_ap = tl.abs(a - pos + eps)
        diff_an = tl.abs(a - neg + eps)
        acc_ap += tl.where(mask, diff_ap, 0.0)
        acc_an += tl.where(mask, diff_an, 0.0)
        if SWAP:
            diff_pn = tl.abs(pos - neg + eps)
            acc_pn += tl.where(mask, diff_pn, 0.0)

    dist_ap = tl.sum(acc_ap, axis=0)
    dist_an = tl.sum(acc_an, axis=0)
    if SWAP:
        dist_pn = tl.sum(acc_pn, axis=0)
        dist_an = tl.minimum(dist_an, dist_pn)

    loss = dist_ap - dist_an + margin
    loss = tl.maximum(loss, 0.0)
    tl.store(out_ptr + row, loss)


@libentry()
@libtuner(configs=TRIPLET_MARGIN_LOSS_CONFIGS, key=["D"])
@triton.jit
def triplet_margin_loss_p2_kernel(
    anchor_ptr,
    positive_ptr,
    negative_ptr,
    out_ptr,
    N,
    D,
    eps,
    margin,
    SWAP: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    acc_dtype = (
        tl.float64 if anchor_ptr.type.element_ty == tl.float64 else tl.float32
    )
    row = tl.program_id(0)
    base = row * D

    acc_ap = tl.zeros([BLOCK_D], dtype=acc_dtype)
    acc_an = tl.zeros([BLOCK_D], dtype=acc_dtype)
    acc_pn = tl.zeros([BLOCK_D], dtype=acc_dtype)

    for start in range(0, D, BLOCK_D):
        cols = start + tl.arange(0, BLOCK_D)
        mask = cols < D
        a = tl.load(anchor_ptr + base + cols, mask=mask, other=0).to(acc_dtype)
        pos = tl.load(positive_ptr + base + cols, mask=mask, other=0).to(acc_dtype)
        neg = tl.load(negative_ptr + base + cols, mask=mask, other=0).to(acc_dtype)

        diff_ap = tl.abs(a - pos + eps)
        diff_an = tl.abs(a - neg + eps)
        acc_ap += tl.where(mask, diff_ap * diff_ap, 0.0)
        acc_an += tl.where(mask, diff_an * diff_an, 0.0)
        if SWAP:
            diff_pn = tl.abs(pos - neg + eps)
            acc_pn += tl.where(mask, diff_pn * diff_pn, 0.0)

    dist_ap = tl.sqrt(tl.sum(acc_ap, axis=0))
    dist_an = tl.sqrt(tl.sum(acc_an, axis=0))
    if SWAP:
        dist_pn = tl.sqrt(tl.sum(acc_pn, axis=0))
        dist_an = tl.minimum(dist_an, dist_pn)

    loss = dist_ap - dist_an + margin
    loss = tl.maximum(loss, 0.0)
    tl.store(out_ptr + row, loss)


@libentry()
@libtuner(configs=TRIPLET_MARGIN_LOSS_CONFIGS, key=["D"])
@triton.jit
def triplet_margin_loss_general_kernel(
    anchor_ptr,
    positive_ptr,
    negative_ptr,
    out_ptr,
    N,
    D,
    eps,
    p,
    margin,
    SWAP: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    acc_dtype = (
        tl.float64 if anchor_ptr.type.element_ty == tl.float64 else tl.float32
    )
    p = p.to(acc_dtype)
    row = tl.program_id(0)
    base = row * D

    acc_ap = tl.zeros([BLOCK_D], dtype=acc_dtype)
    acc_an = tl.zeros([BLOCK_D], dtype=acc_dtype)
    acc_pn = tl.zeros([BLOCK_D], dtype=acc_dtype)

    for start in range(0, D, BLOCK_D):
        cols = start + tl.arange(0, BLOCK_D)
        mask = cols < D
        a = tl.load(anchor_ptr + base + cols, mask=mask, other=0).to(acc_dtype)
        pos = tl.load(positive_ptr + base + cols, mask=mask, other=0).to(acc_dtype)
        neg = tl.load(negative_ptr + base + cols, mask=mask, other=0).to(acc_dtype)

        diff_ap = tl.abs(a - pos + eps)
        diff_an = tl.abs(a - neg + eps)
        acc_ap += tl.where(mask, exp2(p * log2(diff_ap)), 0.0)
        acc_an += tl.where(mask, exp2(p * log2(diff_an)), 0.0)
        if SWAP:
            diff_pn = tl.abs(pos - neg + eps)
            acc_pn += tl.where(mask, exp2(p * log2(diff_pn)), 0.0)

    inv_p = 1.0 / p
    dist_ap = exp2(inv_p * log2(tl.sum(acc_ap, axis=0)))
    dist_an = exp2(inv_p * log2(tl.sum(acc_an, axis=0)))
    if SWAP:
        dist_pn = exp2(inv_p * log2(tl.sum(acc_pn, axis=0)))
        dist_an = tl.minimum(dist_an, dist_pn)

    loss = dist_ap - dist_an + margin
    loss = tl.maximum(loss, 0.0)
    tl.store(out_ptr + row, loss)


@libentry()
@libtuner(configs=TRIPLET_MARGIN_LOSS_CONFIGS, key=["D"])
@triton.jit
def triplet_margin_loss_p0_kernel(
    anchor_ptr,
    positive_ptr,
    negative_ptr,
    out_ptr,
    N,
    D,
    eps,
    margin,
    SWAP: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    acc_dtype = (
        tl.float64 if anchor_ptr.type.element_ty == tl.float64 else tl.float32
    )
    row = tl.program_id(0)
    base = row * D

    acc_ap = tl.zeros([BLOCK_D], dtype=acc_dtype)
    acc_an = tl.zeros([BLOCK_D], dtype=acc_dtype)
    acc_pn = tl.zeros([BLOCK_D], dtype=acc_dtype)

    for start in range(0, D, BLOCK_D):
        cols = start + tl.arange(0, BLOCK_D)
        mask = cols < D
        a = tl.load(anchor_ptr + base + cols, mask=mask, other=0).to(acc_dtype)
        pos = tl.load(positive_ptr + base + cols, mask=mask, other=0).to(acc_dtype)
        neg = tl.load(negative_ptr + base + cols, mask=mask, other=0).to(acc_dtype)

        nz_ap = (a - pos + eps) != 0.0
        nz_an = (a - neg + eps) != 0.0
        acc_ap += tl.where(mask & nz_ap, 1.0, 0.0)
        acc_an += tl.where(mask & nz_an, 1.0, 0.0)
        if SWAP:
            nz_pn = (pos - neg + eps) != 0.0
            acc_pn += tl.where(mask & nz_pn, 1.0, 0.0)

    dist_ap = tl.sum(acc_ap, axis=0)
    dist_an = tl.sum(acc_an, axis=0)
    if SWAP:
        dist_pn = tl.sum(acc_pn, axis=0)
        dist_an = tl.minimum(dist_an, dist_pn)

    loss = dist_ap - dist_an + margin
    loss = tl.maximum(loss, 0.0)
    tl.store(out_ptr + row, loss)


@libentry()
@libtuner(configs=TRIPLET_MARGIN_LOSS_CONFIGS, key=["D"])
@triton.jit
def triplet_margin_loss_inf_kernel(
    anchor_ptr,
    positive_ptr,
    negative_ptr,
    out_ptr,
    N,
    D,
    eps,
    margin,
    SWAP: tl.constexpr,
    IS_MAX: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    acc_dtype = (
        tl.float64 if anchor_ptr.type.element_ty == tl.float64 else tl.float32
    )
    row = tl.program_id(0)
    base = row * D

    if IS_MAX:
        init = 0.0
    else:
        init = float("inf")
    acc_ap = tl.full([BLOCK_D], init, dtype=acc_dtype)
    acc_an = tl.full([BLOCK_D], init, dtype=acc_dtype)
    acc_pn = tl.full([BLOCK_D], init, dtype=acc_dtype)

    for start in range(0, D, BLOCK_D):
        cols = start + tl.arange(0, BLOCK_D)
        mask = cols < D
        a = tl.load(anchor_ptr + base + cols, mask=mask, other=0).to(acc_dtype)
        pos = tl.load(positive_ptr + base + cols, mask=mask, other=0).to(acc_dtype)
        neg = tl.load(negative_ptr + base + cols, mask=mask, other=0).to(acc_dtype)

        diff_ap = tl.abs(a - pos + eps)
        diff_an = tl.abs(a - neg + eps)
        if IS_MAX:
            acc_ap = tl.maximum(acc_ap, tl.where(mask, diff_ap, init))
            acc_an = tl.maximum(acc_an, tl.where(mask, diff_an, init))
        else:
            acc_ap = tl.minimum(acc_ap, tl.where(mask, diff_ap, init))
            acc_an = tl.minimum(acc_an, tl.where(mask, diff_an, init))
        if SWAP:
            diff_pn = tl.abs(pos - neg + eps)
            if IS_MAX:
                acc_pn = tl.maximum(acc_pn, tl.where(mask, diff_pn, init))
            else:
                acc_pn = tl.minimum(acc_pn, tl.where(mask, diff_pn, init))

    if IS_MAX:
        dist_ap = tl.max(acc_ap, axis=0)
        dist_an = tl.max(acc_an, axis=0)
    else:
        dist_ap = tl.min(acc_ap, axis=0)
        dist_an = tl.min(acc_an, axis=0)
    if SWAP:
        if IS_MAX:
            dist_pn = tl.max(acc_pn, axis=0)
        else:
            dist_pn = tl.min(acc_pn, axis=0)
        dist_an = tl.minimum(dist_an, dist_pn)

    loss = dist_ap - dist_an + margin
    loss = tl.maximum(loss, 0.0)
    tl.store(out_ptr + row, loss)


def triplet_margin_loss(
    anchor,
    positive,
    negative,
    margin=1.0,
    p=2.0,
    eps=1e-6,
    swap=False,
    reduction="mean",
):
    logger.debug("GEMS_KUNLUNXIN TRIPLET_MARGIN_LOSS")

    if isinstance(reduction, int):
        reduction = ["none", "mean", "sum"][reduction]

    out_dtype = torch.promote_types(
        torch.promote_types(anchor.dtype, positive.dtype), negative.dtype
    )
    anchor, positive, negative = torch.broadcast_tensors(anchor, positive, negative)
    anchor = anchor.to(out_dtype).contiguous()
    positive = positive.to(out_dtype).contiguous()
    negative = negative.to(out_dtype).contiguous()

    out_shape = anchor.shape[:-1]
    D = anchor.shape[-1] if anchor.ndim > 0 else 1
    N = anchor.numel() // D if D > 0 else anchor.numel()

    out = torch.empty(out_shape, dtype=out_dtype, device=anchor.device)

    if D == 0:
        out.fill_(max(float(margin), 0.0))
    elif N > 0:
        # One program per row; grid is a plain constant tuple (depends only on
        # N) so there is no per-call recompilation.
        grid = (N,)

        with torch_device_fn.device(anchor.device):
            if p == 1.0:
                triplet_margin_loss_p1_kernel[grid](
                    anchor, positive, negative, out, N, D, eps, margin, SWAP=swap
                )
            elif p == 2.0:
                triplet_margin_loss_p2_kernel[grid](
                    anchor, positive, negative, out, N, D, eps, margin, SWAP=swap
                )
            elif p == 0.0:
                triplet_margin_loss_p0_kernel[grid](
                    anchor, positive, negative, out, N, D, eps, margin, SWAP=swap
                )
            elif p == float("inf"):
                triplet_margin_loss_inf_kernel[grid](
                    anchor,
                    positive,
                    negative,
                    out,
                    N,
                    D,
                    eps,
                    margin,
                    SWAP=swap,
                    IS_MAX=True,
                )
            elif p == float("-inf"):
                triplet_margin_loss_inf_kernel[grid](
                    anchor,
                    positive,
                    negative,
                    out,
                    N,
                    D,
                    eps,
                    margin,
                    SWAP=swap,
                    IS_MAX=False,
                )
            else:
                triplet_margin_loss_general_kernel[grid](
                    anchor, positive, negative, out, N, D, eps, p, margin, SWAP=swap
                )

    if reduction == "mean":
        return gems_mean(out)
    elif reduction == "sum":
        return gems_sum(out)
    return out


