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
from torch import Tensor

from flag_gems import runtime
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry, tl_extra_shim

from ._batch_norm_no_update import _batch_norm_no_update

logger = logging.getLogger(__name__)
rsqrt = tl_extra_shim.rsqrt


def make_3d_for_bn(input: Tensor) -> Tensor:
    if input.ndim == 2:
        input = input.unsqueeze(-1)
    elif input.ndim >= 4:
        input = input.flatten(2, -1)
    return input


# NOTE (kunlunxin / XPU forward rewrite):
# The generic batch_norm_forward_kernel uses grid=(feat_dim,) with a 2D [BLOCK_M,BLOCK_N]
# tile, and the previous kunlunxin wrapper worked around the XPU compiler's 2D-tile compile
# failure by transposing to [N*S, C, 1] (spatial_dim=1). That transpose forces stride-C
# discrete access AND only feat_dim(=C, often 8-16) parallel programs -> ~0.002 speedup.
#
# Since batch_norm reduces over batch*spatial PER channel, and in the natural [N, C, S]
# contiguous layout each (n, c) slice is S CONTIGUOUS elements, we instead map one program
# to each (n, c) slice (grid = N*C, like instance_norm). This gives full parallelism and
# fully contiguous block-DMA reads/writes with a clean 1D tile (compiles fine on XPU), and
# needs NO transpose copies. Stats are reduced per-(n,c) then combined across batch in the
# wrapper (a cheap [N,C]->[C] reduce). See harness/solution/batch_norm_forward_perf_fix.md.


# NOTE (kunlunxin / XPU inference rewrite, 2026-08-17):
# Inference (training=False) is now a single launch of the per-(n,c)-slice kernel
# from _batch_norm_no_update.py: each program loads its channel's stats/affine once
# and streams the CONTIGUOUS spatial run as block DMA (TILE_S=4096, masked tail).
# This replaces BOTH previous inference routes: (a) the fused transpose kernel
# (_batch_norm_fused_infer below is now only referenced by training-path comments)
# whose stride-C discrete reads cost ~6-12 ms on the small benchmark shapes, and
# (b) the 3-stage path's separate normalize kernel launch for large shapes.
# Measured: every benchmark case drops from ~0.19-12 ms to ~0.06-0.15 ms.


@libentry()
@triton.jit
def batch_norm_stats_kernel(
    input_pointer,  # [N*C, S] contiguous, flattened
    sum_pointer,  # [N*C] f32
    sqsum_pointer,  # [N*C] f32
    spatial_dim,
    slice_offset,
    TILE_S: tl.constexpr,
    NEED_MASK: tl.constexpr,
):
    pid = slice_offset + tl.program_id(axis=0)
    base = pid * spatial_dim
    s = tl.zeros([TILE_S], dtype=tl.float32)
    sq = tl.zeros([TILE_S], dtype=tl.float32)
    # XPU masked loads with runtime masks can leak whole lanes (see the
    # nll_loss / vector_norm fixes): clamp the index to a valid one and
    # zero invalid lanes with tl.where instead of relying on the mask.
    for off in range(0, spatial_dim, TILE_S):
        idx = off + tl.arange(0, TILE_S)
        mask = idx < spatial_dim
        idx_safe = tl.minimum(idx, spatial_dim - 1)
        x = tl.load(input_pointer + base + idx_safe).to(tl.float32)
        x = tl.where(mask, x, 0.0)
        s += x
        sq += x * x
    tl.store(sum_pointer + pid, tl.sum(s))
    tl.store(sqsum_pointer + pid, tl.sum(sq))


@libentry()
@triton.jit
def batch_norm_reduce_partials_kernel(
    part_sum_pointer,
    part_sqsum_pointer,
    reduced_sum_pointer,
    reduced_sqsum_pointer,
    batch_dim,
    feat_dim,
    TILE_N: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    channel = pid % feat_dim
    chunk = pid // feat_dim
    batch_offsets = chunk * TILE_N + tl.arange(0, TILE_N)
    mask = batch_offsets < batch_dim
    offsets = batch_offsets * feat_dim + channel
    part_sum = tl.load(part_sum_pointer + offsets, mask=mask, other=0.0)
    part_sqsum = tl.load(part_sqsum_pointer + offsets, mask=mask, other=0.0)
    part_sum = tl.where(mask, part_sum, 0.0)
    part_sqsum = tl.where(mask, part_sqsum, 0.0)
    tl.store(reduced_sum_pointer + pid, tl.sum(part_sum))
    tl.store(reduced_sqsum_pointer + pid, tl.sum(part_sqsum))


@libentry()
@triton.jit
def batch_norm_combine_kernel(
    part_sum_pointer,  # [N*C] f32, layout [n, c]
    part_sqsum_pointer,  # [N*C] f32, layout [n, c]
    mean_pointer,  # [C] f32 out
    inv_std_pointer,  # [C] f32 out
    running_mean_pointer,  # [C] or unused
    running_var_pointer,  # [C] or unused
    batch_dim,
    feat_dim,
    count,  # batch_dim * spatial_dim
    momentum,
    eps,
    var_correction,  # 1.0 (biased) or count / (count - 1) (unbiased)
    HAS_RM: tl.constexpr,
    HAS_RV: tl.constexpr,
    TILE_N: tl.constexpr,
):
    # One program per channel. Reduce the batch_dim partial (sum, sqsum) values for this
    # channel (strided by feat_dim), then compute mean / inv_std and fold the running-stat
    # updates in-kernel. Replaces ~14 small torch ops with a single launch -> cuts the
    # small-shape launch floor that regressed gems speedup.
    c = tl.program_id(axis=0)
    idx = tl.arange(0, TILE_N)
    mask = idx < batch_dim
    # Same XPU masked-load leak, and worse here: TILE_N is
    # next_pow2(batch_dim), so e.g. batch 2050 -> tile 4096 lets the
    # masked tail read up to ~2x past the partials buffer. Clamp to a
    # valid index and zero invalid lanes with tl.where.
    idx_safe = tl.minimum(idx, batch_dim - 1)
    part_sum = tl.load(part_sum_pointer + c + idx_safe * feat_dim)
    part_sqsum = tl.load(part_sqsum_pointer + c + idx_safe * feat_dim)
    part_sum = tl.where(mask, part_sum, 0.0)
    part_sqsum = tl.where(mask, part_sqsum, 0.0)
    ssum = tl.sum(part_sum)
    sqsum = tl.sum(part_sqsum)
    mean = ssum / count
    var = sqsum / count - mean * mean
    inv_std = rsqrt(var + eps)
    tl.store(mean_pointer + c, mean)
    tl.store(inv_std_pointer + c, inv_std)
    if HAS_RM:
        running_mean = tl.load(running_mean_pointer + c).to(tl.float32)
        tl.store(
            running_mean_pointer + c,
            ((1 - momentum) * running_mean + momentum * mean).to(
                running_mean_pointer.dtype.element_ty
            ),
        )
    if HAS_RV:
        running_var = tl.load(running_var_pointer + c).to(tl.float32)
        # Default (matching torch@XPU F.batch_norm, measured 2026-08-21): the
        # BIASED batch variance (var_correction == 1.0). The functional variant
        # (unbiased_running_var) folds the UNBIASED variance
        # (var * count / (count - 1)), matching the CPU aten reference.
        tl.store(
            running_var_pointer + c,
            ((1 - momentum) * running_var + momentum * var * var_correction).to(
                running_var_pointer.dtype.element_ty
            ),
        )


@libentry()
@triton.jit
def batch_norm_normalize_kernel(
    input_pointer,  # [N*C, S] contiguous, flattened
    output_pointer,
    mean_pointer,  # [C] f32
    inv_std_pointer,  # [C] f32
    weight_pointer,  # [C] or unused
    bias_pointer,  # [C] or unused
    feat_dim,
    spatial_dim,
    slice_offset,
    HAS_WEIGHT: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    TILE_S: tl.constexpr,
    NEED_MASK: tl.constexpr,
):
    pid = slice_offset + tl.program_id(axis=0)
    c = pid % feat_dim
    base = pid * spatial_dim

    mean = tl.load(mean_pointer + c)
    inv_std = tl.load(inv_std_pointer + c)
    if HAS_WEIGHT:
        weight = tl.load(weight_pointer + c).to(tl.float32)
    else:
        weight = 1.0
    if HAS_BIAS:
        bias = tl.load(bias_pointer + c).to(tl.float32)
    else:
        bias = 0.0

    for off in range(0, spatial_dim, TILE_S):
        idx = off + tl.arange(0, TILE_S)
        if NEED_MASK:
            mask = idx < spatial_dim
            x = tl.load(input_pointer + base + idx, mask=mask).to(tl.float32)
            y = weight * (x - mean) * inv_std + bias
            tl.store(
                output_pointer + base + idx,
                y.to(output_pointer.dtype.element_ty),
                mask=mask,
            )
        else:
            x = tl.load(input_pointer + base + idx).to(tl.float32)
            y = weight * (x - mean) * inv_std + bias
            tl.store(output_pointer + base + idx, y.to(output_pointer.dtype.element_ty))


# NOTE (hybrid routing): the contiguous grid=N*C 3-stage path above trades a per-shape
# ~0.4ms launch floor (stats kernel + torch combine + normalize kernel) for eliminating
# the large-spatial discrete-access catastrophe. For SMALL shapes that floor dominates and
# regresses gems speedup vs the original single fused kernel. So we keep the original fused
# (transpose) kernel below and route small shapes to it (see batch_norm wrapper). The fused
# kernel's stride-C discrete reads only blow up when batch_dim*spatial_dim is large.


# NOTE (kunlunxin / XPU small-shape fast path, 2026-08-21):
# For small per-channel counts (batch_dim * spatial_dim <= BN_FUSED_TRAIN_MAX_ELEMS) the
# 3-stage path pays a per-stage launch floor (~7-20us each: stats + combine + normalize +
# two dtype casts). The XPU compiler cannot lower 2D tiles or per-channel kernels with
# TILE >= 256 (uni_sram / TritonXPUUnrollControl failures), but a per-channel kernel with
# a fixed 128-lane tile compiles fine. Measured crossover (P800, 2026-08-21): fused route
# wins up to N*S <= 2048 ((1,8,4,4) 81.6->23.6us, (16,16,64) 144->84.6us, (16,16,128)
# 146->113.5us), loses beyond it ((16,16,256) 149->193us, (16,16,1024) 150->563us), so the
# 3-stage path is kept for the rest. The fused kernels also fold the running-stat updates
# (in-place, momentum semantics verified against torch@XPU, which uses the BIASED variance
# for the running_var update) and emit the returned save_mean/save_invstd in input dtype
# in-kernel (no extra cast launches).


@libentry()
@triton.jit
def batch_norm_fused_stats_kernel(
    input_pointer,  # [N*C, S] contiguous, flattened
    mean_pointer,  # [C] f32 out
    inv_std_pointer,  # [C] f32 out
    mean_d_pointer,  # [C] input-dtype out (returned save_mean)
    inv_std_d_pointer,  # [C] input-dtype out (returned save_invstd)
    running_mean_pointer,  # [C] or unused
    running_var_pointer,  # [C] or unused
    batch_dim,
    feat_dim,
    spatial_dim,
    momentum,
    eps,
    var_correction,  # 1.0 (biased) or count / (count - 1) (unbiased)
    HAS_RM: tl.constexpr,
    HAS_RV: tl.constexpr,
    TILE_S: tl.constexpr,
    NEED_MASK: tl.constexpr,
):
    c = tl.program_id(axis=0)
    count = batch_dim * spatial_dim
    s = tl.zeros([TILE_S], dtype=tl.float32)
    sq = tl.zeros([TILE_S], dtype=tl.float32)
    for n in range(0, batch_dim):
        base = (n * feat_dim + c) * spatial_dim
        for off in range(0, spatial_dim, TILE_S):
            idx = off + tl.arange(0, TILE_S)
            if NEED_MASK:
                mask = idx < spatial_dim
                x = tl.load(input_pointer + base + idx, mask=mask, other=0.0).to(
                    tl.float32
                )
                s += x
                sq += tl.where(mask, x * x, 0.0)
            else:
                x = tl.load(input_pointer + base + idx).to(tl.float32)
                s += x
                sq += x * x
    mean = tl.sum(s) / count
    var = tl.sum(sq) / count - mean * mean
    inv_std = rsqrt(var + eps)
    tl.store(mean_pointer + c, mean)
    tl.store(inv_std_pointer + c, inv_std)
    tl.store(mean_d_pointer + c, mean.to(mean_d_pointer.dtype.element_ty))
    tl.store(inv_std_d_pointer + c, inv_std.to(inv_std_d_pointer.dtype.element_ty))
    if HAS_RM:
        running_mean = tl.load(running_mean_pointer + c).to(tl.float32)
        tl.store(
            running_mean_pointer + c,
            ((1 - momentum) * running_mean + momentum * mean).to(
                running_mean_pointer.dtype.element_ty
            ),
        )
    if HAS_RV:
        running_var = tl.load(running_var_pointer + c).to(tl.float32)
        # Default (matching torch@XPU F.batch_norm, measured 2026-08-21): the
        # BIASED batch variance (var_correction == 1.0). The functional variant
        # (unbiased_running_var) folds the UNBIASED variance
        # (var * count / (count - 1)), matching the CPU aten reference.
        tl.store(
            running_var_pointer + c,
            ((1 - momentum) * running_var + momentum * var * var_correction).to(
                running_var_pointer.dtype.element_ty
            ),
        )


@libentry()
@triton.jit
def batch_norm_fused_normalize_kernel(
    input_pointer,  # [N*C, S] contiguous, flattened
    output_pointer,
    mean_pointer,  # [C] f32
    inv_std_pointer,  # [C] f32
    weight_pointer,  # [C] or unused
    bias_pointer,  # [C] or unused
    batch_dim,
    feat_dim,
    spatial_dim,
    HAS_WEIGHT: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    TILE_S: tl.constexpr,
    NEED_MASK: tl.constexpr,
):
    c = tl.program_id(axis=0)
    mean = tl.load(mean_pointer + c).to(tl.float32)
    inv_std = tl.load(inv_std_pointer + c).to(tl.float32)
    if HAS_WEIGHT:
        weight = tl.load(weight_pointer + c).to(tl.float32)
    else:
        weight = 1.0
    if HAS_BIAS:
        bias = tl.load(bias_pointer + c).to(tl.float32)
    else:
        bias = 0.0
    for n in range(0, batch_dim):
        base = (n * feat_dim + c) * spatial_dim
        for off in range(0, spatial_dim, TILE_S):
            idx = off + tl.arange(0, TILE_S)
            if NEED_MASK:
                mask = idx < spatial_dim
                x = tl.load(input_pointer + base + idx, mask=mask).to(tl.float32)
                y = weight * (x - mean) * inv_std + bias
                tl.store(
                    output_pointer + base + idx,
                    y.to(output_pointer.dtype.element_ty),
                    mask=mask,
                )
            else:
                x = tl.load(input_pointer + base + idx).to(tl.float32)
                y = weight * (x - mean) * inv_std + bias
                tl.store(
                    output_pointer + base + idx, y.to(output_pointer.dtype.element_ty)
                )


@libentry()
@triton.heuristics(runtime.get_heuristic_config("batch_norm"))
@triton.jit
def batch_norm_forward_kernel(
    input_pointer,
    weight_pointer,
    bias_pointer,
    mean_pointer,
    inv_std_pointer,
    output_pointer,
    running_mean_pointer,
    running_var_pointer,
    batch_dim,
    spatial_dim,
    input_batch_stride,
    input_feat_stride,
    input_spatial_stride,
    output_batch_stride,
    output_feat_stride,
    output_spatial_stride,
    momentum,
    eps,
    is_train: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    feat_pid = tl.program_id(axis=0)

    if is_train:
        total_sum = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        m_num_steps = tl.cdiv(batch_dim, BLOCK_M)
        n_num_steps = tl.cdiv(spatial_dim, BLOCK_N)

        for m_step in range(0, m_num_steps):
            for n_step in range(0, n_num_steps):
                spatial_offset = n_step * BLOCK_N + tl.arange(0, BLOCK_N)
                spatial_mask = spatial_offset < spatial_dim

                batch_offset = m_step * BLOCK_M + tl.arange(0, BLOCK_M)
                batch_mask = batch_offset < batch_dim

                curr_input_pointer = (
                    input_pointer
                    + input_feat_stride * feat_pid
                    + input_batch_stride * batch_offset[:, None]
                    + input_spatial_stride * spatial_offset[None, :]
                )

                mask = batch_mask[:, None] & spatial_mask[None, :]
                curr_input = tl.load(curr_input_pointer, mask=mask, other=0.0).to(
                    tl.float32
                )
                total_sum += curr_input

        n_elements = batch_dim * spatial_dim
        mean = tl.sum(total_sum) / n_elements

        var_sum = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        for m_step in range(0, m_num_steps):
            for n_step in range(0, n_num_steps):
                spatial_offset = n_step * BLOCK_N + tl.arange(0, BLOCK_N)
                spatial_mask = spatial_offset < spatial_dim

                batch_offset = m_step * BLOCK_M + tl.arange(0, BLOCK_M)
                batch_mask = batch_offset < batch_dim

                curr_input_pointer = (
                    input_pointer
                    + input_feat_stride * feat_pid
                    + input_batch_stride * batch_offset[:, None]
                    + input_spatial_stride * spatial_offset[None, :]
                )

                mask = batch_mask[:, None] & spatial_mask[None, :]
                curr_input = tl.load(curr_input_pointer, mask=mask, other=0.0).to(
                    tl.float32
                )
                diff = tl.where(mask, curr_input - mean, 0.0)
                var_sum += diff * diff

        var = tl.sum(var_sum) / n_elements
        inv_std = rsqrt(var + eps)

        tl.store(feat_pid + mean_pointer, mean)
        tl.store(feat_pid + inv_std_pointer, inv_std)

        running_mean_pointer += feat_pid
        running_var_pointer += feat_pid

        running_mean = tl.load(running_mean_pointer)
        running_var = tl.load(running_var_pointer)

        tl.store(running_mean_pointer, (1 - momentum) * running_mean + momentum * mean)
        tl.store(
            running_var_pointer,
            (1 - momentum) * running_var + momentum * var,
        )

    else:
        mean = tl.load(feat_pid + running_mean_pointer)
        inv_std = rsqrt(tl.load(feat_pid + running_var_pointer) + eps)

    if weight_pointer:
        weight = tl.load(feat_pid + weight_pointer).to(tl.float32)
    else:
        weight = 1.0
    if bias_pointer:
        bias = tl.load(feat_pid + bias_pointer).to(tl.float32)
    else:
        bias = 0.0

    for m_step in range(0, tl.cdiv(batch_dim, BLOCK_M)):
        for n_step in range(0, tl.cdiv(spatial_dim, BLOCK_N)):
            batch_offset = m_step * BLOCK_M + tl.arange(0, BLOCK_M)
            batch_mask = batch_offset < batch_dim

            spatial_offset = n_step * BLOCK_N + tl.arange(0, BLOCK_N)
            spatial_mask = spatial_offset < spatial_dim

            curr_input_pointer = (
                input_pointer
                + input_feat_stride * feat_pid
                + input_batch_stride * batch_offset[:, None]
                + input_spatial_stride * spatial_offset[None, :]
            )
            curr_output_pointer = (
                output_pointer
                + output_feat_stride * feat_pid
                + output_batch_stride * batch_offset[:, None]
                + output_spatial_stride * spatial_offset[None, :]
            )

            curr_input = tl.load(
                curr_input_pointer, mask=batch_mask[:, None] & spatial_mask[None, :]
            ).to(tl.float32)
            output = weight * (curr_input - mean) * inv_std + bias

            tl.store(
                curr_output_pointer,
                output,
                mask=batch_mask[:, None] & spatial_mask[None, :],
            )


def batch_norm(
    input: Tensor,
    weight=None,
    bias=None,
    running_mean=None,
    running_var=None,
    training=False,
    momentum=0.1,
    eps=1e-05,
    update_running_all_dtypes=False,
    unbiased_running_var=False,
):
    logger.debug("GEMS_KUNLUNXIN BATCH_NORM")

    # Always use the contiguous 3-stage path: the fused infer kernel's
    # permute(0, 2, 1).reshape forces a strided copy_ that fails to launch
    # on XPU ("CUDA error: invalid device function") for small eval shapes.
    x_nat = make_3d_for_bn(input)  # [N, C, S]
    batch_dim, _, spatial_dim = x_nat.shape
    input_3d = make_3d_for_bn(input).contiguous()  # [N, C, S] contiguous
    batch_dim, feat_dim, spatial_dim = input_3d.shape
    n_slices = batch_dim * feat_dim
    count = batch_dim * spatial_dim

    if count == 0:
        output = torch.empty_like(input_3d)
        mean = torch.empty(feat_dim, device=input.device, dtype=input.dtype)
        inv_std = torch.empty_like(mean)
        return output.view_as(input), mean, inv_std

    output = torch.empty_like(input_3d)
    input_flat = input_3d.reshape(-1)
    output_flat = output.reshape(-1)
    # NOTE (kunlunxin/xpu running-stat semantics, measured 2026-08-21 on-device):
    # torch@XPU F.batch_norm / aten::batch_norm in training mode updates running
    # stats IN-PLACE only for float32 inputs; for float16/bfloat16 inputs the native
    # implementation leaves running_mean/running_var untouched. The updates use the
    # BIASED batch variance (matches torch@XPU fp32 to ~1e-7).
    #
    # `_native_batch_norm_legit_functional` (update_running_all_dtypes /
    # unbiased_running_var) overrides both: the CPU aten reference updates the
    # running stats for EVERY float dtype and folds the UNBIASED batch variance
    # (var * count / (count - 1)) into running_var.
    has_rm = running_mean is not None and (
        update_running_all_dtypes or input.dtype == torch.float32
    )
    has_rv = running_var is not None and (
        update_running_all_dtypes or input.dtype == torch.float32
    )
    has_weight = weight is not None
    has_bias = bias is not None
    var_correction = (count / (count - 1)) if (unbiased_running_var and count > 1) else 1.0

    # Small-shape fast path: grid=C fused stats (per-channel reduction over all N*S,
    # in-kernel running-stat updates and input-dtype save stats) + grid=C fused
    # normalize; 2 launches total (see NOTE above). Below the crossover the 3-stage
    # path below keeps the contiguous per-(n,c)-slice kernels.
    if count <= BN_FUSED_TRAIN_MAX_ELEMS:
        fused_tile_s, fused_need_m = _bn_fused_tile_s(spatial_dim)
        mean_f = torch.empty(feat_dim, device=input.device, dtype=torch.float32)
        inv_std_f = torch.empty(feat_dim, device=input.device, dtype=torch.float32)
        mean_d = torch.empty(feat_dim, device=input.device, dtype=input.dtype)
        inv_std_d = torch.empty_like(mean_d)
        with torch_device_fn.device(input.device):
            batch_norm_fused_stats_kernel[(feat_dim,)](
                input_flat,
                mean_f,
                inv_std_f,
                mean_d,
                inv_std_d,
                running_mean if has_rm else mean_f,
                running_var if has_rv else mean_f,
                batch_dim,
                feat_dim,
                spatial_dim,
                momentum,
                eps,
                var_correction,
                HAS_RM=has_rm,
                HAS_RV=has_rv,
                TILE_S=fused_tile_s,
                NEED_MASK=fused_need_m,
                num_warps=4,
                buffer_size_limit=2048,
                isCloseVectorization=True,
            )
            batch_norm_fused_normalize_kernel[(feat_dim,)](
                input_flat,
                output_flat,
                mean_f,
                inv_std_f,
                weight if has_weight else input_flat,
                bias if has_bias else input_flat,
                batch_dim,
                feat_dim,
                spatial_dim,
                HAS_WEIGHT=has_weight,
                HAS_BIAS=has_bias,
                TILE_S=fused_tile_s,
                NEED_MASK=fused_need_m,
                num_warps=4,
                buffer_size_limit=2048,
                isCloseVectorization=True,
            )
        return output.view_as(input), mean_d, inv_std_d

    tile_s, need_mask = _bn_train_tile_s(spatial_dim)

    mean_f = torch.empty(feat_dim, device=input.device, dtype=torch.float32)
    inv_std_f = torch.empty(feat_dim, device=input.device, dtype=torch.float32)

    # Stage 1: per-(n, c) partial sum / sum-of-squares over contiguous spatial run.
    partial_batch_dim = triton.cdiv(batch_dim, 32) * 32 if batch_dim > 32 else batch_dim
    # Direct path (batch_dim <= 32): the stats kernel writes every one of the N*C
    # slots it reads back, so torch.empty is safe and avoids 2 zero-fill launches.
    # The chunked reduction path keeps torch.zeros: reduce_partials may read slots
    # beyond the rows written by the stats kernel.
    if batch_dim > 32:
        part_sum = torch.zeros(
            partial_batch_dim * feat_dim, device=input.device, dtype=torch.float32
        )
        part_sqsum = torch.zeros_like(part_sum)
    else:
        part_sum = torch.empty(
            partial_batch_dim * feat_dim, device=input.device, dtype=torch.float32
        )
        part_sqsum = torch.empty_like(part_sum)
    with torch_device_fn.device(input.device):
        max_programs = 4096
        for slice_offset in range(0, n_slices, max_programs):
            slice_count = min(max_programs, n_slices - slice_offset)
            batch_norm_stats_kernel[(slice_count,)](
                input_flat[slice_offset * spatial_dim :],
                part_sum[slice_offset:],
                part_sqsum[slice_offset:],
                spatial_dim,
                0,
                TILE_S=tile_s,
                NEED_MASK=need_mask,
            )
        combine_sum = part_sum
        combine_sqsum = part_sqsum
        combine_batch_dim = partial_batch_dim
        while combine_batch_dim > 32:
            reduced_batch_dim = triton.cdiv(combine_batch_dim, 32)
            if reduced_batch_dim > 32:
                storage_batch_dim = triton.cdiv(reduced_batch_dim, 32) * 32
            else:
                storage_batch_dim = triton.next_power_of_2(reduced_batch_dim)
            reduced_sum = torch.zeros(
                storage_batch_dim * feat_dim,
                device=input.device,
                dtype=torch.float32,
            )
            reduced_sqsum = torch.zeros_like(reduced_sum)
            batch_norm_reduce_partials_kernel[(reduced_batch_dim * feat_dim,)](
                combine_sum,
                combine_sqsum,
                reduced_sum,
                reduced_sqsum,
                combine_batch_dim,
                feat_dim,
                TILE_N=32,
                num_warps=4,
                buffer_size_limit=2048,
                isCloseVectorization=True,
            )
            combine_sum = reduced_sum
            combine_sqsum = reduced_sqsum
            combine_batch_dim = storage_batch_dim
        # Stage 2: combine batch partials -> per-channel mean / inv_std and fold the
        # running-stat updates, all in a single kernel (grid=(C,)). One launch instead
        # of ~14 small torch ops -> removes the small-shape launch floor.
        batch_norm_combine_kernel[(feat_dim,)](
            combine_sum,
            combine_sqsum,
            mean_f,
            inv_std_f,
            running_mean if has_rm else part_sum,
            running_var if has_rv else part_sum,
            combine_batch_dim,
            feat_dim,
            count,
            momentum,
            eps,
            var_correction,
            HAS_RM=has_rm,
            HAS_RV=has_rv,
            TILE_N=triton.next_power_of_2(combine_batch_dim),
            num_warps=4,
            buffer_size_limit=2048,
            isCloseVectorization=True,
        )

    # Return stats in input dtype (single cast each; no extra empty+copy).
    mean = mean_f.to(input.dtype)
    inv_std = inv_std_f.to(input.dtype)

    has_weight = weight is not None
    has_bias = bias is not None
    with torch_device_fn.device(input.device):
        max_programs = 4096
        for slice_offset in range(0, n_slices, max_programs):
            slice_count = min(max_programs, n_slices - slice_offset)
            batch_norm_normalize_kernel[(slice_count,)](
                input_flat[slice_offset * spatial_dim :],
                output_flat[slice_offset * spatial_dim :],
                mean_f,
                inv_std_f,
                weight if has_weight else input_flat,
                bias if has_bias else input_flat,
                feat_dim,
                spatial_dim,
                0,
                HAS_WEIGHT=has_weight,
                HAS_BIAS=has_bias,
                TILE_S=tile_s,
                NEED_MASK=need_mask,
            )

    return output.view_as(input), mean, inv_std


# Per-channel element-count (batch_dim * spatial_dim) at/below which the single-launch
# fused backward kernel (2 streaming passes, grid=(C,)) beats the 3-stage path (stats +
# combine + grad, 3 launches). Matches the forward's BN_FUSED_TRAIN_MAX_ELEMS crossover.
BNB_FUSED_MAX_ELEMS = 2048


def batch_norm_backward(
    grad_out,
    input,
    weight=None,
    running_mean=None,
    running_var=None,
    save_mean=None,
    save_invstd=None,
    train=False,
    eps=1e-05,
    output_mask=None,
):
    # The triton backward kernel below was only ever launched with
    # spatial_dim == 1: the old wrapper transposed inputs into [N*S, C, 1]
    # via permute(0, 2, 1).reshape, whose strided copy_ fails on XPU
    # ("invalid device function"), and feeding the kernel the natural
    # [N, C, S] layout (spatial > 1) trips an XPU compiler crash
    # (RuntimeError: PassManager::run failed). Compute the backward with
    # eager fp32 torch ops on the natural contiguous layout instead.
    logger.debug("GEMS_KUNLUNXIN BATCH_NORM_BACKWARD")
    if output_mask is None:
        output_mask = [True, True, True]
    input_3d = make_3d_for_bn(input).contiguous()
    grad_3d = make_3d_for_bn(grad_out).contiguous()
    n, c, sp = input_3d.shape
    count = n * sp
    # CPU aten returns EMPTY save tensors in eval mode; fall back to the
    # running stats whenever the saves are absent or empty.
    if save_mean is not None and save_mean.numel() > 0:
        mean = save_mean.to(torch.float32)
    else:
        mean = running_mean.to(torch.float32)
    if save_invstd is not None and save_invstd.numel() > 0:
        inv_std = save_invstd.to(torch.float32)
    else:
        inv_std = torch.rsqrt(running_var.to(torch.float32) + eps)
    x = input_3d.to(torch.float32)
    dy = grad_3d.to(torch.float32)
    xhat = (x - mean.view(1, c, 1)) * inv_std.view(1, c, 1)
    # flag_gems sum_dim dim_compress permutes non-last reduction dims onto
    # a strided layout whose copy_ fails to launch on XPU ("invalid device
    # function"). Reduce the contiguous last dim only (identity permute,
    # no copy), then fold the batch dim with a ones-vector GEMM.
    if n > 0:
        ones_n = torch.ones((1, n), dtype=torch.float32, device=input.device)
        sum_dy = torch.mm(ones_n, dy.sum(dim=2)).view(c)
        sum_dy_xhat = torch.mm(ones_n, (dy * xhat).sum(dim=2)).view(c)
    else:
        # zero-batch input: gems mm heuristic divides by K == 0
        sum_dy = torch.zeros(c, dtype=torch.float32, device=input.device)
        sum_dy_xhat = torch.zeros(c, dtype=torch.float32, device=input.device)
    if weight is not None:
        w = weight.to(torch.float32)
    else:
        w = torch.ones_like(mean)
    grads = []
    if output_mask[0]:
        if train:
            gi = (
                inv_std.view(1, c, 1)
                * w.view(1, c, 1)
                * (
                    dy
                    - sum_dy.view(1, c, 1) / count
                    - xhat * sum_dy_xhat.view(1, c, 1) / count
                )
            )
        else:
            # eval: running stats are constants, no centering correction
            gi = inv_std.view(1, c, 1) * w.view(1, c, 1) * dy
        grads.append(gi.to(input.dtype).view_as(input))
    else:
        grads.append(None)
    if output_mask[1]:
        if weight is not None:
            grads.append(sum_dy_xhat.to(input.dtype))
        else:
            grads.append(torch.zeros(c, dtype=input.dtype, device=input.device))
    else:
        grads.append(None)
    if output_mask[2]:
        grads.append(sum_dy.to(input.dtype))
    else:
        grads.append(None)
    return tuple(grads)


# ---------------------------------------------------------------------------
# _batch_norm_impl_index family (aten::batch_norm_impl_index and variants)
# ---------------------------------------------------------------------------

logger = logging.getLogger(__name__)


def batch_norm_impl_index(
    input: Tensor,
    weight=None,
    bias=None,
    running_mean=None,
    running_var=None,
    training=False,
    momentum=0.1,
    eps=1e-05,
    cudnn_enabled=True,
):
    # aten::batch_norm (functional) is a CompositeImplicitAutograd op that
    # internally redispatches to aten::_batch_norm_impl_index, so every
    # batch_norm-family entry point (F.batch_norm, native_batch_norm_legit,
    # instance_norm, ...) funnels here. The generic impl_index kernel's 2-D
    # strided pointer arithmetic fails XPU tt.addptr verification ("result
    # type matches ptr type", _batch_norm_impl_index.py:172) and surfaces as
    # OutOfResources/uni_sram on both training and eval paths. Delegate to
    # the kunlunxin 3-stage batch_norm implementation instead.
    logger.debug("GEMS_KUNLUNXIN BATCH_NORM_IMPL_INDEX")
    output, save_mean, save_invstd = batch_norm(
        input,
        weight,
        bias,
        running_mean,
        running_var,
        training,
        momentum,
        eps,
    )
    reserve = torch.empty((0,), dtype=torch.uint8, device=input.device)
    return output.view_as(input), save_mean, save_invstd, reserve, 0


def batch_norm_impl_index_backward(
    impl_index,
    input,
    grad_out,
    weight=None,
    running_mean=None,
    running_var=None,
    save_mean=None,
    save_var=None,
    train=False,
    eps=1e-05,
    output_mask=None,
    reservedSpace=None,
):
    # aten::_batch_norm_impl_index_backward(int impl_index, Tensor input,
    # Tensor grad_output, ... bool[3] output_mask, Tensor reservedSpace):
    # impl_index comes FIRST; keep the formal order in schema order.
    logger.debug("GEMS_KUNLUNXIN BATCH_NORM_IMPL_INDEX_BACKWARD")
    if output_mask is None:
        output_mask = [True, True, True]
    return batch_norm_backward(
        grad_out,
        input,
        weight,
        running_mean,
        running_var,
        save_mean,
        save_var,  # PyTorch's save_var is in fact inv_std
        train,
        eps,
        output_mask,
    )


def batch_norm_no_update(
    input,
    weight=None,
    bias=None,
    running_mean=None,
    running_var=None,
    momentum=0.1,
    eps=1e-05,
):
    # aten::_batch_norm_no_update(Tensor input, Tensor? weight, Tensor? bias,
    # Tensor? running_mean, Tensor? running_var, float momentum, float eps)
    # -> (Tensor, Tensor, Tensor, Tensor). Eval-only: normalize with the
    # running stats, never update them. CPU returns empty fp32 save tensors
    # and an empty uint8 reserve; the generic kernel trips the XPU compiler
    # (RuntimeError: PassManager::run failed) on every shape.
    logger.debug("GEMS_KUNLUNXIN BATCH_NORM_NO_UPDATE")
    output, _, _ = batch_norm(
        input, weight, bias, running_mean, running_var, False, momentum, eps
    )
    empty_f32 = torch.empty((0,), dtype=torch.float32, device=input.device)
    empty_u8 = torch.empty((0,), dtype=torch.uint8, device=input.device)
    return output.view_as(input), empty_f32, empty_f32, empty_u8


def batch_norm_with_update_functional(
    input,
    weight=None,
    bias=None,
    running_mean=None,
    running_var=None,
    momentum=0.1,
    eps=1e-05,
):
    # aten::_batch_norm_with_update_functional is called directly by
    # torch.ops.aten; the generic kernel shares the impl_index addptr
    # failure. Delegate to the kunlunxin 3-stage path, whose training
    # branch folds the running-stat updates in place (combine kernel),
    # so the mutated running_mean/running_var tensors are the new stats.
    logger.debug("GEMS_KUNLUNXIN BATCH_NORM_WITH_UPDATE_FUNCTIONAL")
    output, save_mean, save_invstd = batch_norm(
        input,
        weight,
        bias,
        running_mean,
        running_var,
        True,
        momentum,
        eps,
    )
    # aten schema returns SIX tensors: (output, save_mean, save_var,
    # reserve, running_mean_out, running_var_out).
    reserve = torch.empty((0,), dtype=torch.uint8, device=input.device)
    return (
        output.view_as(input),
        save_mean,
        save_invstd,
        reserve,
        running_mean,
        running_var,
    )


def miopen_batch_norm_backward(
    input,
    grad_output,
    weight,
    running_mean,
    running_var,
    save_mean,
    save_var,
    epsilon,
):
    # aten::miopen_batch_norm_backward mirrors the cudnn variant. Its
    # save_var argument holds the saved inverse standard deviation (rstd),
    # which is exactly what batch_norm_backward expects as save_invstd.
    logger.debug("GEMS_KUNLUNXIN MIOPEN_BATCH_NORM_BACKWARD")
    input_grad, weight_grad, bias_grad = batch_norm_backward(
        grad_output,
        input,
        weight,
        running_mean,
        running_var,
        save_mean,
        save_var,
        True,
        epsilon,
        [True, True, True],
    )
    return input_grad, weight_grad, bias_grad


# ---------------------------------------------------------------------------
# _native_batch_norm_legit family
# ---------------------------------------------------------------------------

logger = logging.getLogger(__name__)


def _native_batch_norm_legit(
    input: Tensor,
    weight,
    bias,
    running_mean: Tensor,
    running_var: Tensor,
    training: bool,
    momentum: float,
    eps: float,
):
    # aten::_native_batch_norm_legit mutates running stats in place and
    # returns (output, save_mean, save_invstd). The kunlunxin batch_norm
    # training branch folds the running-stat update into the combine
    # kernel, so the in-place mutation is covered.
    logger.debug("GEMS_KUNLUNXIN NATIVE_BATCH_NORM_LEGIT")
    output, save_mean, save_invstd = batch_norm(
        input, weight, bias, running_mean, running_var, training, momentum, eps
    )
    if not training:
        # aten CPU returns EMPTY save tensors in eval mode.
        empty = torch.empty(0, dtype=torch.float32, device=input.device)
        return output, empty, empty
    return output, save_mean, save_invstd


def _native_batch_norm_legit_no_stats(
    input: Tensor,
    weight,
    bias,
    training: bool,
    momentum: float,
    eps: float,
):
    logger.debug("GEMS_KUNLUNXIN NATIVE_BATCH_NORM_LEGIT_NO_STATS")
    if not training:
        raise RuntimeError("Expected training to be true, but got false.")
    channels = input.shape[1]
    running_mean = torch.zeros(channels, dtype=input.dtype, device=input.device)
    running_var = torch.ones(channels, dtype=input.dtype, device=input.device)
    return batch_norm(
        input, weight, bias, running_mean, running_var, True, momentum, eps
    )


def _copy_outputs(result, out, save_mean, save_invstd):
    result_out, result_mean, result_invstd = result
    out.resize_as_(result_out).copy_(result_out)
    save_mean.resize_as_(result_mean).copy_(result_mean)
    save_invstd.resize_as_(result_invstd).copy_(result_invstd)
    return out, save_mean, save_invstd


def _native_batch_norm_legit_out(
    input: Tensor,
    weight,
    bias,
    running_mean: Tensor,
    running_var: Tensor,
    training: bool,
    momentum: float,
    eps: float,
    *,
    out: Tensor,
    save_mean: Tensor,
    save_invstd: Tensor,
):
    logger.debug("GEMS_KUNLUNXIN NATIVE_BATCH_NORM_LEGIT_OUT")
    result = _native_batch_norm_legit(
        input, weight, bias, running_mean, running_var, training, momentum, eps
    )
    return _copy_outputs(result, out, save_mean, save_invstd)


def _native_batch_norm_legit_no_stats_out(
    input: Tensor,
    weight,
    bias,
    training: bool,
    momentum: float,
    eps: float,
    *,
    out: Tensor,
    save_mean: Tensor,
    save_invstd: Tensor,
):
    logger.debug("GEMS_KUNLUNXIN NATIVE_BATCH_NORM_LEGIT_NO_STATS_OUT")
    result = _native_batch_norm_legit_no_stats(
        input, weight, bias, training, momentum, eps
    )
    return _copy_outputs(result, out, save_mean, save_invstd)


def _native_batch_norm_legit_functional(
    input: Tensor,
    weight,
    bias,
    running_mean: Tensor,
    running_var: Tensor,
    training: bool,
    momentum: float,
    eps: float,
):
    # Functional variant: the caller's running stats must NOT be mutated;
    # updated stats are returned as new tensors. Clone before delegating
    # so the in-place update inside batch_norm lands on the clones.
    logger.debug("GEMS_KUNLUNXIN NATIVE_BATCH_NORM_LEGIT_FUNCTIONAL")
    rm = running_mean.clone()
    rv = running_var.clone()
    output, save_mean, save_invstd = batch_norm(
        input, weight, bias, rm, rv, training, momentum, eps
    )
    return output, save_mean, save_invstd, rm, rv


def _native_batch_norm_legit_no_training(
    input: Tensor,
    weight,
    bias,
    running_mean: Tensor,
    running_var: Tensor,
    momentum: float,
    eps: float,
):
    # Eval-only variant: normalize with the running stats, no update.
    logger.debug("GEMS_KUNLUNXIN NATIVE_BATCH_NORM_LEGIT_NO_TRAINING")
    output, save_mean, save_invstd = batch_norm(
        input, weight, bias, running_mean, running_var, False, momentum, eps
    )
    return output, save_mean, save_invstd
