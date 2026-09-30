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

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

from .batch_norm import _bn_fused_tile_s, _bn_train_tile_s, make_3d_for_bn

logger = logging.getLogger(__name__)

BNIIB_MAX_PROGRAMS = 4096


@libentry()
@triton.jit
def _bniib_reduce_kernel(
    grad_pointer,
    input_pointer,
    mean_pointer,
    inv_std_pointer,
    term1_pointer,
    term2_pointer,
    weight_grad_pointer,
    bias_grad_pointer,
    batch_dim,
    feat_dim,
    spatial_dim,
    WG_MASK: tl.constexpr,
    BG_MASK: tl.constexpr,
    TILE_S: tl.constexpr,
    NEED_MASK: tl.constexpr,
):
    c = tl.program_id(axis=0)
    mean = tl.load(mean_pointer + c).to(tl.float32)
    inv_std = tl.load(inv_std_pointer + c).to(tl.float32)
    t1 = tl.zeros([TILE_S], dtype=tl.float32)
    t2 = tl.zeros([TILE_S], dtype=tl.float32)
    for n in range(0, batch_dim):
        base = (n * feat_dim + c) * spatial_dim
        for off in range(0, spatial_dim, TILE_S):
            idx = off + tl.arange(0, TILE_S)
            if NEED_MASK:
                mask = idx < spatial_dim
                x = tl.load(input_pointer + base + idx, mask=mask, other=0.0).to(
                    tl.float32
                )
                dy = tl.load(grad_pointer + base + idx, mask=mask, other=0.0).to(
                    tl.float32
                )
                pre_lin = (x - mean) * inv_std
                t1 += tl.where(mask, pre_lin * dy, 0.0)
                t2 += tl.where(mask, dy, 0.0)
            else:
                x = tl.load(input_pointer + base + idx).to(tl.float32)
                dy = tl.load(grad_pointer + base + idx).to(tl.float32)
                pre_lin = (x - mean) * inv_std
                t1 += pre_lin * dy
                t2 += dy
    term1 = tl.sum(t1)
    term2 = tl.sum(t2)
    tl.store(term1_pointer + c, term1)
    tl.store(term2_pointer + c, term2)
    if WG_MASK:
        tl.store(weight_grad_pointer + c, term1.to(weight_grad_pointer.dtype.element_ty))
    if BG_MASK:
        tl.store(bias_grad_pointer + c, term2.to(bias_grad_pointer.dtype.element_ty))


@libentry()
@triton.jit
def _bniib_grad_kernel(
    grad_pointer,
    input_pointer,
    mean_pointer,
    inv_std_pointer,
    term1_pointer,
    term2_pointer,
    weight_pointer,
    input_grad_pointer,
    feat_dim,
    spatial_dim,
    count,
    slice_offset,
    IS_TRAIN: tl.constexpr,
    HAS_WEIGHT: tl.constexpr,
    TILE_S: tl.constexpr,
    NEED_MASK: tl.constexpr,
):
    local_pid = tl.program_id(axis=0)
    c = (slice_offset + local_pid) % feat_dim
    base = local_pid * spatial_dim
    inv_std = tl.load(inv_std_pointer + c).to(tl.float32)
    if HAS_WEIGHT:
        weight = tl.load(weight_pointer + c).to(tl.float32)
    else:
        weight = 1.0
    scale = inv_std * weight
    if IS_TRAIN:
        mean = tl.load(mean_pointer + c).to(tl.float32)
        term1 = tl.load(term1_pointer + c)
        term2 = tl.load(term2_pointer + c)
        rcp = 1.0 / count
    for off in range(0, spatial_dim, TILE_S):
        idx = off + tl.arange(0, TILE_S)
        if NEED_MASK:
            mask = idx < spatial_dim
            dy = tl.load(grad_pointer + base + idx, mask=mask).to(tl.float32)
            if IS_TRAIN:
                x = tl.load(input_pointer + base + idx, mask=mask).to(tl.float32)
                pre_lin = (x - mean) * inv_std
                g = scale * (dy - (term1 * pre_lin + term2) * rcp)
            else:
                g = dy * scale
            tl.store(
                input_grad_pointer + base + idx,
                g.to(input_grad_pointer.dtype.element_ty),
                mask=mask,
            )
        else:
            dy = tl.load(grad_pointer + base + idx).to(tl.float32)
            if IS_TRAIN:
                x = tl.load(input_pointer + base + idx).to(tl.float32)
                pre_lin = (x - mean) * inv_std
                g = scale * (dy - (term1 * pre_lin + term2) * rcp)
            else:
                g = dy * scale
            tl.store(
                input_grad_pointer + base + idx,
                g.to(input_grad_pointer.dtype.element_ty),
            )


def _batch_norm_impl_index_backward(
    impl_index: int,
    input: Tensor,
    grad_output: Tensor,
    weight=None,
    running_mean=None,
    running_var=None,
    save_mean=None,
    save_var_transform=None,
    train: bool = False,
    eps: float = 1e-5,
    output_mask=(True, True, True),
    reservedSpace=None,
):
    """Kunlunxin/XPU override of ``aten::_batch_norm_impl_index_backward``.

    The generic implementation uses a single 2D ``[BLOCK_M, BLOCK_N]`` tile
    kernel with a ``tl.sum`` reduction over a depth-2 tensor loop; that lowering
    fails on the XPU compiler (``TritonXPULegalize`` pass failure) for every
    non-tiny shape. Every training backend (native/cudnn/miopen) stores the
    inverse standard deviation in the ``save_var_transform`` slot, so a single
    native-style backward covers all ``impl_index`` values. This override uses a
    flat 1D grid with scalar reductions and masked addressing: a per-channel
    reduce kernel (``grid = feat_dim``) for the weight/bias gradients and the
    correction terms, and a per-(n, c)-slice kernel for the input gradient. In
    eval mode the input gradient degenerates to ``grad_output * weight *
    inv_std`` and the saved statistics are rebuilt from ``running_var``.
    """
    logger.debug("GEMS_KUNLUNXIN _BATCH_NORM_IMPL_INDEX_BACKWARD")

    input_3d = make_3d_for_bn(input)
    if not input_3d.is_contiguous():
        input_3d = input_3d.contiguous()
    grad_3d = make_3d_for_bn(grad_output)
    if not grad_3d.is_contiguous():
        grad_3d = grad_3d.contiguous()
    batch_dim, feat_dim, spatial_dim = input_3d.shape

    if save_mean is not None and save_mean.numel() != 0:
        mean = save_mean
    else:
        mean = running_mean

    if save_var_transform is not None and save_var_transform.numel() != 0:
        invstd = save_var_transform
    else:
        invstd = torch.rsqrt(running_var.to(torch.float32) + eps)

    # Empty input: emit the requested (zero-element) gradients to keep the
    # autograd graph intact, matching the aten fallback.
    if input.numel() == 0:
        grad_input = grad_weight = grad_bias = None
        if output_mask[0]:
            grad_input = torch.empty_like(input_3d).view_as(input)
        if output_mask[1]:
            grad_weight = torch.zeros(
                (feat_dim,), dtype=input.dtype, device=input.device
            )
        if output_mask[2]:
            grad_bias = torch.zeros((feat_dim,), dtype=input.dtype, device=input.device)
        return grad_input, grad_weight, grad_bias

    mean_f = mean.to(torch.float32)
    invstd_f = invstd.to(torch.float32)
    input_flat = input_3d.reshape(-1)
    grad_flat = grad_3d.reshape(-1)
    n_slices = batch_dim * feat_dim
    count = batch_dim * spatial_dim
    has_weight = weight is not None

    if output_mask[1]:
        weight_grad = torch.empty((feat_dim,), dtype=input.dtype, device=input.device)
    else:
        weight_grad = None
    if output_mask[2]:
        bias_grad = torch.empty((feat_dim,), dtype=input.dtype, device=input.device)
    else:
        bias_grad = None

    term1 = torch.empty(feat_dim, device=input.device, dtype=torch.float32)
    term2 = torch.empty(feat_dim, device=input.device, dtype=torch.float32)

    # The reduction is needed for weight/bias gradients and (in training) for the
    # input-gradient correction term.
    need_reduce = output_mask[1] or output_mask[2] or (train and output_mask[0])
    if need_reduce:
        rt_s, rt_mask = _bn_fused_tile_s(spatial_dim)
        with torch_device_fn.device(input.device):
            _bniib_reduce_kernel[(feat_dim,)](
                grad_flat,
                input_flat,
                mean_f,
                invstd_f,
                term1,
                term2,
                weight_grad if output_mask[1] else term1,
                bias_grad if output_mask[2] else term2,
                batch_dim,
                feat_dim,
                spatial_dim,
                WG_MASK=output_mask[1],
                BG_MASK=output_mask[2],
                TILE_S=rt_s,
                NEED_MASK=rt_mask,
                num_warps=4,
                buffer_size_limit=2048,
                isCloseVectorization=True,
            )

    input_grad = None
    if output_mask[0]:
        input_grad = torch.empty_like(input_3d)
        input_grad_flat = input_grad.reshape(-1)
        gt_s, gt_mask = _bn_train_tile_s(spatial_dim)
        with torch_device_fn.device(input.device):
            for slice_offset in range(0, n_slices, BNIIB_MAX_PROGRAMS):
                slice_count = min(BNIIB_MAX_PROGRAMS, n_slices - slice_offset)
                _bniib_grad_kernel[(slice_count,)](
                    grad_flat[slice_offset * spatial_dim :],
                    input_flat[slice_offset * spatial_dim :],
                    mean_f,
                    invstd_f,
                    term1,
                    term2,
                    weight if has_weight else grad_flat,
                    input_grad_flat[slice_offset * spatial_dim :],
                    feat_dim,
                    spatial_dim,
                    count,
                    slice_offset,
                    IS_TRAIN=bool(train),
                    HAS_WEIGHT=has_weight,
                    TILE_S=gt_s,
                    NEED_MASK=gt_mask,
                    num_warps=4,
                    buffer_size_limit=2048,
                    isCloseVectorization=True,
                )
        input_grad = input_grad.view_as(input)

    return input_grad, weight_grad, bias_grad
