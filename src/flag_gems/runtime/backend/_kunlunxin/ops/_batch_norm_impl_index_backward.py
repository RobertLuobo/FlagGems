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
from torch import Tensor

from flag_gems.runtime import torch_device_fn

from .batch_norm import (
    _bn_train_tile_s,
    batch_norm_backward,
    batch_norm_normalize_kernel,
    make_3d_for_bn,
)

logger = logging.getLogger(__name__)


def _eval_input_grad(grad_3d, weight, inv_std, feat_dim, spatial_dim):
    """Eval-mode input gradient ``grad_output * weight * inv_std`` per channel.

    Reuses the proven ``batch_norm_normalize_kernel`` with ``mean = 0`` and
    ``bias = None`` so that ``y = weight * (grad - 0) * inv_std`` yields exactly
    the degenerate eval-mode input gradient.
    """
    batch_dim = grad_3d.shape[0]
    n_slices = batch_dim * feat_dim
    input_grad = torch.empty_like(grad_3d)
    grad_flat = grad_3d.reshape(-1)
    input_grad_flat = input_grad.reshape(-1)
    has_weight = weight is not None
    mean_zero = torch.zeros(feat_dim, device=grad_3d.device, dtype=torch.float32)
    inv_std_f = inv_std.to(torch.float32)
    tile_s, need_mask = _bn_train_tile_s(spatial_dim)
    max_programs = 4096
    with torch_device_fn.device(grad_3d.device):
        for slice_offset in range(0, n_slices, max_programs):
            slice_count = min(max_programs, n_slices - slice_offset)
            batch_norm_normalize_kernel[(slice_count,)](
                grad_flat,
                input_grad_flat,
                mean_zero,
                inv_std_f,
                weight if has_weight else grad_flat,
                grad_flat,
                feat_dim,
                spatial_dim,
                slice_offset,
                HAS_WEIGHT=has_weight,
                HAS_BIAS=False,
                TILE_S=tile_s,
                NEED_MASK=need_mask,
            )
    return input_grad


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
    """Backward dispatcher for ``_batch_norm_impl_index`` on XPU.

    Every training backend (native / cudnn / miopen) stores the inverse
    standard deviation in the ``save_var_transform`` slot, so a single
    native-style backward covers all ``impl_index`` values.  In training mode
    this delegates to the vendor ``batch_norm_backward`` (trailing-axis
    single-program scan, which lowers on XPU3).  In eval mode the saved
    statistics are empty: the inverse std is rebuilt from ``running_var`` and
    the input gradient degenerates to ``grad_output * weight * inv_std``.
    """
    logger.debug("GEMS_KUNLUNXIN _BATCH_NORM_IMPL_INDEX_BACKWARD")

    input_3d = make_3d_for_bn(input)
    grad_output_3d = make_3d_for_bn(grad_output)
    batch_dim, feat_dim, spatial_dim = input_3d.shape

    if save_mean is not None and save_mean.numel() != 0:
        mean = save_mean
    else:
        mean = running_mean

    if save_var_transform is not None and save_var_transform.numel() != 0:
        invstd = save_var_transform
    else:
        invstd = torch.rsqrt(running_var.to(torch.float32) + eps).to(input.dtype)

    if input.numel() == 0:
        grad_input = None
        grad_weight = None
        grad_bias = None
        if output_mask[0]:
            grad_input = torch.empty_like(input_3d)
        if output_mask[1]:
            grad_weight = torch.zeros(
                (feat_dim,), dtype=input.dtype, device=input.device
            )
        if output_mask[2]:
            grad_bias = torch.zeros(
                (feat_dim,), dtype=input.dtype, device=input.device
            )
        return (
            grad_input.view_as(input) if grad_input is not None else None,
            grad_weight,
            grad_bias,
        )

    if train:
        return batch_norm_backward(
            grad_output,
            input,
            weight,
            running_mean,
            running_var,
            mean,
            invstd,
            train=True,
            eps=eps,
            output_mask=tuple(bool(m) for m in output_mask),
        )

    weight_grad = None
    bias_grad = None
    if output_mask[1] or output_mask[2]:
        _, weight_grad, bias_grad = batch_norm_backward(
            grad_output,
            input,
            weight,
            running_mean,
            running_var,
            mean,
            invstd,
            train=True,
            eps=eps,
            output_mask=(True, bool(output_mask[1]), bool(output_mask[2])),
        )

    input_grad = None
    if output_mask[0]:
        grad_3d = grad_output_3d
        if not grad_3d.is_contiguous():
            grad_3d = grad_3d.contiguous()
        input_grad = _eval_input_grad(
            grad_3d, weight, invstd, feat_dim, spatial_dim
        ).view_as(input)

    return (
        input_grad if output_mask[0] else None,
        weight_grad if output_mask[1] else None,
        bias_grad if output_mask[2] else None,
    )
