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

from .native_batch_norm import make_3d_for_bn, native_batch_norm

logger = logging.getLogger(__name__)


def _batch_norm_with_update_functional(
    input: Tensor,
    weight=None,
    bias=None,
    running_mean=None,
    running_var=None,
    momentum=0.1,
    eps=1e-05,
):
    logger.debug("GEMS_KUNLUNXIN _BATCH_NORM_WITH_UPDATE_FUNCTIONAL")
    logger.debug("GEMS _BATCH_NORM_WITH_UPDATE_FUNCTIONAL")

    input_3d = make_3d_for_bn(input)
    batch_dim, feat_dim, spatial_dim = input_3d.shape

    if running_mean is None or running_var is None:
        raise RuntimeError(
            "_batch_norm_with_update_functional expects running_mean and "
            "running_var to be specified."
        )

    for name, param in (
        ("weight", weight),
        ("bias", bias),
        ("running_mean", running_mean),
        ("running_var", running_var),
    ):
        if param is not None and param.numel() != feat_dim:
            raise RuntimeError(
                f"_batch_norm_with_update_functional expects {name} to have "
                f"{feat_dim} elements, but got {param.numel()}."
            )

    reserve = torch.empty((0,), dtype=torch.uint8, device=input.device)

    # Functional variant: return NEW running-stat tensors rather than mutating
    # the inputs in place, so start from a clone that native_batch_norm updates.
    new_running_mean = running_mean.clone().contiguous()
    new_running_var = running_var.clone().contiguous()

    if batch_dim == 0 or spatial_dim == 0:
        mean = torch.zeros(feat_dim, device=input.device, dtype=input.dtype)
        inv_std = torch.zeros(feat_dim, device=input.device, dtype=input.dtype)
        output = torch.empty_like(input)
        return output, mean, inv_std, reserve, new_running_mean, new_running_var

    # The vendor native_batch_norm kernels index the 1-D parameters by feature
    # id with an implicit unit stride, so strided views must be packed first.
    weight_c = weight.contiguous() if weight is not None else None
    bias_c = bias.contiguous() if bias is not None else None

    # _batch_norm_with_update_functional follows standard PyTorch training BN,
    # which keeps UNBIASED running_var (count / (count - 1) correction). Unlike
    # the sibling _batch_norm_with_update, this functional variant's test expects
    # save_mean / save_invstd to follow the INPUT dtype (not the FP32 accumulation
    # dtype), so stats_dtype is left at the default (input dtype).
    output, save_mean, save_invstd = native_batch_norm(
        input,
        weight=weight_c,
        bias=bias_c,
        running_mean=new_running_mean,
        running_var=new_running_var,
        training=True,
        momentum=momentum,
        eps=eps,
        unbiased_running_var=True,
    )

    return output, save_mean, save_invstd, reserve, new_running_mean, new_running_var
