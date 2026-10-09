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

from .native_batch_norm import native_batch_norm

logger = logging.getLogger("flag_gems.ops._native_batch_norm_legit")


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
    logger.debug("GEMS_KUNLUNXIN _NATIVE_BATCH_NORM_LEGIT")
    logger.debug("GEMS _NATIVE_BATCH_NORM_LEGIT")
    output, save_mean, save_invstd = native_batch_norm(
        input,
        weight=weight,
        bias=bias,
        running_mean=running_mean,
        running_var=running_var,
        training=training,
        momentum=momentum,
        eps=eps,
        unbiased_running_var=True,
    )
    return output, save_mean, save_invstd


def _native_batch_norm_legit_no_stats(
    input: Tensor,
    weight,
    bias,
    training: bool,
    momentum: float,
    eps: float,
):
    logger.debug("GEMS_KUNLUNXIN _NATIVE_BATCH_NORM_LEGIT_NO_STATS")
    logger.debug("GEMS _NATIVE_BATCH_NORM_LEGIT_NO_STATS")
    if not training:
        raise RuntimeError("Expected has_running_mean to be true, but got false.")

    channels = input.shape[1]
    running_mean = torch.zeros(channels, dtype=input.dtype, device=input.device)
    running_var = torch.ones(channels, dtype=input.dtype, device=input.device)
    return native_batch_norm(
        input,
        weight=weight,
        bias=bias,
        running_mean=running_mean,
        running_var=running_var,
        training=True,
        momentum=momentum,
        eps=eps,
        unbiased_running_var=True,
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
    logger.debug("GEMS_KUNLUNXIN _NATIVE_BATCH_NORM_LEGIT_OUT")
    logger.debug("GEMS _NATIVE_BATCH_NORM_LEGIT_OUT")
    result = _native_batch_norm_legit(
        input,
        weight,
        bias,
        running_mean,
        running_var,
        training,
        momentum,
        eps,
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
    logger.debug("GEMS_KUNLUNXIN _NATIVE_BATCH_NORM_LEGIT_NO_STATS_OUT")
    logger.debug("GEMS _NATIVE_BATCH_NORM_LEGIT_NO_STATS_OUT")
    result = _native_batch_norm_legit_no_stats(
        input,
        weight,
        bias,
        training,
        momentum,
        eps,
    )
    return _copy_outputs(result, out, save_mean, save_invstd)
