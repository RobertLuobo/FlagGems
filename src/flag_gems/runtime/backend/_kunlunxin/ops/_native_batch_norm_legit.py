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

from torch import Tensor

from .batch_norm import batch_norm

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
    logger.debug("GEMS_KUNLUNXIN _NATIVE_BATCH_NORM_LEGIT")
    output, save_mean, save_invstd = batch_norm(
        input,
        weight,
        bias,
        running_mean,
        running_var,
        training,
        momentum,
        eps,
        update_running_all_dtypes=True,
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
    if not training:
        raise RuntimeError("Expected has_running_mean to be true, but got false.")
    output, save_mean, save_invstd = batch_norm(
        input,
        weight,
        bias,
        None,
        None,
        True,
        momentum,
        eps,
        update_running_all_dtypes=True,
        unbiased_running_var=True,
    )
    return output, save_mean, save_invstd


def _copy_outputs(result, out, save_mean, save_invstd):
    result_out, result_mean, result_invstd = result
    out.resize_as_(result_out).copy_(result_out)
    save_mean.resize_as_(result_mean).copy_(result_mean)
    save_invstd.resize_as_(result_invstd).copy_(result_invstd)
    return out, save_mean, save_invstd


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
    result = _native_batch_norm_legit_no_stats(
        input,
        weight,
        bias,
        training,
        momentum,
        eps,
    )
    return _copy_outputs(result, out, save_mean, save_invstd)
