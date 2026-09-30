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

from .batch_norm import batch_norm

logger = logging.getLogger(__name__)


def _batch_norm_with_update_functional(
    input,
    weight=None,
    bias=None,
    running_mean=None,
    running_var=None,
    momentum=0.1,
    eps=1e-05,
):
    logger.debug("GEMS_KUNLUNXIN _BATCH_NORM_WITH_UPDATE_FUNCTIONAL")
    running_mean_out = running_mean.clone() if running_mean is not None else None
    running_var_out = running_var.clone() if running_var is not None else None
    output, save_mean, save_invstd = batch_norm(
        input,
        weight,
        bias,
        running_mean_out,
        running_var_out,
        True,
        momentum,
        eps,
        update_running_all_dtypes=True,
        unbiased_running_var=True,
    )
    reserve = torch.empty((0,), dtype=torch.uint8, device=input.device)
    return output, save_mean, save_invstd, reserve, running_mean_out, running_var_out
