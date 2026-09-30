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

from flag_gems.runtime.backend._kunlunxin.ops.sum import sum as _sum
from flag_gems.runtime.backend._kunlunxin.ops.sum import sum_dim as _sum_dim

logger = logging.getLogger(__name__)


def sum_to_size(inp, *size):
    logger.debug("GEMS_KUNLUNXIN SUM_TO_SIZE")
    if len(size) == 1 and isinstance(size[0], (list, tuple)):
        size = tuple(size[0])
    else:
        size = tuple(size)

    target_shape = list(size)
    inp_shape = list(inp.shape)

    leading_dims = inp.ndim - len(target_shape)
    reduce_dims = list(range(leading_dims))
    for i in range(leading_dims, inp.ndim):
        if target_shape[i - leading_dims] == 1 and inp_shape[i] != 1:
            reduce_dims.append(i)

    if len(reduce_dims) == 0:
        return inp.reshape(target_shape)

    if inp.dtype is torch.bool:
        inp = inp.to(torch.int64)

    result_numel = 1
    for s in target_shape:
        result_numel *= s

    if result_numel <= 1:
        out = _sum(inp)
        return out.reshape(target_shape)

    out = _sum_dim(inp, reduce_dims, keepdim=True)
    return out.reshape(target_shape)
