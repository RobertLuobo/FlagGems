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

logger = logging.getLogger(__name__)


def tile(inp: torch.Tensor, dims) -> torch.Tensor:
    logger.debug("GEMS_KUNLUNXIN TILE")

    dims = list(dims)
    in_rank = inp.dim()
    dims_rank = len(dims)
    in_shape = list(inp.shape)

    if dims_rank < in_rank:
        dims = [1] * (in_rank - dims_rank) + dims
    elif dims_rank > in_rank:
        in_shape = [1] * (dims_rank - in_rank) + in_shape

    rank = len(in_shape)

    out_shape = []
    is_empty = False
    for i in range(rank):
        assert dims[i] >= 0, (
            "the number of repetitions per dimension out of range "
            "(expected to >= 0) but got {}".format(dims[i])
        )
        if dims[i] == 0:
            is_empty = True
        out_shape.append(in_shape[i] * dims[i])

    if is_empty:
        return torch.empty(out_shape, device=inp.device, dtype=inp.dtype)

    view_shape = []
    expand_shape = []
    for i in range(rank):
        view_shape += [1, in_shape[i]]
        expand_shape += [dims[i], in_shape[i]]

    return (
        inp.reshape(view_shape)
        .expand(expand_shape)
        .reshape(out_shape)
        .contiguous()
    )
