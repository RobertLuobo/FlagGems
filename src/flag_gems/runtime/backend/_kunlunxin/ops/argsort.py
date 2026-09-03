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

from .sort import radix_argsort, sort_stable

logger = logging.getLogger(__name__)


def argsort(inp, dim=-1, descending=False):
    logger.debug("GEMS_KUNLUNXIN ARGSORT")
    if dim < 0:
        dim = dim + inp.dim()
    # Trivial cases: an empty or single-element row has a unique stable order.
    if inp.shape[dim] <= 1:
        return torch.zeros(inp.shape, dtype=torch.int64, device=inp.device)
    # 64-bit dtypes (int64/uint64/float64): the packed 64-bit key
    # (u32(value) << 32) | column cannot hold the full key, so fall back to
    # the (validated) value+index radix chain.
    if inp.element_size() * 8 > 32:
        _, indices = sort_stable(inp, stable=True, dim=dim, descending=descending)
        return indices
    # Move the sorted dim to the end (radix_argsort sorts the last dim),
    # then move the indices back.
    if dim != inp.dim() - 1:
        inp = inp.movedim(dim, -1)
        indices = radix_argsort(inp, descending=descending)
        return indices.movedim(-1, dim)
    return radix_argsort(inp, descending=descending)
