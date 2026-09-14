# Copyright 2026, The FlagOS Contributors.
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


def resize(inp: torch.Tensor, size, memory_format=None): 
    logger.debug("GEMS RESIZE")

    if not isinstance(size, tuple):
        size = tuple(size)

    out = torch.empty(size, device=inp.device, dtype=inp.dtype)

    if inp.numel() == 0 or out.numel() == 0:
        return out

    # resize preserves the first min(old_numel, new_numel) elements; the rest
    # (when growing) is left uninitialized, matching native semantics.
    copy_numel = min(inp.numel(), out.numel())
    src = inp.reshape(-1)[:copy_numel]
    dst = out.reshape(-1)[:copy_numel]
    # Native contiguous copy (bypasses the slow gems/Triton copy path).
    torch.ops.aten._copy_from(src, dst, False)

    return out


def resize_(inp: torch.Tensor, size, memory_format=None):
    logger.debug("GEMS RESIZE_")

    if not isinstance(size, tuple):
        size = tuple(size)

    inp.set_(inp.untyped_storage(), 0, size)
    return inp
