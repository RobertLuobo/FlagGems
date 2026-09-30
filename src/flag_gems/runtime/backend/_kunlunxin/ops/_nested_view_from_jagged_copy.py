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

from flag_gems.ops._nested_view_from_jagged import _nested_view_from_jagged
from flag_gems.ops._nested_view_from_jagged_copy import (
    _nested_view_from_jagged_copy_kernel,
)
from flag_gems.runtime import torch_device_fn

logger = logging.getLogger("flag_gems." + __name__)

# Small-shape default: matches the generic kernel; large blocks serialize a
# single program and regress launch-bound small copies.
_SMALL_BLOCK = 1024
# Large-shape flat block-DMA path: BLOCK=65536 / 32 warps. n>=1M switches over
# because the generic 1024-element blocks become launch-bound (16.7M elements ->
# 16384 programs). Threshold sits between the benchmark's 262144 and 2097152.
_LARGE_BLOCK = 65536
_LARGE_WARPS = 32
_LARGE_GATE = 1 << 20


def _nested_view_from_jagged_copy(
    self: torch.Tensor,
    offsets: torch.Tensor,
    dummy: torch.Tensor,
    lengths=None,
    ragged_idx=1,
    min_seqlen=None,
    max_seqlen=None,
):
    logger.debug("GEMS_KUNLUNXIN _NESTED_VIEW_FROM_JAGGED_COPY")

    src = self.contiguous() if not self.is_contiguous() else self
    values_copy = torch.empty_like(src)
    n_elements = values_copy.numel()

    if n_elements > 0:
        if n_elements >= _LARGE_GATE:
            grid = (triton.cdiv(n_elements, _LARGE_BLOCK),)
            with torch_device_fn.device(src.device):
                _nested_view_from_jagged_copy_kernel[grid](
                    src,
                    values_copy,
                    n_elements,
                    BLOCK_SIZE=_LARGE_BLOCK,
                    num_warps=_LARGE_WARPS,
                )
        else:
            grid = (triton.cdiv(n_elements, _SMALL_BLOCK),)
            with torch_device_fn.device(src.device):
                _nested_view_from_jagged_copy_kernel[grid](
                    src,
                    values_copy,
                    n_elements,
                    BLOCK_SIZE=_SMALL_BLOCK,
                )

    return _nested_view_from_jagged(
        values_copy,
        offsets,
        dummy,
        lengths,
        ragged_idx,
        min_seqlen,
        max_seqlen,
    )


__all__ = ["_nested_view_from_jagged_copy"]
