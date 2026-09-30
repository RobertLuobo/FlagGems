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
from typing import List, Union

import torch

logger = logging.getLogger(__name__)


def vsplit(input: torch.Tensor, indices_or_sections: Union[int, List[int]]):
    logger.debug("GEMS_KUNLUNXIN VSPLIT")

    if input.ndim < 2:
        raise RuntimeError(
            f"vsplit requires a tensor with 2 or more dimensions, got {input.ndim}"
        )

    dim = 0
    dim_size = input.shape[dim]

    if isinstance(indices_or_sections, int):
        n = indices_or_sections
        if n <= 0:
            raise ValueError(f"indices_or_sections must be positive, got {n}")
        if dim_size % n != 0:
            raise RuntimeError(
                f"torch.vsplit attempted to split along dimension {dim}, but the "
                f"size of the dimension {dim_size} is not divisible by the "
                f"split_size {n}!"
            )
        chunk_sizes = [dim_size // n] * n
    else:
        chunk_sizes = []
        prev = 0
        for idx in indices_or_sections:
            idx = max(0, min(int(idx), dim_size))
            chunk_sizes.append(idx - prev)
            prev = idx
        chunk_sizes.append(dim_size - prev)

    outputs = []
    start = 0
    for chunk_size in chunk_sizes:
        outputs.append(torch.narrow(input, dim, start, chunk_size))
        start += chunk_size

    return tuple(outputs)
