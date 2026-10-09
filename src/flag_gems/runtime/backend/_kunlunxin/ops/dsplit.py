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
#
import logging
from typing import List, Union

import torch

logger = logging.getLogger(__name__)


def dsplit(input: torch.Tensor, indices_or_sections: Union[int, List[int]]):
    """Split a tensor depth-wise (``dim=2``) on kunlunxin / XPU (zero-copy view).

    ``torch.dsplit`` is ``torch.tensor_split`` along ``dim=2`` and is natively a
    view operation: each returned chunk shares storage with ``input``. The
    returned chunks are produced with ``torch.narrow`` (a registered view impl,
    same precedent as ``_kunlunxin/ops/narrow.py`` / ``slice.py``); this avoids
    the generic copy kernel whose ``start`` ``tl.constexpr`` specialization is
    mis-baked on XPU3 (any ``start != 0`` chunk from an unequal split reads the
    wrong depth offset) and matches native perf (zero-copy instead of a full
    device-side copy).
    """
    logger.debug("GEMS_KUNLUNXIN DSPLIT")

    if input.ndim < 3:
        raise RuntimeError(
            f"dsplit requires a tensor with 3 or more dimensions, got {input.ndim}"
        )

    dim = 2
    dim_size = input.shape[dim]

    # Resolve chunk sizes along the depth dimension (torch.tensor_split rules).
    if isinstance(indices_or_sections, int):
        n = indices_or_sections
        if n <= 0:
            raise ValueError(f"indices_or_sections must be positive, got {n}")
        base = dim_size // n
        remainder = dim_size % n
        chunk_sizes = [base + 1 if i < remainder else base for i in range(n)]
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
        # torch.narrow -> zero-copy strided view sharing input storage.
        outputs.append(torch.narrow(input, dim, start, chunk_size))
        start += chunk_size

    return tuple(outputs)
