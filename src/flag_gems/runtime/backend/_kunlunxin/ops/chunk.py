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
# Kunlunxin (TritonXPU) specialization of chunk.
#
# Why this override exists (2026-09-05, XPU 6)
# --------------------------------------------
# The general implementation in src/flag_gems/ops/chunk.py materializes each
# chunk with ``A[tuple(slices)]``, which lowers to ``aten::slice.Tensor``.
# Under ``use_gems()`` the registered ``slice.Tensor`` python impl is invoked
# with 4 positional arguments (self, dim, start, end -- ``step`` is omitted),
# so any *non-full* slice raises::
#
#     TypeError: slice() missing 1 required positional argument: 'step'
#
# -> tests/test_chunk.py: 0 passed / 72 failed
# -> benchmark/test_chunk.py aborts on the first Gems measurement.
#
# This override re-implements the same chunking math (ceiling division, and at
# most one smaller final chunk; returns fewer than ``chunks`` entries when
# ``ceil(size/chunks) * chunks > size``) through ``torch.narrow``, a zero-copy
# view equivalent to slicing that is safe inside ``use_gems()``.

import logging
from typing import List

import torch

logger = logging.getLogger(__name__)


def chunk(A: torch.Tensor, chunks: int, dim: int = 0) -> List[torch.Tensor]:
    r"""Split a tensor into a specific number of chunks.

    The last chunk will be smaller if the tensor size along the given
    dimension is not divisible by :attr:`chunks`. If :attr:`chunks` exceeds
    the size along :attr:`dim`, fewer than :attr:`chunks` tensors are
    returned (matching ``torch.chunk``).

    Args:
        A (torch.Tensor): Input tensor.
        chunks (int): Number of chunks to produce.
        dim (int): Dimension along which to split the tensor.

    Returns:
        List[torch.Tensor]: List of tensor chunks (views of the original).
    """
    logger.debug("GEMS_KUNLUNXIN CHUNK")

    if chunks <= 0:
        raise RuntimeError(
            f"chunk expects `chunks` to be greater than 0, got: {chunks}"
        )

    # Handle negative dim
    if dim < 0:
        dim = dim + A.ndim

    # Calculate the size of each chunk (ceiling division)
    dim_size = A.size(dim)
    chunk_size = (dim_size + chunks - 1) // chunks

    # Create list to hold chunks
    result = []

    for i in range(chunks):
        start = i * chunk_size

        # Stop if this chunk would be empty
        if start >= dim_size:
            break

        end = min(start + chunk_size, dim_size)
        # ``A[a:b, ...]`` would dispatch to the registered ``slice.Tensor``
        # python impl (missing the 'step' argument) under ``use_gems()``;
        # ``torch.narrow`` is the zero-copy view equivalent.
        result.append(torch.narrow(A, dim, start, end - start))

    return result
