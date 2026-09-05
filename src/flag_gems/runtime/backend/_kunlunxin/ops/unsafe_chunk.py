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
# Kunlunxin (TritonXPU) specialization of unsafe_chunk.
#
# Why this override exists (2026-09-05, XPU 6)
# --------------------------------------------
# The general implementation in src/flag_gems/ops/unsafe_chunk.py materializes
# each chunk with ``A[tuple(slices)]``, which lowers to ``aten::slice.Tensor``
# and is only correct here because the vendor ``CUSTOMIZED_UNUSED_OPS``
# excludes ``"slice"`` (native torch_xmlir zero-copy view).  It also has two
# deviations from ``torch.unsafe_chunk`` reference semantics:
#
#   * ``chunks <= 0`` raises ``ZeroDivisionError`` (``(size + chunks - 1) // 0``)
#     instead of torch's ``RuntimeError("chunk expects `chunks` to be greater
#     than 0, got: ...")``;
#   * a zero-sized ``dim`` (e.g. ``torch.empty(0)``) returns ``[]`` whereas
#     ``torch.unsafe_chunk`` returns exactly ``chunks`` empty views.
#
# This override re-implements ``torch.unsafe_chunk`` semantics (ceiling
# division, at most one smaller final chunk, returns fewer than ``chunks``
# entries when ``chunks > dim_size``) through ``torch.narrow``, a zero-copy
# view equivalent to slicing that is safe inside ``use_gems()``.

import logging
from typing import List

import torch

logger = logging.getLogger(__name__)


def unsafe_chunk(A: torch.Tensor, chunks: int, dim: int = 0) -> List[torch.Tensor]:
    r"""Split a tensor into :attr:`chunks` pieces along the given dimension.

    The last chunk will be smaller if the tensor size along the given
    dimension is not divisible by :attr:`chunks`. If :attr:`chunks` exceeds
    the size along :attr:`dim`, fewer than :attr:`chunks` tensors are
    returned (matching ``torch.unsafe_chunk``).

    Args:
        A (torch.Tensor): Input tensor.
        chunks (int): Number of chunks to produce.
        dim (int): Dimension along which to split the tensor.

    Returns:
        List[torch.Tensor]: List of tensor chunks (views of the original).
    """
    logger.debug("GEMS_KUNLUNXIN UNSAFE_CHUNK")

    if chunks <= 0:
        raise RuntimeError(
            f"chunk expects `chunks` to be greater than 0, got: {chunks}"
        )

    # Handle negative dim
    if dim < 0:
        dim = dim + A.ndim

    dim_size = A.size(dim)

    # Match torch.unsafe_chunk: a zero-sized dim yields exactly `chunks`
    # empty views (narrow of length 0), not an empty list.
    if dim_size == 0:
        return [torch.narrow(A, dim, 0, 0) for _ in range(chunks)]

    # Calculate the size of each chunk (ceiling division)
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
        # python impl under ``use_gems()``; ``torch.narrow`` is the zero-copy
        # view equivalent.
        result.append(torch.narrow(A, dim, start, end - start))

    return result
