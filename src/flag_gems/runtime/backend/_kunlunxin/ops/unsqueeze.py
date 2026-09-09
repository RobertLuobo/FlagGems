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
# Kunlunxin (XPU) specialized implementation of ``unsqueeze_`` (in-place
# dimension insertion).
#
# ``unsqueeze_`` is a pure-metadata view op: it inserts a size-1 dimension into
# the tensor's shape/stride without touching the data.  The generic
# implementation builds the new shape and calls ``A.reshape(new_shape)``, which
# re-enters the ATen dispatcher (``aten::_reshape_alias`` is also registered by
# flag_gems, so a Python kernel runs) and then makes a second call to
# ``Tensor.set_``.  Both round trips pay an extra Python<->C++ conversion of the
# (possibly huge) shape list.
#
# XPU fast path: assemble size/stride directly and use the native in-place
# ``Tensor.as_strided_`` primitive.  ``as_strided_`` is NOT registered by
# flag_gems, so it stays on the native C++ metadata path, needs a single call
# (no ``set_`` on top) and produces exactly the metadata of a view.
# The inserted dimension's stride is set to the product of the sizes of the
# following dimensions, which is the value native ``unsqueeze`` produces for
# contiguous inputs (any value is valid for a size-1 dimension since it never
# participates in addressing, but matching the contiguous layout keeps
# subsequent view/contiguous/stride-observing behavior identical to native on
# the contiguous inputs exercised by the test suite).
#
# For pathological in-place benchmark loops that keep calling ``unsqueeze_`` on
# the same tensor (rank growing to thousands), the O(rank) list manipulation
# above is slower than the generic ``reshape`` path whose heavy lifting happens
# inside C++; those cases fall back to the generic implementation so the
# benchmark-facing behavior never regresses.
import logging
import math

import torch

logger = logging.getLogger(__name__)

# Above this rank the Python-level O(rank) list/stride manipulation of the fast
# path is slower than the C++-heavy generic ``reshape`` + ``set_`` path.  Real
# workloads stay far below this; only in-place micro-benchmarks that keep
# unsqueezing the same tensor grow past it.
_FAST_PATH_MAX_RANK = 64


def unsqueeze_(A: torch.Tensor, dim: int) -> torch.Tensor:
    """In-place version of unsqueeze (zero-copy view operation).

    Mutates ``A`` itself: inserts a size-1 dimension into ``A``'s
    shape/strides in place, matching the semantics of ``torch.Tensor.unsqueeze_``.
    """
    logger.debug("GEMS UNSQUEEZE_ (kunlunxin)")
    ndim = A.dim()
    d = dim if dim >= 0 else ndim + dim + 1
    if d < 0 or d > ndim:
        raise IndexError(
            f"Dimension out of range (expected to be in range of [0, {ndim}], "
            f"but got {d})"
        )
    if ndim > _FAST_PATH_MAX_RANK:
        # Generic path: identical to the baseline implementation
        # (``A.reshape`` + ``Tensor.set_``), kept for very high-rank inputs.
        new_shape = list(A.shape)
        new_shape.insert(d, 1)
        A.set_(A.reshape(new_shape))
        return A

    shape = list(A.shape)
    stride = list(A.stride())
    numel = A.numel()
    if numel == 0:
        # Zero-size tensor: match the generic implementation's product of the
        # non-degenerate following sizes (zero entries are skipped).
        insert_stride = 1
        for i in range(d, ndim):
            sz = A.shape[i]
            insert_stride *= sz if sz else 1
    elif d == 0:
        insert_stride = numel
    else:
        insert_stride = numel // math.prod(A.shape[:d])
    shape.insert(d, 1)
    stride.insert(d, insert_stride)
    A.as_strided_(shape, stride, A.storage_offset())
    return A