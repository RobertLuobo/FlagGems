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
# Kunlunxin (TritonXPU) specialization of slice.
#
# Why this override exists (2026-09-05, XPU 6)
# --------------------------------------------
# The general implementation in src/flag_gems/ops/slice.py has the signature
# ``slice(input_tensor, dim, start, end, step)`` with a *required* ``step``.
# Under ``use_gems()`` the registered ``slice.Tensor`` python impl is invoked
# from C++ ``Tensor::slice(dim, start, end)`` callsites (e.g. the ATen
# implementation of ``tensor_split``) with only 4 positional arguments
# (self, dim, start, end -- ``step`` is omitted), which raises::
#
#     TypeError: slice() missing 1 required positional argument: 'step'
#
# -> tests/test_tensor_split.py: 36 failed with that error on the first run.
#
# This override re-implements ``slice.Tensor`` with a defaulted ``step=1`` and
# as a zero-copy *view* through ``torch.narrow`` (step == 1) or
# ``torch.as_strided`` (step != 1), i.e. the same storage-sharing semantics as
# ``aten::slice.Tensor``. Both helpers are safe inside ``use_gems()``:
# ``torch.narrow`` is itself a registered view-based impl and
# ``torch.as_strided`` is not registered at all (no re-dispatch/recursion).

import logging

import torch

logger = logging.getLogger(__name__)


def slice(
    input_tensor: torch.Tensor, dim: int, start, end, step: int = 1
) -> torch.Tensor:
    r"""Return a zero-copy view of ``input_tensor`` along ``dim``.

    Mirrors ``aten::slice.Tensor``: ``start``/``end`` may be ``None`` or
    negative, and a non-unit ``step`` produces a strided (still zero-copy)
    view.  The ``step`` argument is optional because the C++
    ``Tensor::slice(dim, start, end)`` callsite (used internally by
    ``tensor_split`` and friends) omits it.
    """
    logger.debug("GEMS_KUNLUNXIN SLICE")

    if step == 0:
        raise RuntimeError("slice step cannot be 0")

    ndim = input_tensor.ndim
    if ndim == 0:
        raise RuntimeError("slice() cannot be applied to a 0-dim tensor.")
    dim = dim % ndim  # normalize negative dim
    dim_size = input_tensor.size(dim)

    # start/end may arrive as 0-dim integral tensors (from narrow-style
    # callsites); normalize to plain ints for the arithmetic below.
    if isinstance(start, torch.Tensor):
        start = start.item()
    if isinstance(end, torch.Tensor):
        end = end.item()

    if step > 0:
        if start is None:
            start = 0
        elif start < 0:
            start = dim_size + start
        start = max(0, min(start, dim_size))

        if end is None:
            end = dim_size
        elif end < 0:
            end = dim_size + end
        end = max(0, min(end, dim_size))

        length = max(0, (end - start + step - 1) // step)

        if step == 1:
            # ``torch.narrow`` is a registered view-based impl (zero-copy via
            # ``as_strided``); equivalent to ``input_tensor[.., start:end, ..]``.
            return torch.narrow(input_tensor, dim, start, length)
    else:
        # Negative step: element positions start, start+step, ... > end.
        if start is None:
            start = dim_size - 1
        elif start < 0:
            start = dim_size + start
        start = max(-1, min(start, dim_size - 1))

        if end is None:
            end = -1
        elif end < 0:
            end = dim_size + end
        end = max(-1, min(end, dim_size - 1))

        length = max(0, (start - end + (-step) - 1) // (-step))

    # General (strided) zero-copy view.
    size = list(input_tensor.shape)
    size[dim] = length
    stride = list(input_tensor.stride())
    stride[dim] = input_tensor.stride(dim) * step
    storage_offset = input_tensor.storage_offset() + start * input_tensor.stride(dim)

    return torch.as_strided(input_tensor, size, stride, storage_offset)
