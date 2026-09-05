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


def _resize_output(inp: torch.Tensor, size, device):
    """Functional ``_resize_output`` (kunlunxin / XPU).

    The generic implementation (``flag_gems.ops._resize_output``) short-circuits
    to ``inp.reshape(size)`` whenever the new numel equals the old one and the
    device matches, i.e. it returns a VIEW aliasing the input; the expected
    semantics (see the reference implementation used by the tests and ATen's
    ``_resize_output_``) instead produce a fresh output tensor whose first
    ``min(old, new)`` elements are copied in logical (row-major) order. On top
    of that, the generic grow/shrink path goes through ``flag_gems.ops.copy_``
    (a Triton kernel) which is ~22x slower than the vendor's native copy engine
    on XPU (see ``_kunlunxin/ops/resize.py`` for the same analysis of
    ``aten::resize``).

    Fix: allocate the output on the requested device and copy the preserved
    ``min(old, new)`` elements through the ATen ``_copy_from`` primitive. gems
    overrides ``copy_``/``copy`` but never ``_copy_from``, so this reaches the
    native strided-copy engine and runs at native speed even while use_gems is
    active. Growing leaves the tail uninitialized (matching native semantics).
    """
    logger.debug("GEMS _RESIZE_OUTPUT")

    if not isinstance(size, tuple):
        size = tuple(size)

    out = torch.empty(size, device=device, dtype=inp.dtype)

    if inp.numel() == 0 or out.numel() == 0:
        return out

    # Preserve the first min(old_numel, new_numel) elements; the rest
    # (when growing) is left uninitialized, matching native semantics.
    copy_numel = min(inp.numel(), out.numel())
    src = inp.reshape(-1)[:copy_numel]
    dst = out.reshape(-1)[:copy_numel]
    # Native contiguous copy (bypasses the slow gems/Triton copy path).
    torch.ops.aten._copy_from(src, dst, False)

    return out


def _resize_output_(inp: torch.Tensor, size, device):
    """In-place ``_resize_output_`` (kunlunxin / XPU).

    Mirrors ATen's ``_resize_output_``: the tensor must already live on
    ``device`` (TORCH_CHECK in the native implementation), then ``resize_`` runs
    in place (keeps the storage, preserves the first ``min(old, new)`` elements)
    and returns ``self``.
    """
    logger.debug("GEMS _RESIZE_OUTPUT_")

    if not isinstance(size, tuple):
        size = tuple(size)

    if inp.device != torch.device(device):
        raise RuntimeError(
            f"_resize_output_: device mismatch, input tensor is on {inp.device} "
            f"but the requested device is {device}"
        )

    # resize_ is itself backed by the kunlunxin vendor implementation
    # (set_-based, keeps storage, returns self).
    inp.resize_(size)
    return inp
