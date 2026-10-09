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

from .index import index

logger = logging.getLogger(__name__)


_DTYPE_LEGACY_NAMES = {
    torch.bool: "Bool",
    torch.int8: "Char",
    torch.uint8: "Byte",
    torch.int16: "Short",
    torch.float16: "Half",
    torch.float32: "Float",
    torch.float64: "Double",
    torch.bfloat16: "BFloat16",
}


def _check_indices(inp, indices):
    """Validate index dtypes and move cross-device indices.

    Mirrors ``aten._unsafe_index``: only int32/int64 index tensors are
    accepted (bool/int8/uint8/float/... are all rejected with a RuntimeError),
    and an index living on a different device than the input is moved onto it.
    """
    checked = []
    for idx in indices:
        if idx is None:
            checked.append(None)
            continue
        if idx.dtype not in (torch.int32, torch.int64):
            name = _DTYPE_LEGACY_NAMES.get(idx.dtype, str(idx.dtype))
            raise RuntimeError(f"_unsafe_index found unexpected index type {name}")
        if idx.device != inp.device:
            idx = idx.to(inp.device)
        checked.append(idx)
    return checked


def _eliminate_scalar_indices(inp, indices):
    """Resolve 0-d (scalar) tensor indices on the host via ``select`` views.

    In aten a 0-d index tensor behaves like an integer index: the indexed dim
    is removed from the output.  Resolving them here keeps the delegated
    gather restricted to index tensors of rank >= 1.
    """
    if not any(idx is not None and idx.ndim == 0 for idx in indices):
        return inp, indices
    remaining = []
    removed = 0
    for i, idx in enumerate(indices):
        if idx is not None and idx.ndim == 0:
            pos = i - removed
            v = int(idx.item())
            if v < 0:
                v += inp.shape[pos]
            inp = inp.select(pos, v)
            removed += 1
        else:
            remaining.append(idx)
    return inp, remaining


def _is_split(indices):
    """True when the advanced (non-None) indices are not contiguous."""
    advanced = [i for i, idx in enumerate(indices) if idx is not None]
    return bool(advanced) and advanced != list(
        range(advanced[0], advanced[0] + len(advanced))
    )


def unsafe_index(inp, indices):
    """``aten._unsafe_index`` for Kunlunxin.

    The generic code-generated strided gather miscompiles on XPU3, so the
    gather is delegated to the vendor ``index`` kernel (validated correct)
    after unsafe-index-specific host preprocessing: int32/int64-only dtype
    checks, 0-d scalar resolution, negative-index wrapping, and the aten
    subspace-split placement rule when a resolved 0-d scalar collapses a
    non-contiguous advanced-index block.
    """
    logger.debug("GEMS_KUNLUNXIN UNSAFE_INDEX")
    if not indices:
        raise ValueError("at least one index must be provided")

    indices = _check_indices(inp, list(indices))
    if len(indices) > inp.ndim:
        raise IndexError(
            f"too many indices for tensor of dimension {inp.ndim} (got {len(indices)})"
        )

    original_split = _is_split(indices)

    inp, indices = _eliminate_scalar_indices(inp, indices)
    if not indices:
        return inp.contiguous()

    tensor_indices = [idx for idx in indices if idx is not None]
    if not tensor_indices:
        # Every advanced index was a resolved 0-d scalar; the rest are full
        # slices, so this is plain basic indexing (a contiguous copy).
        return inp.contiguous()
    if len(tensor_indices) > 1:
        try:
            broadcasted = list(torch.broadcast_tensors(*tensor_indices))
        except RuntimeError:
            shapes = ", ".join(str(list(t.shape)) for t in tensor_indices)
            raise IndexError(
                "shape mismatch: indexing tensors could not be broadcast "
                f"together with shapes {shapes}"
            ) from None
        it = iter(broadcasted)
        indices = [None if idx is None else next(it) for idx in indices]
        tensor_indices = broadcasted
    index_rank = tensor_indices[0].ndim

    wrapped = [
        None if idx is None else torch.where(idx < 0, idx + inp.shape[i], idx)
        for i, idx in enumerate(indices)
    ]

    out = index(inp, wrapped)

    # aten front-places the broadcast subspace whenever the advanced indices
    # were split in the original call; a resolved 0-d scalar can hide that
    # split from the vendor index kernel, so re-apply the front placement.
    if original_split and not _is_split(indices):
        lead = 0
        for idx in indices:
            if idx is None:
                lead += 1
            else:
                break
        order = (
            list(range(lead, lead + index_rank))
            + list(range(0, lead))
            + list(range(lead + index_rank, out.ndim))
        )
        out = out.permute(order).contiguous()

    return out
