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
"""Kunlunxin (XPU) unique_consecutive.

History / why this file no longer looks like the generic implementation:

The previous body was a verbatim copy of ``flag_gems/ops/unique_consecutive.py``
and carried two XPU-specific defects that its functional tests never exercised
end to end:

* **Wrong results** (shapes where ``num_tasks % tile_size == 0``, e.g.
  ``(1024, 1024)``, ``(20, 320, 15)``, ``(16, 128, 64, 1280)``): the
  ``global_cumsum_consecutive_impl`` kernel mixes a ``tl.sum`` carry with a
  ``tl.cumsum`` in the *same* kernel.  On this backend that pairing miscompiles
  (measured garbage); the sibling ``_kunlunxin/ops/unique.py`` documents the very
  same finding and works around it by splitting ``tl.sum`` and ``tl.cumsum`` into
  separate kernels.
* **Hang** (shapes with a partial last tile, e.g. ``N=9000`` and
  ``(16, 7, 57, 32, 29)``): the tail lanes of ``local_ne_consecutive_impl`` are
  loaded with a ``mask`` but no ``other=``, so their undefined values feed
  ``ne_result``; the resulting inflated ``out_idx`` reaches >= ``num_tasks`` and
  the (masked) scatter store into ``data_out`` runs off the end of the buffer.
  Worse, on the last tile ``global_cumsum_consecutive_impl`` stores a
  ``tile_size``-wide vector at ``tile_sum_ptr + global_pid`` even though
  ``tile_sum`` only has ``global_ctas_num`` elements -- an overrun of up to
  ``tile_size - 1`` int64 past the allocation.  Those wild writes are what took
  the device into the ``KL_XID_KERNEL_EXCEPTION`` / ``-299`` state.

The rewrite below follows the backend's endorsed strategy (see the
``unique_dim.py`` docstring): no ``other=``, no masked stores on a discrete
scatter, every buffer over-allocated to whole tiles, and ``tl.sum`` / ``tl.cumsum``
never mixed in one kernel.  It reuses the multi-level chunked scan toolkit that
``_unique2`` already exercises (``_triton_inclusive_scan``), so the scan itself is
the identical, verified code path.

A later device review of that first rewrite found a *second* XPU-specific defect:
the finalize kernel used to emit two int64-valued stores (the int64
``inverse_indices`` writer and the int64 group-start writer), and on this backend
a kernel with **>= 2 int64-valued stores** fails the ``TritonXPUUnrollControl``
MLIR pass -- reported as
``'arith.extsi' op failed to verify that input and output have the same tensor
dimensions`` followed by ``OutOfResources: uni_sram`` -- so the op compiled for
*no* shape.  The fix splits the work so that no single kernel emits more than one
int64 store: the finalize kernel writes the data-dtype output and an **int32**
group-start buffer, ``_uc_inverse_kernel`` writes the int64 inverse, and
``_uc_run_lengths_kernel`` widens the int32 starts to the int64 counts.
"""

import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils.libentry import libentry

# Reuse the already-verified scan machinery from the sibling vendor module
# (`unique.py` is imported before this module by ops/__init__.py, so this import
# cannot be circular, and `unique.py` does not import this file).
from .unique import _SCAN_BLOCK, _triton_inclusive_scan

logger = logging.getLogger(__name__)

# One tile width for every stage.  Equal to _SCAN_BLOCK so the ne/cum buffers are
# block-aligned and the finalize kernel can read them unmasked.
_UC_BLOCK = _SCAN_BLOCK


@libentry()
@triton.jit
def _ne_consecutive_kernel(
    data_ptr,
    ne_ptr,
    N,
    BLOCK: tl.constexpr,
):
    """ne[i] = 1 if element i starts a new consecutive group, else 0 (int64).

    Loads are unmasked with clamped addresses and the result is written unmasked,
    so nothing depends on masked-load semantics or on `other=`.  `ne_ptr` is
    over-allocated to a whole number of tiles (N_pad) and the padding lanes are
    written as 0.
    """
    pid = tl.program_id(0)
    r = tl.arange(0, BLOCK)
    offs = pid * BLOCK + r

    live = offs < N
    # Clamp both addresses into [0, N) so no load ever leaves the tensor.
    src = tl.where(live, offs, 0)
    prev_raw = tl.where(offs > 0, offs - 1, 0)
    src_prev = tl.where(prev_raw < N, prev_raw, 0)

    a = tl.load(data_ptr + src)
    b = tl.load(data_ptr + src_prev)

    # Pure int arithmetic (no vector i1 and/or) then a single cast, per the
    # backend notes on narrow i1 vectors.
    is_first = tl.where(offs == 0, 1, 0)
    diff = tl.where(a != b, 1, 0)
    has_prev = tl.where(offs > 0, 1, 0)
    ne = is_first + diff * has_prev
    ne = tl.where(live, ne, 0).to(tl.int64)
    tl.store(ne_ptr + offs, ne)


@libentry()
@triton.jit
def _unique_consecutive_finalize_kernel(
    data_ptr,
    cum_ptr,
    ne_ptr,
    out_ptr,
    start_ptr,
    n_unique,
    N,
    BLOCK: tl.constexpr,
    return_counts: tl.constexpr,
):
    """Scatter the unique values and the group start offsets.

    `cum` is the inclusive prefix sum of `ne`, so `group = cum - 1` is the output
    index of every element and `n_unique = cum[N-1]`.

    Inactive lanes and padding lanes are *not* masked off: they are redirected to
    a per-lane scratch slot ``n_unique + r`` (in-block unique), which is inside the
    over-allocated buffers.  This keeps every store unmasked, so it is immune to
    the backend's masked-store hazard.  For a group-start lane the destination is
    the group index and exactly one lane per group writes it, so there is no
    scatter race; the stored value is the group's first element, and every other
    lane of the group would write the identical bytes anyway.

    **Compile-failure workaround (TritonXPUUnrollControl).**  On this backend a
    kernel that emits >= 2 stores whose *value* is int64 fails the
    `TritonXPUUnrollControl` pass with a bogus
    ``'arith.extsi' op failed to verify that input and output have the same tensor
    dimensions`` (the real line is the last int64 store).  Hence this kernel
    writes only two things: the data-dtype `out` value and -- when counts are
    requested -- the group start offset as **int32**.  The int64 `inverse_indices`
    moved to its own kernel (`_uc_inverse_kernel`) and the int64 `counts` to
    `_uc_run_lengths_kernel`, so each kernel has at most one int64-valued store.
    `start` values fit in int32 because ``num_tasks <= 2**31``.
    """
    pid = tl.program_id(0)
    r = tl.arange(0, BLOCK)
    offs = pid * BLOCK + r
    idx = pid * BLOCK + r

    live = offs < N
    src = tl.where(live, offs, 0)
    a = tl.load(data_ptr + src)
    c = tl.load(cum_ptr + idx)
    ne = tl.load(ne_ptr + idx)

    group = (c - 1).to(tl.int32)
    scratch = n_unique + r
    is_start = ne != 0
    dst = tl.where(is_start, group, scratch)

    # data_out[group] = first element of the group (value: data dtype).
    tl.store(out_ptr + dst, a)

    if return_counts:
        # Only the single group-start lane per group writes here, so the value is
        # the start offset (int32) of that group -- exactly what
        # `_uc_run_lengths_kernel` wants.
        tl.store(start_ptr + dst, offs)


@libentry()
@triton.jit
def _uc_inverse_kernel(
    cum_ptr,
    inv_ptr,
    N,
    BLOCK: tl.constexpr,
):
    """inv[i] = cum[i] - 1 = the output index of input element i (int64).

    Single int64-valued store in the kernel, per the compile-failure workaround
    documented on `_unique_consecutive_finalize_kernel`.
    """
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    live = offs < N
    c = tl.load(cum_ptr + offs)
    tl.store(inv_ptr + offs, tl.where(live, c - 1, 0))


@libentry()
@triton.jit
def _uc_run_lengths_kernel(
    start32_ptr,
    counts_ptr,
    N,
    n,
    BLOCK: tl.constexpr,
):
    """counts[i] = start32[i+1] - start32[i] (last: N - start32[n-1]).

    `start32` is int32 (written by the finalize kernel); the subtraction is done
    in int32 and only the final store is widened to int64 -- one int64-valued
    store, per the compile-failure workaround.  The loads clamp their addresses
    (so no read depends on mask semantics or on `other=`), but the store MUST be
    masked: `counts` holds exactly `n_unique` elements while the grid covers
    `ceil(n_unique / BLOCK)` whole tiles, so an unmasked store would write up to
    `BLOCK - 1` int64 past the end of the allocation.  `n_unique` is an arbitrary
    run count, so that overflow is the common case, not a corner case.
    """
    pid = tl.program_id(0)
    i = pid * BLOCK + tl.arange(0, BLOCK)
    cur = tl.load(start32_ptr + tl.minimum(i, n - 1))
    nxt = tl.load(start32_ptr + tl.minimum(i + 1, n - 1))
    cnt = tl.where(i + 1 < n, nxt - cur, N - cur)
    tl.store(counts_ptr + i, cnt.to(tl.int64), mask=i < n)


def unique_consecutive(
    input: torch.Tensor,
    return_inverse: bool = False,
    return_counts: bool = False,
    dim: int = None,
):
    """
    Eliminates all but the first element from every consecutive group of equivalent elements.

    Args:
        input: the input tensor
        return_inverse: Whether to return inverse indices
        return_counts: Whether to return counts for each unique element
        dim: the dimension to apply unique. If None, the unique of the flattened input is returned.

    Returns:
        (Tensor, Tensor (optional), Tensor (optional)): output, inverse_indices, counts
    """
    logger.debug("GEMS_KUNLUNXIN UNIQUE_CONSECUTIVE")

    if dim is not None:
        raise NotImplementedError(
            "Kunlunxin unique_consecutive currently supports only dim=None"
        )

    # Flatten input for the None dim case
    flat_input = input.ravel()
    num_tasks = flat_input.numel()
    device = flat_input.device

    if num_tasks == 0:
        # Handle empty input
        output = torch.empty(0, dtype=input.dtype, device=device)
        inverse_indices = (
            torch.empty(0, dtype=torch.int64, device=device)
            if return_inverse
            else None
        )
        counts = (
            torch.empty(0, dtype=torch.int64, device=device)
            if return_counts
            else None
        )
        return output, inverse_indices, counts

    num_ctas = triton.cdiv(num_tasks, _UC_BLOCK)
    num_pad = num_ctas * _UC_BLOCK

    # --- stage 1: group-start flag, int64, padded to whole tiles -------------
    ne = torch.empty(num_pad, dtype=torch.int64, device=device)
    with torch_device_fn.device(device.index):
        _ne_consecutive_kernel[(num_ctas,)](flat_input, ne, num_tasks, BLOCK=_UC_BLOCK)

    # --- stage 2: exclusive/inclusive prefix sum (verified toolkit) ----------
    cum = _triton_inclusive_scan(ne)
    n_unique = int(cum[num_tasks - 1].item())

    # Buffers over-allocated by one tile: the extra `_UC_BLOCK` slots are the
    # per-lane scratch target for inactive lanes.
    data_out = torch.empty(n_unique + _UC_BLOCK, dtype=flat_input.dtype, device=device)
    # Group start offsets are int32 here and widened to int64 only inside the
    # counts kernel (see the compile-failure note on the finalize kernel).
    start_positions = (
        torch.empty(n_unique + _UC_BLOCK, dtype=torch.int32, device=device)
        if return_counts
        else None
    )

    # --- stage 3: scatter outputs + group start offsets ---------------------
    dummy = data_out
    with torch_device_fn.device(device.index):
        _unique_consecutive_finalize_kernel[(num_ctas,)](
            flat_input,
            cum,
            ne,
            data_out,
            start_positions if start_positions is not None else dummy,
            n_unique,
            num_tasks,
            BLOCK=_UC_BLOCK,
            return_counts=return_counts,
        )

    output = data_out[:n_unique]

    # --- stage 4: inverse indices (own kernel -> a single int64 store) ------
    inverse_indices = None
    if return_inverse:
        inverse_indices = torch.empty(num_pad, dtype=torch.int64, device=device)
        with torch_device_fn.device(device.index):
            _uc_inverse_kernel[(num_ctas,)](
                cum, inverse_indices, num_tasks, BLOCK=_UC_BLOCK
            )
        inverse_indices = inverse_indices[:num_tasks].view_as(input)

    # --- stage 5: run lengths (own kernel -> a single int64 store) ----------
    counts = None
    if return_counts:
        counts = torch.empty(n_unique, dtype=torch.int64, device=device)
        with torch_device_fn.device(device.index):
            _uc_run_lengths_kernel[(triton.cdiv(n_unique, _UC_BLOCK),)](
                start_positions[:n_unique],
                counts,
                num_tasks,
                n_unique,
                BLOCK=_UC_BLOCK,
            )

    return output, inverse_indices, counts
