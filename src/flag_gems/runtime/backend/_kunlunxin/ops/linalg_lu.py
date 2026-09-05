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

"""Kunlunxin (XPU) linalg_lu backend override.

The generic ``flag_gems/ops/linalg_lu.py`` implementation cannot be used on
this backend: its fused kernel (``_linalg_lu_fused_kernel``) and the generic
blocked-path ``linalg_lu_factor`` panel kernel both embed ``tl.sum`` /
``tl.max`` / ``tl.min`` reductions inside a ``tl.range`` elimination loop,
which the XPU ``TritonXPUCoreTiling`` pass rejects with an UNREACHABLE
("Not All Reduce Op can be Optimized", CoreTiling.cpp:203), surfaced by the
compiler as ``OutOfResources: uni_sram`` at linalg_lu.py:96 /
linalg_lu_factor.py:149.

This override composes the two backend-local, XPU-verified primitives:

- ``_linalg_lu_factor`` (``_kunlunxin/ops/linalg_lu_factor.py``): 64-row
  aligned block-parallel pivot search (exact-length vectors, no masked
  ``tl.argmax``), plus row-swap / column-scale / 2-D trailing-update kernels
  with ``J`` as a runtime scalar (one compilation per shape).
- ``lu_unpack`` (``_kunlunxin/ops/lu_unpack.py``): O(k) index-vector swap
  permutation for m > 512, generic L/U extraction otherwise (delegation to
  the generic kernels only for the P/L/U materialization, which are simple
  masked load/store patterns — no reduction in a loop).

No CPU/ATen/native/composite fallback is involved: factorization and unpack
are pure Triton kernels on the XPU device.

``pivot=False`` is not supported on this backend: the vendor lu_factor
primitive rejects it and no XPU-safe no-pivot kernel is available (matching
``_kunlunxin/ops/linalg_lu_factor.py``; the generics for the local test
matrix only exercise ``pivot=True``).
"""

import logging
from collections import namedtuple

import torch

from .linalg_lu_factor import _check_linalg_lu_factor, _linalg_lu_factor
from .lu_unpack import lu_unpack

logger = logging.getLogger(__name__)

LinalgLUResult = namedtuple("LinalgLUResult", ["P", "L", "U"])


def _check_linalg_lu(input, pivot):
    _check_linalg_lu_factor(input, pivot)
    if not pivot:
        raise NotImplementedError(
            "Kunlunxin linalg_lu does not support pivot=False: "
            "the vendor lu_factor primitive rejects it and no XPU-safe "
            "no-pivot kernel is available"
        )


def _linalg_lu_impl(input, *, pivot=True, P=None, L=None, U=None):
    input_contiguous = input.contiguous()
    lu, pivots = _linalg_lu_factor(input_contiguous, pivot)
    p, l, u = lu_unpack(lu, pivots, unpack_data=True, unpack_pivots=True)

    # Write back through the raw native strided-copy engine
    # (``aten::_copy_from``) instead of the gems-registered ``copy_``
    # to avoid a nested dispatch through the overridden operator.
    if P is not None:
        torch.ops.aten._copy_from(p, P, False)
        p = P
    if L is not None:
        torch.ops.aten._copy_from(l, L, False)
        l = L
    if U is not None:
        torch.ops.aten._copy_from(u, U, False)
        u = U
    return LinalgLUResult(p, l, u)


def linalg_lu(input, *, pivot=True):
    logger.debug("GEMS_KUNLUNXIN LINALG_LU")
    _check_linalg_lu(input, pivot)
    return _linalg_lu_impl(input, pivot=pivot)


def _resolve_linalg_lu_out_args(P, L, U, out):
    if out is not None:
        if P is not None or L is not None or U is not None:
            raise TypeError("linalg_lu(): out and P/L/U cannot both be set")
        if len(out) != 3:
            raise TypeError(
                "linalg_lu(): out must be a tuple of 3 tensors, " f"got {len(out)}"
            )
        return out
    if P is None or L is None or U is None:
        raise TypeError("linalg_lu(): P, L and U must all be provided for out variant")
    return P, L, U


def linalg_lu_out(input, *, pivot=True, P=None, L=None, U=None, out=None):
    logger.debug("GEMS_KUNLUNXIN LINALG_LU_OUT")
    _check_linalg_lu(input, pivot)
    p_out, l_out, u_out = _resolve_linalg_lu_out_args(P, L, U, out)
    return _linalg_lu_impl(input, pivot=pivot, P=p_out, L=l_out, U=u_out)