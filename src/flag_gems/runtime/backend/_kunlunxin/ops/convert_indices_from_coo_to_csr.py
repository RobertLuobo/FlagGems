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

import triton
import triton.language as tl


@triton.jit
def _coo_to_csr_lower_bound_kernel(
    idx_ptr,
    out_ptr,
    n,
    size,
    stride,
    steps,
    BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    k = pid * BLOCK_K + tl.arange(0, BLOCK_K)

    lo = tl.zeros([BLOCK_K], dtype=tl.int32)
    # XPU defect 2: an all-false masked load kills the program's stores, and
    # `n == 0` makes `mid < n` false on every lane.  The empty vector needs no
    # search at all -- its lower bound is 0 for every key.
    if n > 0:
        hi = tl.zeros([BLOCK_K], dtype=tl.int32) + n
        for _ in range(steps):
            mid = (lo + hi) >> 1
            m = mid < n
            v = tl.load(idx_ptr + mid * stride, mask=m, other=0)
            c = m & (v.to(tl.int32) < k)
            lo = tl.where(c, mid + 1, lo)
            hi = tl.where(c, hi, mid)

    # XPU defect 1: `mask=k <= size` is lowered as an unmasked full-tile store.
    # The strict comparison against the materialised row count is honoured.
    nrow = size + 1
    tl.store(out_ptr + k, lo.to(out_ptr.dtype.element_ty), mask=k < nrow)


def _patch_generic_kernel():
    import importlib
    import sys

    name = "flag_gems.ops.convert_indices_from_coo_to_csr"
    module = sys.modules.get(name)
    if module is None:
        try:
            module = importlib.import_module(name)
        except ImportError:
            return
    module._coo_to_csr_lower_bound_kernel = _coo_to_csr_lower_bound_kernel


_patch_generic_kernel()


__all__ = ["_coo_to_csr_lower_bound_kernel"]
