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

logger = logging.getLogger(__name__)

# The vendor XPU build ships `aten::_linalg_eigvals` as an eigenvalues-only
# kernel (its eager-fallback path).  The previous implementation went through
# ``torch.linalg.eig(inp).eigenvalues``, which makes the vendor additionally
# compute the full eigenvector matrix (``linalg_eig_fallback_cpu``) -- strictly
# more work for the same eigenvalues, costing ~1.3-2x the latency.  Instead we
# fetch the vendor kernel handle for the CUDA/XPU dispatch key once at import
# time -- before any ``use_gems()`` registration can shadow it -- and invoke it
# from inside our override.  The handle keeps pointing at the vendor kernel
# even while our own impl is registered on the same key, so the eigenvalues
# are computed exactly once, with the same (vendor) path the torch reference
# uses.
_linalg_eigvals_vendor_kernel = None
_linalg_eigvals_ks = None
try:
    _linalg_eigvals_vendor_kernel = torch.library.get_kernel(
        torch.ops.aten._linalg_eigvals.default, "CUDA"
    )
    _linalg_eigvals_ks = torch._C.DispatchKeySet(torch._C.DispatchKey.CUDA)
except Exception:  # noqa: BLE001 - keep historical behavior if unavailable
    _linalg_eigvals_vendor_kernel = None
    _linalg_eigvals_ks = None


def _linalg_eigvals(inp):
    logger.debug("GEMS_KUNLUNXIN _LINALG_EIGVALS %s %s", inp.shape, inp.dtype)
    if _linalg_eigvals_vendor_kernel is not None:
        return _linalg_eigvals_vendor_kernel.call_boxed(_linalg_eigvals_ks, inp)
    # Fallback: full eig with eigenvectors, then drop them.
    return torch.linalg.eig(inp).eigenvalues
