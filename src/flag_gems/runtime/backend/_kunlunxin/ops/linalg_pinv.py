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

from .bmm import bmm
from .linalg_svd import linalg_svd

logger = logging.getLogger(__name__)


def linalg_pinv(A, *, atol=None, rtol=None, hermitian=False):
    """Kunlunxin overlay for ``torch.linalg.pinv``.

    The generic ``flag_gems.ops.linalg_pinv`` builds the pseudoinverse from a
    one-sided Jacobi SVD expressed with 2-D ``tl.sum(axis=0)`` reductions and a
    ``tl.dot`` / ``tl.trans`` reconstruction; both fail to lower on XPU3
    (``ConvertTritonXPUToLLVM`` aborts in the blocked ``_jacobi_round_kernel``).

    This overlay instead reuses the vendor ``linalg_svd`` (one-sided Jacobi SVD
    that already compiles and passes on this backend) and the vendor ``bmm`` to
    assemble ``pinv(A) = V @ diag(1/sigma) @ U^H``, thresholding singular values
    with the same ``max(atol, rtol * sigma_max)`` cutoff torch uses.
    """
    logger.debug("GEMS_KUNLUNXIN LINALG_PINV")

    if A.dtype != torch.float32:
        raise NotImplementedError(
            f"linalg_pinv on this device supports float32 only, got {A.dtype}"
        )
    assert A.dim() >= 2, "linalg_pinv: input must be at least 2D"

    if not A.is_contiguous():
        A = A.contiguous()

    orig_shape = A.shape
    m, n = orig_shape[-2], orig_shape[-1]
    batch_shape = orig_shape[:-2]

    Aw = A.reshape(-1, m, n)

    # Reduced SVD: U (b, m, k), S (b, k), Vh (b, k, n), k = min(m, n).
    U, S, Vh = linalg_svd(Aw, full_matrices=False)
    if U.dim() == 2:
        U = U.unsqueeze(0)
        S = S.unsqueeze(0)
        Vh = Vh.unsqueeze(0)

    # cutoff = max(atol, rtol * sigma_max); rtol defaults to max(m, n) * eps and
    # drops to 0 when a positive atol is supplied (torch semantics).
    atol_val = 0.0 if atol is None else float(atol)
    if rtol is None:
        eps = torch.finfo(torch.float32).eps
        rtol_val = 0.0 if atol_val > 0.0 else float(max(m, n)) * eps
    else:
        rtol_val = float(rtol)

    sigma_max = S.amax(dim=-1, keepdim=True)
    cutoff = torch.clamp(sigma_max * rtol_val, min=atol_val)
    inv_s = torch.where(S > cutoff, 1.0 / S, torch.zeros_like(S))

    # pinv = V @ diag(1/sigma) @ U^H = (Vh^T * inv_s) @ U^T  -> (b, n, m)
    V_scaled = (Vh.transpose(-2, -1) * inv_s.unsqueeze(-2)).contiguous()
    Ut = U.transpose(-2, -1).contiguous()
    pinv = bmm(V_scaled, Ut)

    out_shape = list(batch_shape) + [n, m]
    return pinv.reshape(out_shape)
