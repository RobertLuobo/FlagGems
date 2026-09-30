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

from flag_gems.runtime import device

from .bmm import bmm
from .linalg_svd import _osj_svd_impl

logger = logging.getLogger(__name__)


def pinverse(inp, rcond=1e-15):
    """Moore-Penrose pseudoinverse for the Kunlunxin (XPU3) backend.

    The generic implementation routes through ``flag_gems.ops.svd``, whose
    XPU3 Jacobi/native paths either fail to compile for square ``k`` shapes
    or return numerically wrong factors, and reconstructs via ``tl.dot``
    which is also numerically unreliable on XPU3.  This overlay reuses the
    XPU3-verified one-sided Jacobi SVD from ``linalg_svd`` and the verified
    ``bmm`` kernel to assemble the pseudoinverse.
    """
    logger.debug("GEMS_KUNLUNXIN PINVERSE")

    if inp.ndim < 2:
        raise RuntimeError("pinverse: expected a tensor with at least 2 dimensions")
    if inp.device.type != device.name or inp.dtype != torch.float32:
        raise NotImplementedError(
            f"FlagGems pinverse currently supports only float32 {device.name} tensors"
        )
    if inp.requires_grad:
        raise NotImplementedError(
            "FlagGems pinverse does not yet support autograd inputs"
        )
    if rcond < 0.0:
        raise NotImplementedError(
            "FlagGems pinverse does not yet support negative rcond values"
        )

    m, n = inp.shape[-2:]
    if inp.numel() == 0:
        return torch.empty((*inp.shape[:-2], n, m), dtype=inp.dtype, device=inp.device)

    k = min(m, n)
    batch = inp.numel() // (m * n)
    A = inp.reshape(batch, m, n).contiguous()

    U, S, Vh = _osj_svd_impl(A, full_matrices=False)
    if U.dim() == 2:
        U = U.unsqueeze(0)
        S = S.unsqueeze(0)
        Vh = Vh.unsqueeze(0)
    # U: (batch, m, k)  S: (batch, k) descending  Vh: (batch, k, n)

    largest = S[..., 0:1]
    scaled_tolerance = rcond * largest
    tolerance = torch.where(
        scaled_tolerance > 0.0, scaled_tolerance, torch.zeros_like(scaled_tolerance)
    )
    valid_rcond = rcond == rcond  # False only for NaN rcond
    active = S > tolerance
    if not valid_rcond:
        active = torch.zeros_like(active)
    inv_s = torch.where(active, 1.0 / S, torch.zeros_like(S))

    # pinv = V @ diag(inv_s) @ U^H = (Vh^T * inv_s) @ U^T
    # A rejected singular value can carry a NaN row in ``Vh`` (0 * (1/0) from
    # ``_osj_svd_impl``) or a non-finite column in ``U`` (non-finite inputs).
    # Masking both factors on the inactive columns with ``torch.where`` keeps
    # the discarded directions from leaking NaN through the matmul.
    active_col = active.unsqueeze(-2)
    scaled_v = Vh.transpose(-2, -1) * inv_s.unsqueeze(-2)  # (batch, n, k)
    scaled_v = torch.where(active_col, scaled_v, torch.zeros_like(scaled_v))
    masked_u = torch.where(active_col, U, torch.zeros_like(U))  # (batch, m, k)
    output = bmm(scaled_v.contiguous(), masked_u.transpose(-2, -1).contiguous())

    return output.reshape((*inp.shape[:-2], n, m)).contiguous()
