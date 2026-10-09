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
import math

import torch

from flag_gems.ops.linalg_polar import (
    _check_out,
    _copy_kernel,
    _scale_rows_kernel,
    _symmetrize_kernel,
    _validate_input,
)

from .bmm import bmm
from .mm import mm
from .svd import svd

logger = logging.getLogger(__name__)


def _batched_matmul(left, right):
    if left.ndim == 2:
        return mm(left, right)

    batch_shape = left.shape[:-2]
    batch = math.prod(batch_shape)
    if batch == 0:
        return torch.empty(
            (*batch_shape, left.shape[-2], right.shape[-1]),
            dtype=left.dtype,
            device=left.device,
        )

    left_3d = left.reshape(batch, left.shape[-2], left.shape[-1])
    right_3d = right.reshape(batch, right.shape[-2], right.shape[-1])
    result = bmm(left_3d, right_3d)
    return result.reshape(*batch_shape, left.shape[-2], right.shape[-1])


def _linalg_polar_impl(A, out_U=None, out_H=None):
    _validate_input(A)
    n = A.shape[-1]
    U_shape = A.shape
    H_shape = (*A.shape[:-2], n, n)

    if out_U is not None:
        _check_out(A, out_U, "U")
        _check_out(A, out_H, "H")
        if out_U.shape != U_shape:
            out_U.resize_(U_shape)
        if out_H.shape != H_shape:
            out_H.resize_(H_shape)

    if A.numel() == 0:
        U = out_U
        H = out_H
        if U is None:
            U = torch.empty_like(A, memory_format=torch.contiguous_format)
            H = torch.empty(H_shape, dtype=A.dtype, device=A.device)
        return U, H

    # A = Up @ diag(S) @ Vh, so the right polar decomposition is
    # U = Up @ Vh and H = Vh^H @ diag(S) @ Vh.  The heavy linear-algebra steps
    # delegate to the vendor svd/mm/bmm Triton kernels; the 2-D-tile tl.dot
    # postprocess kernel from the generic op is intentionally dropped (it OORs
    # SRAM on XPU3).
    Up, S, V = svd(A, some=True, compute_uv=True)
    Vh = V.mH
    U = _batched_matmul(Up, Vh)
    scaled_Vh = _scale_rows_kernel(Vh, S.unsqueeze(-1))
    H = _batched_matmul(V, scaled_Vh)
    H = _symmetrize_kernel(H, H.mT)
    if out_U is not None:
        _copy_kernel(U, out0=out_U)
        _copy_kernel(H, out0=out_H)
        return out_U, out_H
    return U.contiguous(), H.contiguous()


def linalg_polar(A):
    logger.debug("GEMS_KUNLUNXIN LINALG_POLAR")
    return _linalg_polar_impl(A)


def linalg_polar_out(A, *, U, H):
    logger.debug("GEMS_KUNLUNXIN LINALG_POLAR_OUT")
    return _linalg_polar_impl(A, U, H)
