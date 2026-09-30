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

import importlib
import logging
import math

import torch
import triton
import triton.language as tl

from flag_gems.runtime.backend._kunlunxin.ops.bmm import bmm
from flag_gems.runtime.backend._kunlunxin.ops.linalg_svd import linalg_svd
from flag_gems.runtime.backend._kunlunxin.ops.mm import mm
from flag_gems.utils import pointwise_dynamic

logger = logging.getLogger(__name__)

_host = importlib.import_module("flag_gems.ops.linalg_polar")


@pointwise_dynamic(is_tensor=[True], promotion_methods=[(0, "DEFAULT")])
@triton.jit
def _finite_or_zero_kernel(value):
    return tl.where((value == value) & (tl.abs(value) < 1e30), value, 0.0)


def _matmul(left, right):
    if left.ndim == 2:
        return mm(left.contiguous(), right.contiguous())
    batch_shape = left.shape[:-2]
    batch = math.prod(batch_shape)
    if batch == 0:
        return torch.empty(
            (*batch_shape, left.shape[-2], right.shape[-1]),
            dtype=left.dtype,
            device=left.device,
        )
    left_3d = left.reshape(batch, left.shape[-2], left.shape[-1]).contiguous()
    right_3d = right.reshape(batch, right.shape[-2], right.shape[-1]).contiguous()
    result = bmm(left_3d, right_3d)
    return result.reshape(*batch_shape, left.shape[-2], right.shape[-1])


def _svd(A):
    U, S, Vh = linalg_svd(A.contiguous(), full_matrices=False)
    Vh = _finite_or_zero_kernel(Vh)
    return U, S, Vh


def _linalg_polar_impl(A, out_U=None, out_H=None):
    _host._validate_input(A)
    n = A.shape[-1]
    U_shape = A.shape
    H_shape = (*A.shape[:-2], n, n)

    if out_U is not None:
        _host._check_out(A, out_U, "U")
        _host._check_out(A, out_H, "H")
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

    Up, S, Vh = _svd(A)
    V = Vh.mH
    U = _matmul(Up, Vh)
    scaled_Vh = _host._scale_rows_kernel(Vh, S.unsqueeze(-1))
    H = _matmul(V, scaled_Vh)
    H = _host._symmetrize_kernel(H, H.mT)
    if out_U is not None:
        _host._copy_kernel(U, out0=out_U)
        _host._copy_kernel(H, out0=out_H)
        return out_U, out_H
    return U.contiguous(), H.contiguous()


def linalg_polar(A):
    logger.debug("GEMS_KUNLUNXIN LINALG_POLAR")
    return _linalg_polar_impl(A)


def linalg_polar_out(A, *, U, H):
    logger.debug("GEMS_KUNLUNXIN LINALG_POLAR_OUT")
    return _linalg_polar_impl(A, U, H)
