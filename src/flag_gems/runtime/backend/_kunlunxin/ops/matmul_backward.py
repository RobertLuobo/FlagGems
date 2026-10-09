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

from .bmm import bmm
from .mm import mm
from .sum import sum_dim

logger = logging.getLogger(__name__)


def _matmul_2d_or_batched(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    if a.dim() == 2 and b.dim() == 2:
        return mm(a.contiguous(), b.contiguous())

    M, K = a.shape[-2], a.shape[-1]
    K2, N = b.shape[-2], b.shape[-1]
    batch_shape = torch.broadcast_shapes(a.shape[:-2], b.shape[:-2])

    a_exp = a.expand(*batch_shape, M, K).reshape(-1, M, K).contiguous()
    b_exp = b.expand(*batch_shape, K2, N).reshape(-1, K2, N).contiguous()
    out = bmm(a_exp, b_exp)
    return out.reshape(*batch_shape, M, N)


def _sum_to(grad: torch.Tensor, shape) -> torch.Tensor:
    shape = tuple(shape)
    while grad.dim() > len(shape):
        grad = sum_dim(grad, dim=[0], keepdim=False)
    for i in range(len(shape)):
        if shape[i] == 1 and grad.shape[i] != 1:
            grad = sum_dim(grad, dim=[i], keepdim=True)
    return grad


def matmul_backward(
    grad: torch.Tensor,
    self: torch.Tensor,
    other: torch.Tensor,
    mask,
):
    logger.debug("GEMS_KUNLUNXIN MATMUL_BACKWARD")

    need_self, need_other = bool(mask[0]), bool(mask[1])

    dim_self = self.dim()
    dim_other = other.dim()

    compute_dtype = self.dtype
    upcast = self.dtype in (torch.float16, torch.bfloat16)
    if upcast:
        compute_dtype = torch.float32

    s = self.to(compute_dtype) if self.dtype != compute_dtype else self
    o = other.to(compute_dtype) if other.dtype != compute_dtype else other
    g = grad.to(compute_dtype) if grad.dtype != compute_dtype else grad

    folded_self = dim_self == 1
    folded_other = dim_other == 1

    if folded_self:
        s = s.unsqueeze(-2)
    if folded_other:
        o = o.unsqueeze(-1)

    if folded_other:
        g = g.unsqueeze(-1)
    if folded_self:
        g = g.unsqueeze(-2)

    grad_self = None
    grad_other = None

    if need_self:
        gs = _matmul_2d_or_batched(g, o.transpose(-2, -1))
        gs = _sum_to(gs, s.shape)
        if folded_self:
            gs = gs.squeeze(-2)
        grad_self = gs.to(self.dtype) if gs.dtype != self.dtype else gs

    if need_other:
        go = _matmul_2d_or_batched(s.transpose(-2, -1), g)
        go = _sum_to(go, o.shape)
        if folded_other:
            go = go.squeeze(-1)
        grad_other = go.to(other.dtype) if go.dtype != other.dtype else go

    return grad_self, grad_other
