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
import triton
import triton.language as tl

from flag_gems.ops._weight_norm_differentiable_backward import _composite_backward
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as ext

logger = logging.getLogger(__name__)


def _heur_row_first(args):
    if 1024 <= args["N"] <= 8192:
        return 1
    return triton.next_power_of_2(triton.cdiv(args["M"], 12))


def _heur_col_first(args):
    if 1024 <= args["N"] <= 8192:
        return triton.next_power_of_2(args["N"])
    return 1


@libentry()
@triton.heuristics(
    values={
        "BLOCK_ROW_SIZE": _heur_row_first,
        "BLOCK_COL_SIZE": _heur_col_first,
    },
)
@triton.jit
def _wnb_first_kernel(
    grad_v,
    grad_g,
    grad_w,
    saved_v,
    saved_g,
    saved_norms,
    M,
    N,
    BLOCK_ROW_SIZE: tl.constexpr,
    BLOCK_COL_SIZE: tl.constexpr,
):
    ty = tl.arange(0, BLOCK_ROW_SIZE)[:, None]
    by = ext.program_id(axis=0) * BLOCK_ROW_SIZE
    row = by + ty
    row_mask = row < M

    g = tl.load(saved_g + row, mask=row_mask).to(tl.float32)
    norm = tl.load(saved_norms + row, mask=row_mask).to(tl.float32)

    tx = tl.arange(0, BLOCK_COL_SIZE)[None, :]
    acc = tl.zeros([BLOCK_ROW_SIZE, BLOCK_COL_SIZE], dtype=tl.float32)
    for base in range(0, N, BLOCK_COL_SIZE):
        col = base + tx
        mask = col < N and row_mask
        v = tl.load(saved_v + row * N + col, mask=mask).to(tl.float32)
        w = tl.load(grad_w + row * N + col, mask=mask).to(tl.float32)
        acc += v * w
    dot = tl.sum(acc, axis=1)[:, None]

    scale = g / norm
    projection = dot / (norm * norm)
    for base in range(0, N, BLOCK_COL_SIZE):
        col = base + tx
        mask = col < N and row_mask
        v = tl.load(saved_v + row * N + col, mask=mask).to(tl.float32)
        w = tl.load(grad_w + row * N + col, mask=mask).to(tl.float32)
        tl.store(grad_v + row * N + col, scale * (w - v * projection), mask=mask)
    tl.store(grad_g + row, dot / norm, mask=row_mask)


def _heur_col_last(args):
    return triton.next_power_of_2(triton.cdiv(args["N"], 12))


@libentry()
@triton.heuristics(
    values={
        "BLOCK_ROW_SIZE": lambda args: 1,
        "BLOCK_COL_SIZE": _heur_col_last,
    },
)
@triton.jit
def _wnb_last_kernel(
    grad_v,
    grad_g,
    grad_w,
    saved_v,
    saved_g,
    saved_norms,
    M,
    N,
    BLOCK_ROW_SIZE: tl.constexpr,
    BLOCK_COL_SIZE: tl.constexpr,
):
    tx = tl.arange(0, BLOCK_COL_SIZE)[:, None]
    bx = ext.program_id(axis=0) * BLOCK_COL_SIZE
    col = bx + tx
    col_mask = col < N

    g = tl.load(saved_g + col, mask=col_mask).to(tl.float32)
    norm = tl.load(saved_norms + col, mask=col_mask).to(tl.float32)

    ty = tl.arange(0, BLOCK_ROW_SIZE)[None, :]
    acc = tl.zeros([BLOCK_COL_SIZE, BLOCK_ROW_SIZE], dtype=tl.float32)
    for base in range(0, M, BLOCK_ROW_SIZE):
        row = base + ty
        mask = row < M and col_mask
        v = tl.load(saved_v + row * N + col, mask=mask).to(tl.float32)
        w = tl.load(grad_w + row * N + col, mask=mask).to(tl.float32)
        acc += v * w
    dot = tl.sum(acc, axis=1)[:, None]

    scale = g / norm
    projection = dot / (norm * norm)
    for base in range(0, M, BLOCK_ROW_SIZE):
        row = base + ty
        mask = row < M and col_mask
        v = tl.load(saved_v + row * N + col, mask=mask).to(tl.float32)
        w = tl.load(grad_w + row * N + col, mask=mask).to(tl.float32)
        tl.store(grad_v + row * N + col, scale * (w - v * projection), mask=mask)
    tl.store(grad_g + col, dot / norm, mask=col_mask)


def weight_norm_differentiable_backward(grad_w, saved_v, saved_g, saved_norms, dim):
    logger.debug("GEMS_KUNLUNXIN _WEIGHT_NORM_DIFFERENTIABLE_BACKWARD")

    for name, tensor in (
        ("grad_w", grad_w),
        ("saved_v", saved_v),
        ("saved_g", saved_g),
        ("saved_norms", saved_norms),
    ):
        if not tensor.is_contiguous():
            raise RuntimeError(f"{name} must be contiguous")
    if dim != 0 and dim != saved_v.ndim - 1:
        raise RuntimeError("Expected dim to be the first or last dimension")
    if saved_v.ndim == 0:
        raise IndexError("Dimension specified as -1 but tensor has no dimensions")

    broadcast_shape = [1] * saved_v.ndim
    broadcast_shape[dim] = saved_v.shape[dim]
    expected_norm_dtype = (
        torch.float32
        if saved_v.dtype in (torch.float16, torch.bfloat16)
        else saved_v.dtype
    )
    can_fuse = (
        not (
            torch.is_grad_enabled()
            and any(x.requires_grad for x in (grad_w, saved_v, saved_g, saved_norms))
        )
        and saved_v.dtype in (torch.float16, torch.bfloat16, torch.float32)
        and grad_w.dtype == saved_v.dtype
        and saved_g.dtype == saved_v.dtype
        and saved_norms.dtype == expected_norm_dtype
        and grad_w.shape == saved_v.shape
        and list(saved_g.shape) == broadcast_shape
        and list(saved_norms.shape) == broadcast_shape
        and grad_w.device == saved_v.device == saved_g.device == saved_norms.device
        and saved_v.numel() > 0
    )
    if not can_fuse:
        return _composite_backward(grad_w, saved_v, saved_g, saved_norms, dim)

    grad_v = torch.empty_like(saved_v)
    grad_g = torch.empty_like(saved_g)
    with torch_device_fn.device(saved_v.device):
        if dim == 0:
            M = saved_v.shape[0]
            N = math.prod(saved_v.shape[1:])
            grid = lambda META: (triton.cdiv(M, META["BLOCK_ROW_SIZE"]),)
            _wnb_first_kernel[grid](
                grad_v, grad_g, grad_w, saved_v, saved_g, saved_norms, M, N
            )
        else:
            N = saved_v.shape[-1]
            M = math.prod(saved_v.shape[:-1])
            grid = lambda META: (triton.cdiv(N, META["BLOCK_COL_SIZE"]),)
            _wnb_last_kernel[grid](
                grad_v, grad_g, grad_w, saved_v, saved_g, saved_norms, M, N
            )
    return grad_v, grad_g
