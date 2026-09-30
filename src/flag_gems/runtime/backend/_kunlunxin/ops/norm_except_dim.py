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
import triton
import triton.language as tl

logger = logging.getLogger(__name__)


@triton.jit
def _norm_inner_kernel(
    v_ptr,
    out_ptr,
    inner,
    POW,
    IS_P2: tl.constexpr,
    BLOCK: tl.constexpr,
):
    i = tl.program_id(0)
    base = i * inner
    acc = tl.zeros([BLOCK], tl.float32)
    for q0 in range(0, inner, BLOCK):
        q = q0 + tl.arange(0, BLOCK)
        mask = q < inner
        x = tl.load(v_ptr + base + q, mask=mask, other=0.0).to(tl.float32)
        if IS_P2:
            acc += x * x
        else:
            term = tl.exp(POW * tl.log(tl.abs(x)))
            acc += tl.where(mask, term, 0.0)
    s = tl.sum(acc, axis=0)
    if IS_P2:
        r = tl.sqrt(s)
    else:
        r = tl.exp((1.0 / POW) * tl.log(s))
    tl.store(out_ptr + i, r)


@triton.heuristics(
    values={
        "BLOCK_ROW_SIZE": lambda a: triton.next_power_of_2(
            triton.cdiv(a.get("D", 1), 12)
        ),
        "BLOCK_COL_SIZE": lambda a: 1,
    },
)
@triton.jit
def _norm_general_kernel(
    v_ptr,
    out_ptr,
    pre,
    D,
    inner,
    POW,
    IS_P2: tl.constexpr,
    BLOCK_ROW_SIZE: tl.constexpr,
    BLOCK_COL_SIZE: tl.constexpr,
):
    tid_m = tl.arange(0, BLOCK_ROW_SIZE)[:, None]
    pid = tl.program_id(axis=0) * BLOCK_ROW_SIZE
    row_offset = pid + tid_m
    row_mask = row_offset < D

    tid_n = tl.arange(0, BLOCK_COL_SIZE)[None, :]
    v_block = tl.zeros([BLOCK_ROW_SIZE, BLOCK_COL_SIZE], dtype=tl.float32)
    for base in range(0, pre * inner, BLOCK_COL_SIZE):
        col_offset = base + tid_n
        m_idx = col_offset // inner
        k_idx = col_offset % inner
        mask = (m_idx < pre) and row_mask
        v_offsets = m_idx * D * inner + row_offset * inner + k_idx
        x = tl.load(v_ptr + v_offsets, mask=mask, other=0.0).to(tl.float32)
        if IS_P2:
            v_block += x * x
        else:
            term = tl.exp(POW * tl.log(tl.abs(x)))
            v_block += tl.where(mask, term, 0.0)
    v_sum = tl.sum(v_block, axis=1)[:, None]
    if IS_P2:
        r = tl.sqrt(v_sum)
    else:
        r = tl.exp((1.0 / POW) * tl.log(v_sum))
    tl.store(out_ptr + row_offset, r, mask=row_mask)


def norm_except_dim(v, pow=2, dim=0):
    logger.debug("GEMS_KUNLUNXIN NORM_EXCEPT_DIM")

    assert v.dtype in (
        torch.float16,
        torch.bfloat16,
        torch.float32,
        torch.float64,
    ), f"norm_except_dim: unsupported dtype {v.dtype}"

    v = v.contiguous()
    full_norm = int(dim) == -1
    if full_norm:
        v = v.reshape(1, -1)
    ndim = v.dim()
    d = 0 if full_norm else int(dim) % ndim
    numel = v.numel()
    D = v.size(d)
    inner = 1
    for k in range(d + 1, ndim):
        inner *= v.size(k)
    pre = numel // (D * inner) if D * inner > 0 else 0

    out_shape = [1] * ndim
    out_shape[d] = D

    is_p2 = float(pow) == 2.0
    pval = float(pow)

    if numel == 0:
        acc = torch.zeros(out_shape, device=v.device, dtype=v.dtype)
        return acc.reshape(()) if full_norm else acc

    out = torch.empty(out_shape, device=v.device, dtype=v.dtype)

    if d == 0:
        # pre == 1: each output row reduces its contiguous inner block.
        BLOCK = 1024
        _norm_inner_kernel[(D,)](v, out, inner, pval, is_p2, BLOCK)
    else:
        # d == last (inner == 1) or a middle kept dim: flatten the reduced axes
        # into a single (pre * inner) column loop with a row tile over D.
        grid = lambda META: (triton.cdiv(D, META["BLOCK_ROW_SIZE"]),)
        _norm_general_kernel[grid](v, out, pre, D, inner, pval, is_p2)

    return out.reshape(()) if full_norm else out
