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

from flag_gems.utils import libentry

logger = logging.getLogger(__name__)

# One program per (query, head, batch) row; the row-wise running max/sum are
# 0-d (scalar) values so the XPU `TritonXPUCoreTiling` pipeline sees only
# {scalar <-> [BLOCK_K]} broadcasts -- the pattern validated by the vendor
# `softmax`/`log_softmax` kernels. The previous vectorized online-rescale
# (`m_old - m` on [BLOCK_Q] vectors, 2D tiles loaded from bf16/fp16 memory)
# intermittently fails to compile on this backend ("arith.subf op requires the
# same encoding" in TritonXPUCoreTiling) and the autotuned launch then
# silently leaves the output buffer uninitialized (all-zeros), so autotuning
# was dropped in favour of a fixed tile width.
_BLOCK_K = 1024


@libentry()
@triton.jit
def scaled_softmax_forward_kernel(
    output_ptr,
    input_ptr,
    scale_factor,
    query_seq_len,
    key_seq_len,
    stride_b,
    stride_h,
    stride_q,
    BLOCK_K: tl.constexpr,
):
    q_idx = tl.program_id(0)
    h_idx = tl.program_id(1)
    b_idx = tl.program_id(2)

    row_ptr = (
        input_ptr
        + b_idx * stride_b
        + h_idx * stride_h
        + q_idx * stride_q
    )
    k_offsets = tl.arange(0, BLOCK_K)

    m = -float("inf")
    z = 0.0
    for k0 in range(0, tl.cdiv(key_seq_len, BLOCK_K)):
        offs = k0 * BLOCK_K + k_offsets
        mask = offs < key_seq_len
        x = tl.load(row_ptr + offs, mask=mask, other=-float("inf")) * scale_factor
        m_new = tl.max(x, 0)
        m_c = tl.maximum(m, m_new)
        z = z * tl.exp(m - m_c) + tl.sum(tl.where(mask, tl.exp(x - m_c), 0.0), 0)
        m = m_c

    inv = 1.0 / z
    out_row_ptr = (
        output_ptr
        + b_idx * stride_b
        + h_idx * stride_h
        + q_idx * stride_q
    )
    for k0 in range(0, tl.cdiv(key_seq_len, BLOCK_K)):
        offs = k0 * BLOCK_K + k_offsets
        mask = offs < key_seq_len
        x = tl.load(row_ptr + offs, mask=mask, other=-float("inf")) * scale_factor
        tl.store(out_row_ptr + offs, tl.exp(x - m) * inv, mask=mask)


def scaled_softmax_forward(input_t: torch.Tensor, scale_factor: float):
    logger.debug("GEMS_KUNLUNXIN SCALED_SOFTMAX_FORWARD")
    assert input_t.dim() == 4, "expected 4D tensor"
    batch_size, attn_heads, query_seq_len, key_seq_len = input_t.shape
    assert input_t.dtype in [
        torch.float16,
        torch.bfloat16,
    ], "Only fp16 and bf16 are supported"
    assert key_seq_len <= 16384, "Key sequence length must be 16384 or less"
    assert key_seq_len % 8 == 0, "Key sequence length must be divisible by 8"
    assert query_seq_len > 1, "Query sequence length must be greater than 1"

    output_t = torch.empty_like(input_t)
    grid = (query_seq_len, attn_heads, batch_size)
    scaled_softmax_forward_kernel[grid](
        output_t,
        input_t,
        scale_factor,
        query_seq_len,
        key_seq_len,
        input_t.stride(0),
        input_t.stride(1),
        input_t.stride(2),
        BLOCK_K=_BLOCK_K,
        num_warps=4,
        num_stages=2,
    )
    return output_t


@libentry()
@triton.jit
def scaled_softmax_backward_kernel(
    grad_input_ptr,
    grad_output_ptr,
    output_ptr,
    scale_factor,
    query_seq_len,
    key_seq_len,
    stride_b,
    stride_h,
    stride_q,
    BLOCK_K: tl.constexpr,
):
    q_idx = tl.program_id(0)
    h_idx = tl.program_id(1)
    b_idx = tl.program_id(2)

    P_row = (
        output_ptr
        + b_idx * stride_b
        + h_idx * stride_h
        + q_idx * stride_q
    )
    dP_row = (
        grad_output_ptr
        + b_idx * stride_b
        + h_idx * stride_h
        + q_idx * stride_q
    )
    k_offsets = tl.arange(0, BLOCK_K)

    d = 0.0
    for k0 in range(0, tl.cdiv(key_seq_len, BLOCK_K)):
        offs = k0 * BLOCK_K + k_offsets
        mask = offs < key_seq_len
        P = tl.load(P_row + offs, mask=mask, other=0.0)
        dP = tl.load(dP_row + offs, mask=mask, other=0.0)
        d += tl.sum(tl.where(mask, P * dP, 0.0), 0)

    dS_row = (
        grad_input_ptr
        + b_idx * stride_b
        + h_idx * stride_h
        + q_idx * stride_q
    )
    for k0 in range(0, tl.cdiv(key_seq_len, BLOCK_K)):
        offs = k0 * BLOCK_K + k_offsets
        mask = offs < key_seq_len
        P = tl.load(P_row + offs, mask=mask, other=0.0)
        dP = tl.load(dP_row + offs, mask=mask, other=0.0)
        dS = scale_factor * P * (dP - d)
        tl.store(dS_row + offs, dS, mask=mask)


def scaled_softmax_backward(
    grad_output: torch.Tensor, softmax_results: torch.Tensor, scale_factor: float
):
    logger.debug("GEMS_KUNLUNXIN SCALED_SOFTMAX_BACKWARD")
    assert grad_output.dim() == 4, "expected 4D tensor"
    assert softmax_results.dim() == 4, "expected 4D tensor"
    assert grad_output.dtype in [
        torch.float16,
        torch.bfloat16,
    ], "Only fp16 and bf16 are supported"
    assert softmax_results.dtype in [
        torch.float16,
        torch.bfloat16,
    ], "Only fp16 and bf16 are supported"

    grad_output = grad_output.contiguous()
    softmax_results = softmax_results.contiguous()

    batch_size, attn_heads, query_seq_len, key_seq_len = softmax_results.shape

    grad_input = torch.empty_like(grad_output)
    grid = (query_seq_len, attn_heads, batch_size)
    scaled_softmax_backward_kernel[grid](
        grad_input,
        grad_output,
        softmax_results,
        scale_factor,
        query_seq_len,
        key_seq_len,
        softmax_results.stride(0),
        softmax_results.stride(1),
        softmax_results.stride(2),
        BLOCK_K=_BLOCK_K,
        num_warps=4,
        num_stages=2,
    )

    return grad_input
