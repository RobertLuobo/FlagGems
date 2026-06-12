import logging
import os

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)


@triton.jit
def moe_sum_kernel(
    input_ptr,
    output_ptr,
    topk: tl.constexpr,
    hidden_size: tl.constexpr,
    input_stride_token: tl.constexpr,
    input_stride_topk: tl.constexpr,
    input_stride_hidden: tl.constexpr,
    output_stride_token: tl.constexpr,
    output_stride_hidden: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    token_idx = tl.program_id(0)
    block_idx = tl.program_id(1)
    hidden_start = block_idx * BLOCK_SIZE
    hidden_offsets = hidden_start + tl.arange(0, BLOCK_SIZE)
    hidden_mask = hidden_offsets < hidden_size
    input_base = input_ptr + token_idx * input_stride_token
    hidden_ptr_offsets = hidden_offsets * input_stride_hidden

    acc = tl.load(input_base + hidden_ptr_offsets, mask=hidden_mask, other=0.0).to(
        tl.float32
    )
    expert_ptr = input_base + input_stride_topk
    acc = acc + tl.load(
        expert_ptr + hidden_ptr_offsets, mask=hidden_mask, other=0.0
    ).to(tl.float32)

    if topk == 6:
        expert_ptr = input_base + 2 * input_stride_topk
        acc = acc + tl.load(
            expert_ptr + hidden_ptr_offsets, mask=hidden_mask, other=0.0
        ).to(tl.float32)
        expert_ptr = input_base + 3 * input_stride_topk
        acc = acc + tl.load(
            expert_ptr + hidden_ptr_offsets, mask=hidden_mask, other=0.0
        ).to(tl.float32)
        expert_ptr = input_base + 4 * input_stride_topk
        acc = acc + tl.load(
            expert_ptr + hidden_ptr_offsets, mask=hidden_mask, other=0.0
        ).to(tl.float32)
        expert_ptr = input_base + 5 * input_stride_topk
        acc = acc + tl.load(
            expert_ptr + hidden_ptr_offsets, mask=hidden_mask, other=0.0
        ).to(tl.float32)

    output_ptr_pos = (
        output_ptr
        + token_idx * output_stride_token
        + hidden_offsets * output_stride_hidden
    )

    tl.store(
        output_ptr_pos,
        acc.to(output_ptr.dtype.element_ty),
        mask=hidden_mask,
    )


def moe_sum(
    input: torch.Tensor,
    output: torch.Tensor,
):
    logger.debug("GEMS MOE SUM")
    num_tokens, topk, hidden_size = input.shape
    input_strides = input.stride()
    output_strides = output.stride()
    grid = lambda meta: (num_tokens, triton.cdiv(hidden_size, meta["BLOCK_SIZE"]))
    os.environ["TRITONXPU_STORE_MASK_SIM"] = "1"
    try:
        moe_sum_kernel[grid](
            input,
            output,
            topk,
            hidden_size,
            input_strides[0],
            input_strides[1],
            input_strides[2],
            output_strides[0],
            output_strides[1],
            BLOCK_SIZE=512,
        )
    finally:
        if "TRITONXPU_STORE_MASK_SIM" in os.environ:
            del os.environ["TRITONXPU_STORE_MASK_SIM"]
