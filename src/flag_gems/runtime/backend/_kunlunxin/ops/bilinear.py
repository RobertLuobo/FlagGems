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

from flag_gems.runtime import torch_device_fn

logger = logging.getLogger(__name__)


@triton.jit
def bilinear_kernel(
    input1_ptr,
    input2_ptr,
    weight_ptr,
    bias_ptr,
    output_ptr,
    M,
    N,
    K1,
    K2,
    stride_i1m,
    stride_i1k,
    stride_i2m,
    stride_i2k,
    stride_wn,
    stride_wk1,
    stride_wk2,
    stride_om,
    stride_on,
    BIAS: tl.constexpr,
    BLOCK: tl.constexpr,
    BLOCK_K2: tl.constexpr,
):
    """Each program computes a single output element (m, o).

    The (in1_features, in2_features) reduction grid is flattened to a single 1D
    axis so the accumulation is a plain 1D ``tl.sum``. This avoids XPU codegen
    failures triggered by 2D reductions and broadcasted 2D reductions.
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs = tl.arange(0, BLOCK)
    i = offs // BLOCK_K2
    j = offs % BLOCK_K2
    mask_i = i < K1
    mask_j = j < K2

    x1 = tl.load(
        input1_ptr + pid_m * stride_i1m + i * stride_i1k, mask=mask_i, other=0.0
    ).to(tl.float32)
    x2 = tl.load(
        input2_ptr + pid_m * stride_i2m + j * stride_i2k, mask=mask_j, other=0.0
    ).to(tl.float32)
    w = tl.load(
        weight_ptr + pid_n * stride_wn + i * stride_wk1 + j * stride_wk2,
        mask=mask_i & mask_j,
        other=0.0,
    ).to(tl.float32)

    acc = tl.sum(x1 * w * x2)

    if BIAS:
        acc += tl.load(bias_ptr + pid_n).to(tl.float32)

    output = acc.to(output_ptr.dtype.element_ty)
    tl.store(output_ptr + pid_m * stride_om + pid_n * stride_on, output)


def bilinear(input1, input2, weight, bias=None):
    """Applies a bilinear transformation: y = x1^T A x2 + b."""
    logger.debug("GEMS_KUNLUNXIN BILINEAR")

    batch_dims = input1.shape[:-1]
    M = 1
    for dim in batch_dims:
        M *= dim
    K1 = input1.shape[-1]  # in1_features
    K2 = input2.shape[-1]  # in2_features
    N = weight.shape[0]  # out_features

    input1_flat = input1.reshape(M, K1)
    input2_flat = input2.reshape(M, K2)
    weight = weight.contiguous()

    output = torch.empty((M, N), device=input1.device, dtype=input1.dtype)

    BLOCK_K1 = triton.next_power_of_2(K1)
    BLOCK_K2 = triton.next_power_of_2(K2)
    BLOCK = BLOCK_K1 * BLOCK_K2

    grid = (M, N)

    with torch_device_fn.device(input1.device):
        bilinear_kernel[grid](
            input1_flat,
            input2_flat,
            weight,
            bias if bias is not None else weight,
            output,
            M,
            N,
            K1,
            K2,
            input1_flat.stride(0),
            input1_flat.stride(1),
            input2_flat.stride(0),
            input2_flat.stride(1),
            weight.stride(0),
            weight.stride(1),
            weight.stride(2),
            output.stride(0),
            output.stride(1),
            BIAS=bias is not None,
            BLOCK=BLOCK,
            BLOCK_K2=BLOCK_K2,
        )

    output = output.view(*batch_dims, N)
    return output
