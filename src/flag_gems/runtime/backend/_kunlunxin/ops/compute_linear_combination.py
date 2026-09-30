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
def _compute_linear_combination_kernel(
    In,  # input of shape (K, N), flattened to 2D
    Coeff,  # coefficients of shape (M, K)
    Out,  # output of shape (M, N), flattened to 2D
    M,
    N,
    K,
    stride_in_k,
    stride_in_n,
    stride_coeff_m,
    stride_coeff_k,
    stride_out_m,
    stride_out_n,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_k = tl.arange(0, BLOCK_K)
    k_mask = offs_k < K

    coeff = tl.load(
        Coeff + pid_m * stride_coeff_m + offs_k * stride_coeff_k,
        mask=k_mask,
        other=0.0,
    ).to(tl.float32)
    inp = tl.load(
        In + offs_k * stride_in_k + pid_n * stride_in_n,
        mask=k_mask,
        other=0.0,
    ).to(tl.float32)

    acc = tl.sum(coeff * inp)
    tl.store(Out + pid_m * stride_out_m + pid_n * stride_out_n, acc.to(Out.dtype.element_ty))


def _compute_linear_combination(input, coefficients, *, out=None):
    logger.debug("GEMS_KUNLUNXIN _COMPUTE_LINEAR_COMBINATION")
    assert input.ndimension() > 0 and input.numel() > 0, "Empty tensor not supported"

    # coefficients is [m, n]; input is [n, ...]
    assert coefficients.dim() == 2, "coefficients must be 2-dimensional"
    m, n = coefficients.shape
    assert input.shape[0] == n, "incompatible dimensions: input and coefficients"

    output_shape = (m,) + tuple(input.shape[1:])

    if out is None:
        out = torch.empty(output_shape, device=input.device, dtype=input.dtype)
    else:
        assert tuple(out.shape) == output_shape, "Incompatible output shape"

    # Flatten the non-contraction dimensions for the kernel.
    in_2d = input.reshape(n, -1)
    out_2d = out.reshape(m, -1)
    N = in_2d.shape[1]

    coefficients = coefficients.contiguous()

    BLOCK_K = triton.next_power_of_2(n)

    grid = (m, N)
    with torch_device_fn.device(input.device):
        _compute_linear_combination_kernel[grid](
            in_2d,
            coefficients,
            out_2d,
            m,
            N,
            n,
            in_2d.stride(0),
            in_2d.stride(1),
            coefficients.stride(0),
            coefficients.stride(1),
            out_2d.stride(0),
            out_2d.stride(1),
            BLOCK_K=BLOCK_K,
        )
    return out


def _compute_linear_combination_out(input, coefficients, *, out=None):
    logger.debug("GEMS_KUNLUNXIN _COMPUTE_LINEAR_COMBINATION OUT")
    return _compute_linear_combination(input, coefficients, out=out)
