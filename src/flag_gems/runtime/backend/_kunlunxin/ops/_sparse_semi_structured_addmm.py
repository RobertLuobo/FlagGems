# Copyright 2026, The FlagOS Contributors.
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
from typing import Optional

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn

from ._sparse_semi_structured_mm import _sparse_gate_kernel
from .mm import mm as _gems_mm

logger = logging.getLogger(__name__)


@triton.jit
def _affine_epilogue_kernel(
    MM,
    Inp,
    O,
    M,
    N,
    alpha,
    beta,
    stride_mm_m,
    stride_mm_n,
    stride_im,
    stride_in,
    stride_om,
    stride_on,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Compute O = alpha * MM + beta * Inp over 2D tiles.

    Pure 2D elementwise tiles with masked load/store (no tl.sum / tl.dot),
    so it lowers cleanly on TritonXPU without SRAM over-allocation.
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)

    mm_vals = tl.load(
        MM + offs_m[:, None] * stride_mm_m + offs_n[None, :] * stride_mm_n,
        mask=mask,
        other=0.0,
    )
    inp_vals = tl.load(
        Inp + offs_m[:, None] * stride_im + offs_n[None, :] * stride_in,
        mask=mask,
        other=0.0,
    )
    out = mm_vals * alpha + inp_vals * beta
    tl.store(
        O + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on,
        out,
        mask=mask,
    )


def _sparse_semi_structured_addmm(
    input_tensor: torch.Tensor,
    mat1: torch.Tensor,
    mat1_meta: torch.Tensor,
    mat2: torch.Tensor,
    *,
    alpha: float = 1.0,
    beta: float = 1.0,
    out_dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """Sparse (2:4) semi-structured addmm, XPU-safe.

    Computes: alpha * (mat1 @ sparse(mat1_meta)) @ mat2 + beta * input_tensor

    Mirrors the vendor ``_sparse_semi_structured_mm`` overlay: a 2D gate kernel
    materialises the 2:4-masked ``mat1`` (True keeps the first pair of each
    group of 4, False keeps the second pair), the dense product is delegated to
    the existing vendor ``mm`` kernel, and a 2D elementwise epilogue applies the
    affine ``alpha``/``beta`` scaling. This avoids the generic kernel's rank-3
    broadcast + ``tl.sum`` reduction which over-allocates uni_sram on XPU3.
    """
    logger.debug("GEMS_KUNLUNXIN SPARSE_SEMI_STRUCTURED_ADDMM")

    M = mat1.shape[0]
    K4 = mat1_meta.shape[1]
    N = mat2.shape[1]

    assert mat1.shape == (
        M,
        4 * K4,
    ), f"Expected mat1 shape ({M}, {4 * K4}), got {mat1.shape}"
    assert mat2.shape == (
        4 * K4,
        N,
    ), f"Expected mat2 shape ({4 * K4}, {N}), got {mat2.shape}"
    assert mat1_meta.shape == (
        M,
        K4,
    ), f"Expected mat1_meta shape ({M}, {K4}), got {mat1_meta.shape}"
    assert input_tensor.shape == (
        M,
        N,
    ), f"Expected input_tensor shape ({M}, {N}), got {input_tensor.shape}"

    output_dtype = out_dtype if out_dtype is not None else mat1.dtype

    masked = torch.empty((M, 4 * K4), device=mat1.device, dtype=mat1.dtype)

    BLOCK_M = 32
    BLOCK_K = 32
    gate_grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(K4, BLOCK_K))
    with torch_device_fn.device(mat1.device):
        _sparse_gate_kernel[gate_grid](
            mat1,
            mat1_meta,
            masked,
            M,
            K4,
            mat1.stride(0),
            mat1.stride(1),
            mat1_meta.stride(0),
            mat1_meta.stride(1),
            masked.stride(0),
            masked.stride(1),
            BLOCK_M=BLOCK_M,
            BLOCK_K=BLOCK_K,
        )

    mm_out = _gems_mm(masked, mat2)

    out = torch.empty((M, N), device=mat1.device, dtype=output_dtype)

    BLOCK_EM = 32
    BLOCK_EN = 32
    epi_grid = (triton.cdiv(M, BLOCK_EM), triton.cdiv(N, BLOCK_EN))
    with torch_device_fn.device(mat1.device):
        _affine_epilogue_kernel[epi_grid](
            mm_out,
            input_tensor,
            out,
            M,
            N,
            alpha,
            beta,
            mm_out.stride(0),
            mm_out.stride(1),
            input_tensor.stride(0),
            input_tensor.stride(1),
            out.stride(0),
            out.stride(1),
            BLOCK_M=BLOCK_EM,
            BLOCK_N=BLOCK_EN,
        )

    return out
