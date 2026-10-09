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

from flag_gems.runtime import torch_device_fn

from .rms_norm import (
    rms_norm_grad_dw_kernel,
    rms_norm_grad_dx_kernel,
    rms_norm_grad_dx_kernel_tile,
)

logger = logging.getLogger(__name__)


def _fused_rms_norm_backward(
    grad_out, input, normalized_shape, rstd, weight=None, output_mask=(True, True)
):
    """Kunlunxin override of ``aten::_fused_rms_norm_backward``.

    The generic implementation uses a single large row load for dx (fails for
    N > 64*128 on XPU3) and a ROW_BLOCK_SIZE=16 ``tl.sum(axis=0)`` 2D-tile reduce
    for dw (the batch-dim tile reduce is miscompiled on XPU3). This reuses the
    XPU3-safe kernels already in rms_norm.py: a chunked-loop dx kernel for large
    N and a ROW_BLOCK_SIZE=1 dw kernel whose cross-row accumulate is finished
    with torch.sum, matching the sibling rms_norm_backward.
    """
    logger.debug("GEMS_KUNLUNXIN _FUSED_RMS_NORM_BACKWARD")

    if len(output_mask) != 2:
        raise ValueError("output_mask must contain two booleans")
    if output_mask[1] and weight is None:
        raise RuntimeError("weight gradient requested without a weight tensor")

    eps = 1e-5
    dim = input.ndim - len(normalized_shape)
    M = math.prod(input.shape[:dim])
    N = math.prod(normalized_shape)

    x = input.contiguous()
    dy = grad_out.contiguous()
    inv_rms = rstd.contiguous()

    dx = None
    if output_mask[0]:
        # dx needs a weight pointer; synthesize unit weights when absent
        # (allocation + fill only, mirrors the no-weight forward path).
        w = weight.contiguous() if weight is not None else torch.ones(
            N, dtype=x.dtype, device=x.device
        )
        dx = torch.empty_like(x)
        BLOCK_SIZE = triton.next_power_of_2(N)
        with torch_device_fn.device(x.device):
            if N > 64 * 128:
                BLOCK_SIZE = 8192
                rms_norm_grad_dx_kernel_tile[M,](
                    x,
                    dy,
                    inv_rms,
                    dx,
                    w,
                    N,
                    1,
                    N,
                    1,
                    N,
                    eps,
                    BLOCK_SIZE,
                    isCloseUnrollControl=True,
                    isCloseVectorization=True,
                )
            else:
                rms_norm_grad_dx_kernel[M,](
                    x,
                    dy,
                    inv_rms,
                    dx,
                    w,
                    N,
                    1,
                    N,
                    1,
                    N,
                    eps,
                    BLOCK_SIZE,
                    isCloseUnrollControl=True,
                )

    dw = None
    if output_mask[1] and weight is not None:
        # ROW_BLOCK_SIZE=1 keeps the per-program tile reduce trivial; the real
        # reduction over rows is finished by torch.sum (2-stage reduction).
        ROW_BLOCK_SIZE = 1
        COL_BLOCK_SIZE = 256
        row_block_num = triton.cdiv(M, ROW_BLOCK_SIZE)
        col_block_num = triton.cdiv(N, COL_BLOCK_SIZE)

        partial_buffer = torch.empty(
            (row_block_num, N), dtype=torch.float32, device=x.device
        )

        with torch_device_fn.device(x.device):
            rms_norm_grad_dw_kernel[row_block_num, col_block_num](
                x,
                dy,
                inv_rms,
                partial_buffer,
                N,
                1,
                N,
                1,
                M,
                N,
                ROW_BLOCK_SIZE,
                COL_BLOCK_SIZE,
                isCloseUnrollControl=True,
                isCloseCoreTiling=True,
            )
            dw = (
                torch.sum(partial_buffer, dim=0, dtype=torch.float32)
                .to(x.dtype)
                .reshape(normalized_shape)
            )

    return dx, dw
