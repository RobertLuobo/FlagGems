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
from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as ext

from .mm import mm

logger = logging.getLogger(__name__)

SCALE_BLOCK_N = 128


@libentry()
@triton.jit
def _col_scale_kernel(
    C,
    Scales,
    M,
    N,
    stride_cm,
    stride_cn,
    stride_scales,
    BLOCK_N: tl.constexpr,
):
    pid_m = ext.program_id(0)
    pid_n = ext.program_id(1)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    valid = offs_n < N
    # Clamp OOB column indices to 0 before loading: TritonXPU mis-lowers masked
    # loads whose address leaves the allocation (reads neighbour memory), so we
    # read an in-bounds lane and mask the value instead.
    offs_n_safe = tl.where(valid, offs_n, 0)
    s = tl.load(Scales + offs_n_safe * stride_scales)
    s = s.to(tl.float32)
    c_ptr = C + pid_m * stride_cm + offs_n_safe * stride_cn
    cur = tl.load(c_ptr).to(tl.float32)
    out = (cur * s).to(C.dtype.element_ty)
    tl.store(c_ptr, out, mask=valid)


def weight_int8pack_mm(A, B, scales):
    logger.debug("GEMS_KUNLUNXIN WEIGHT_INT8PACK_MM")

    assert (
        A.shape[1] == B.shape[1]
    ), f"incompatible K dimensions: A.shape[1]={A.shape[1]}, B.shape[1]={B.shape[1]}"
    assert (
        B.shape[0] == scales.shape[0]
    ), f"incompatible N dimensions: B.shape[0]={B.shape[0]}, scales.shape[0]={scales.shape[0]}"
    assert B.dtype == torch.int8, f"B must be int8, got {B.dtype}"

    M, K = A.shape
    N = B.shape[0]

    # Dequantize the int8 weight to the activation dtype (pure cast glue), then
    # view as (K, N) for the GEMM. The matmul itself (the compute basis) runs on
    # the vendor Kunlunxin Triton mm kernel, which is the XPU3-safe GEMM path
    # (host-padded, in-bounds-by-construction; no masked OOB loads/stores).
    Bt = B.to(A.dtype).t()

    C = mm(A, Bt)

    # Per-output-column scaling to match the reference (matmul then * scales).
    grid = (M, triton.cdiv(N, SCALE_BLOCK_N))
    with torch_device_fn.device(A.device):
        _col_scale_kernel[grid](
            C,
            scales,
            M,
            N,
            C.stride(0),
            C.stride(1),
            scales.stride(0),
            BLOCK_N=SCALE_BLOCK_N,
        )

    return C
