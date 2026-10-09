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

from .addmm import addmm
from .bmm import bmm
from .mm import mm

logger = logging.getLogger(__name__)


def bilinear(input1, input2, weight, bias=None):
    """Applies a bilinear transformation: y[m, o] = sum_{i, j} x1[m, i] * W[o, i, j] * x2[m, j] + b[o].

    Routed through the vendor GEMM kernels (bmm/mm/addmm) because the generic
    element-wise kernel builds a 2-D tile and performs an all-axis ``tl.sum``,
    which fails TritonXPULegalize / ConvertTritonXPUToLLVM on XPU3.

    The computation is split into:
      1. outer product  outer[m, i, j] = x1[m, i] * x2[m, j]   (vendor bmm, K=1)
      2. contraction    out[m, o]      = sum_{i, j} outer[m, i, j] * W[o, i, j]
                                       = outer_flat @ W_flat^T  (vendor mm/addmm)
    Accumulation is kept in fp32 (fp16/bf16 inputs are upcast then cast back).
    """
    logger.debug("GEMS_KUNLUNXIN BILINEAR")

    batch_dims = input1.shape[:-1]
    K1 = input1.shape[-1]  # in1_features
    K2 = input2.shape[-1]  # in2_features
    N = weight.shape[0]  # out_features

    M = 1
    for dim in batch_dims:
        M *= dim

    orig_dtype = input1.dtype
    compute_dtype = (
        torch.float32
        if orig_dtype in (torch.float16, torch.bfloat16)
        else orig_dtype
    )

    x1 = input1.reshape(M, K1).to(compute_dtype)
    x2 = input2.reshape(M, K2).to(compute_dtype)
    w_flat = weight.reshape(N, K1 * K2).to(compute_dtype)

    # Step 1: per-row outer product via vendor bmm -> (M, K1, K2).
    outer = bmm(x1.reshape(M, K1, 1), x2.reshape(M, 1, K2))
    outer_flat = outer.reshape(M, K1 * K2)

    # Step 2: single contraction over (i, j) via vendor mm/addmm.
    if bias is not None:
        out = addmm(bias.to(compute_dtype), outer_flat, w_flat.t())
    else:
        out = mm(outer_flat, w_flat.t())

    out = out.to(orig_dtype).reshape(*batch_dims, N)
    return out
