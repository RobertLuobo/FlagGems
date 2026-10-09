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

import triton
import triton.language as tl
from _kunlunxin.utils.codegen_config_utils import CodeGenConfig

from ..utils.pointwise_dynamic import pointwise_dynamic

logger = logging.getLogger(__name__)

config_ = CodeGenConfig(
    512,
    (65536, 65536, 65536),
    32,
    True,
    prefer_1d_tile=True,
    buffer_size_limit=4096,
    isCloseVectorization=True,
    unroll_num=8,
)


@pointwise_dynamic(
    is_tensor=[True, True, True, False, False],
    promotion_methods=[(0, 1, 2, "DEFAULT")],
    config=config_,
)
@triton.jit
def addr_forward(inp, v1, v2, beta, alpha):
    return beta * inp.to(tl.float32) + alpha * (v1.to(tl.float32) * v2.to(tl.float32))


def addr_(input, vec1, vec2, *, beta=1, alpha=1):
    logger.debug("GEMS_KUNLUNXIN ADDR_")
    if vec1.dim() != 1 or vec2.dim() != 1:
        raise ValueError("addr_: expected 1-D vectors")

    M, N = input.shape
    if vec1.shape[0] != M or vec2.shape[0] != N:
        raise ValueError(
            f"addr_: vec1 size {vec1.shape[0]} must match input rows {M}, "
            f"vec2 size {vec2.shape[0]} must match input cols {N}"
        )

    addr_forward(
        input,
        vec1.reshape(M, 1),
        vec2.reshape(1, N),
        beta,
        alpha,
        out0=input,
    )
    return input
