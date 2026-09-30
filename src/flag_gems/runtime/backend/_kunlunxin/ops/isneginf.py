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

from flag_gems.utils import tl_extra_shim

from ..utils.codegen_config_utils import CodeGenConfig
from ..utils.pointwise_dynamic import pointwise_dynamic

logger = logging.getLogger(__name__)
_isinf = tl_extra_shim.isinf

_config = CodeGenConfig(
    512,
    (65536, 65536, 65536),
    32,
    True,
    prefer_1d_tile=True,
    isCloseMemoryAsync=False,
    kunlunAutoGrid=True,
    unroll_num=16,
)


@pointwise_dynamic(promotion_methods=[(0, "ALWAYS_BOOL")], config=_config)
@triton.jit
def isneginf_func(x):
    return x.to(tl.float32) == -float("inf")


# fp16-only fast path: isinf(min(x,0)) lowers to the vectorized isinf extern.
@pointwise_dynamic(promotion_methods=[(0, "ALWAYS_BOOL")], config=_config)
@triton.jit
def isneginf_func_fp16(x):
    return _isinf(tl.minimum(x.to(tl.float32), 0.0))


def _select(dtype):
    return isneginf_func_fp16 if dtype == torch.float16 else isneginf_func


def isneginf(A):
    logger.debug("GEMS_KUNLUNXIN ISNEGINF")
    return _select(A.dtype)(A)


def isneginf_out(A, *, out=None):
    logger.debug("GEMS_KUNLUNXIN ISNEGINF_OUT")
    fn = _select(A.dtype)
    if out is None:
        return fn(A)
    fn(A, out0=out)
    return out
