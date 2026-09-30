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

import copy
import logging

import triton

from ..utils.codegen_config_utils import get_codegen_config
from ..utils.pointwise_dynamic import pointwise_dynamic

logger = logging.getLogger(__name__)

_config = copy.deepcopy(get_codegen_config())
_config.unroll_num = 4


@pointwise_dynamic(promotion_methods=[(0, 1, "DEFAULT")], config=_config)
@triton.jit
def bitwise_left_shift_kernel(a, b):
    return a << b


def bitwise_left_shift(self, other, *, out=None):
    logger.debug("GEMS_KUNLUNXIN BITWISE_LEFT_SHIFT")
    if out is None:
        return bitwise_left_shift_kernel(self, other)
    return bitwise_left_shift_kernel(self, other, out=out)
