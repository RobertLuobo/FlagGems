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

# Kunlunxin (XPU) override of native_dropout_backward.
#
# The generic adapter (src/flag_gems/ops/native_dropout_backward.py) does
# `from flag_gems.ops.dropout import dropout_backward` at module-import time,
# so it always binds the GENERIC dropout_backward (heuristic BLOCK<=1024,
# grid = cdiv(N, 1024)) => hundreds of thousands of tiny programs on large
# shapes -> launch-bound at 0.01-0.06x speedup. The vendor-optimized
# _kunlunxin.ops.dropout.dropout_backward (1-D tiles up to 131072 elems,
# int8-view mask load, NEED_MASK constexpr branch) was unreachable through
# aten::native_dropout_backward. This override routes the ATen op to the
# vendor-optimized kernel.
import logging

from .dropout import dropout_backward

logger = logging.getLogger(__name__)


def native_dropout_backward(grad_output, mask, scale):
    """Canonical adapter for aten::native_dropout_backward (XPU)."""
    logger.debug("GEMS_KUNLUNXIN NATIVE_DROPOUT_BACKWARD")
    return dropout_backward(grad_output, mask, scale)
