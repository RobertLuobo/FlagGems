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

import importlib
import logging

_generic = importlib.import_module("flag_gems.ops.linalg_matrix_power")

logger = logging.getLogger(__name__)

_generic.SINGLE_TILE_MAX = 0
_generic.TILED_MAX = 0
_generic.GRID_SYNC_MAX = 0
_generic.TRITON_THRESHOLD = 0


def linalg_matrix_power(A, n, *, out=None):
    logger.debug("GEMS_KUNLUNXIN LINALG_MATRIX_POWER")
    return _generic.linalg_matrix_power(A, n, out=out)


def linalg_matrix_power_out(A, n, *, out=None):
    logger.debug("GEMS_KUNLUNXIN LINALG_MATRIX_POWER_OUT")
    return _generic.linalg_matrix_power_out(A, n, out=out)
