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
from typing import Optional, Tuple

from torch import Tensor

from .native_batch_norm import native_batch_norm

logger = logging.getLogger(__name__)


def miopen_batch_norm(
    input: Tensor,
    weight: Optional[Tensor],
    bias: Optional[Tensor],
    running_mean: Optional[Tensor],
    running_var: Optional[Tensor],
    training: bool,
    exponential_average_factor: float,
    epsilon: float,
) -> Tuple[Tensor, Tensor, Tensor]:
    """aten::miopen_batch_norm on the Kunlunxin 1D-tile batch-norm kernels.

    The generic ``batch_norm`` forward kernel (a single 2D ``[BLOCK_M, BLOCK_N]``
    online-accumulator loop with ``BLOCK_M * BLOCK_N`` up to 16384) fails to
    lower on XPU3 (``TritonXPUUnrollControl`` ``out of resource: uni_sram``).
    The vendor ``native_batch_norm`` implements the identical math with fixed,
    bounded 1D ``TILE_S`` stats/normalize kernels that compile, so this override
    reuses it. ``exponential_average_factor`` maps to ``momentum`` and returns
    ``(output, save_mean, save_inv_std)`` as miopen expects.
    """
    logger.debug("GEMS_KUNLUNXIN MIOPEN_BATCH_NORM")

    return native_batch_norm(
        input,
        weight,
        bias,
        running_mean,
        running_var,
        bool(training),
        exponential_average_factor,
        epsilon,
    )
