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

from torch import Tensor

from .batch_norm import batch_norm_backward

logger = logging.getLogger(__name__)


def _miopen_train_tile_s(spatial_dim):
    """Exact-fit XPU tile policy for the miopen backward 3-stage path (2026-09-04).

    The vendor ``_bn_train_tile_s`` uses a fixed 2048-lane (masked) tile for every
    ``S <= 2048``. Measured on P800 (micro A/B, 2026-09-04): the XPU backend vectorizes
    each program's loads only when at least 4 elements/thread are used (128-bit
    accesses), and a 512/1024-lane *unmasked-or-lightly-masked* tile is 2-10x faster
    than 128/256-lane tiles while being ~15% faster than the 2048-masked tile for the
    S=384/704/1024 shapes that dominate the miopen backward matrix:
      S=384:  187.5us (T=2048) -> 160.3us (T=512)   stats+combine+grad
      S=704:  191.1us (T=2048) -> 172.8us (T=1024)
      S=1024: 185.6us (T=2048) -> 159.7us (T=1024)
    For S > 2048 the pow2-4096 tile remains optimal (unchanged from the vendor policy).
    """
    if spatial_dim <= 0:
        return 1, False
    if spatial_dim <= 512:
        # 256+ S-runs are fully masked at 512 (validity fraction >= 0.5); 128/256-lane
        # tiles force 32-bit (scalar) accesses and cost 2-10x more per element.
        return 512, (spatial_dim % 512) != 0
    if spatial_dim <= 1024:
        return 1024, (spatial_dim % 1024) != 0
    tile = min(triton.next_power_of_2(spatial_dim), 4096)
    return tile, (spatial_dim % tile) != 0


def miopen_batch_norm_backward(
    input: Tensor,
    grad_output: Tensor,
    weight: Tensor,
    running_mean=None,
    running_var=None,
    save_mean=None,
    save_var=None,
    epsilon: float = 1e-05,
) -> tuple:
    """Backward pass for batch normalization (MIOpen variant) on Kunlunxin XPU.

    The MIOpen schema calls the saved inverse standard deviation argument
    ``save_var``. This override delegates to the Kunlunxin ``batch_norm_backward``
    kernel path: the generic implementation in ``flag_gems.ops.miopen_batch_norm_backward``
    relies on the generic ``batch_norm_backward_kernel`` whose 2D-tile lowering fails on
    XPU ("triton_xpu.convert_layout" shape mismatch), while the vendor kernel
    (transposed [N*S, C, 1] view + per-feature grid) compiles and passes the whole
    batch-norm backward matrix.

    Returns:
        Tuple of (grad_input, grad_weight, grad_bias).
    """
    logger.debug("GEMS_KUNLUNXIN MIOPEN_BATCH_NORM_BACKWARD")
    return batch_norm_backward(
        grad_output,
        input,
        weight=weight,
        running_mean=running_mean,
        running_var=running_var,
        save_mean=save_mean,
        save_invstd=save_var,
        train=True,
        eps=epsilon,
        output_mask=(True, True, True),
    )
