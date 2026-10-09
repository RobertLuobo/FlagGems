# Copyright 2026, The FlagOS Contributors.
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

from .linalg_inv_ex import linalg_inv_ex

logger = logging.getLogger(__name__)


def inverse(A):
    """Invert a square matrix (batched), equivalent to ``torch.inverse``.

    The generic Gauss-Jordan kernel keeps a BLOCK x BLOCK augmented tile in
    registers and drives it with 2-D ``tl.gather``/``tl.where`` ops; that tile
    shape fails ``ConvertTritonToTritonXPU`` on xpu3. This overlay routes to the
    backend ``linalg_inv_ex`` (vendor LU factor + gems triangular solve), which
    is the same primitive ``torch.linalg.inv`` is built on and already compiles
    and runs correctly on this device.
    """
    logger.debug("GEMS_KUNLUNXIN INVERSE")

    assert A.dim() >= 2, "inverse: input must be at least 2D"
    assert A.shape[-1] == A.shape[-2], "inverse: input must be a square matrix"
    assert A.dtype in (
        torch.float32,
        torch.float64,
    ), f"inverse: unsupported dtype {A.dtype}"

    return linalg_inv_ex(A).inverse
