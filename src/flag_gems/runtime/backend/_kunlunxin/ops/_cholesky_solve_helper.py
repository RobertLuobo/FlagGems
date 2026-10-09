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

from .cholesky_solve import cholesky_solve

logger = logging.getLogger(__name__)


def _cholesky_solve_helper(
    self: torch.Tensor, A: torch.Tensor, upper: bool
) -> torch.Tensor:
    logger.debug("GEMS_KUNLUNXIN _CHOLESKY_SOLVE_HELPER")
    if self.dtype not in (torch.float32, torch.float64):
        raise RuntimeError(
            "_cholesky_solve_helper currently supports float32 and float64 only"
        )
    if self.dtype != A.dtype or self.device != A.device:
        raise RuntimeError("self and A must have the same dtype and device")
    if self.ndim < 2 or A.ndim < 2 or A.shape[-1] != A.shape[-2]:
        raise RuntimeError("expected a matrix right-hand side and a square factor")
    if self.shape[-2] != A.shape[-1]:
        raise RuntimeError("self and A have incompatible matrix dimensions")

    return cholesky_solve(self, A, upper=upper)
