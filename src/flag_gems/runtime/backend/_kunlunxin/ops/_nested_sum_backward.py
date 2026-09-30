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

logger = logging.getLogger(__name__)


def _nested_sum_backward(grad, self, dim=None, keepdim=False):
    logger.debug("GEMS_KUNLUNXIN _NESTED_SUM_BACKWARD")

    shape = list(self.shape)
    ndim = self.dim()

    if dim is None:
        return torch.broadcast_to(grad, shape).contiguous()

    d = int(dim[0]) if isinstance(dim, (list, tuple)) else int(dim)
    if d < 0:
        d += ndim

    if not keepdim:
        grad = grad.unsqueeze(d)

    return grad.expand(shape).contiguous()
