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

from .vector_norm import vector_norm

logger = logging.getLogger(__name__)


def norm_except_dim(v, pow=2, dim=0):
    logger.debug("GEMS_KUNLUNXIN NORM_EXCEPT_DIM")

    # aten treats the literal dim == -1 as a sentinel meaning "norm over the
    # whole tensor" (0-dim scalar output), not "the last dim".
    if int(dim) == -1:
        return vector_norm(v, ord=pow)

    ndim = v.dim()
    d = int(dim) % ndim
    reduce_dims = [i for i in range(ndim) if i != d]
    # keepdim leaves the reduced axes as size 1 and keeps dim d at its size,
    # which is exactly aten's out_shape ([1, ..., D at d, ..., 1]).
    return vector_norm(v, ord=pow, dim=reduce_dims, keepdim=True)
