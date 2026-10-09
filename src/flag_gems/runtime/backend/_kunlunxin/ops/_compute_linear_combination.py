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

from .mm import mm as _gems_mm

logger = logging.getLogger(__name__)


def _compute_linear_combination(input, coefficients, *, out=None):
    logger.debug("GEMS_KUNLUNXIN _COMPUTE_LINEAR_COMBINATION")
    assert input.ndimension() > 0 and input.numel() > 0, "Empty tensor not supported"
    assert coefficients.dim() == 2, "coefficients must be 2-dimensional"
    m, n = coefficients.shape
    assert input.shape[0] == n, "incompatible dimensions: input and coefficients"

    output_shape = (m,) + tuple(input.shape[1:])

    # output[i, ...] = sum_j coefficients[i, j] * input[j, ...]
    #   == coefficients (m, n) @ input.reshape(n, -1) -> (m, N) -> output_shape
    in_2d = input.reshape(n, -1).contiguous()
    coeff_2d = coefficients.contiguous()
    res_2d = _gems_mm(coeff_2d, in_2d)
    result = res_2d.reshape(output_shape)

    if out is None:
        return result
    assert tuple(out.shape) == output_shape, "Incompatible output shape"
    out.copy_(result)
    return out


def _compute_linear_combination_out(input, coefficients, *, out=None):
    logger.debug("GEMS_KUNLUNXIN _COMPUTE_LINEAR_COMBINATION OUT")
    return _compute_linear_combination(input, coefficients, out=out)
