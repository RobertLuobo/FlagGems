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


def vstack(tensors: list):
    logger.debug("GEMS_KUNLUNXIN VSTACK")

    tensors = torch.atleast_2d(tensors)
    num_tensors = len(tensors)
    assert num_tensors > 0

    # Ensure all tensors are on the same device and have the same dtype
    device = tensors[0].device
    dtype = tensors[0].dtype
    for tensor in tensors:
        assert (
            tensor.device == device
            and tensor.dtype == dtype
            and tensors[0].shape[1:] == tensor.shape[1:]
        )

    c_tensors = [t.contiguous() for t in tensors]
    return torch.cat(c_tensors, dim=0)
