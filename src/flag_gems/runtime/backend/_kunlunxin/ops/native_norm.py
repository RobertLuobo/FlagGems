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
import triton

from flag_gems.runtime import torch_device_fn

from .vector_norm import (
    l1_norm_kernel_1,
    l1_norm_kernel_2,
    l1_norm_tail_kernel,
    zero_workspace_kernel,
)

logger = logging.getLogger(__name__)


def native_norm(self, p=2):
    logger.debug("GEMS_KUNLUNXIN NATIVE_NORM")

    if self.dtype not in [torch.float16, torch.float32, torch.bfloat16]:
        raise NotImplementedError(f"native_norm not implemented for {self.dtype}")

    ord = float(p)
    x = self.contiguous()
    M = x.numel()
    dtype = self.dtype

    with torch_device_fn.device(x.device):
        cluster_num = 12
        BLOCK_SIZE = min(
            triton.next_power_of_2(triton.cdiv(M, cluster_num)),
            32768,
        )
        MID_SIZE = triton.cdiv(M, BLOCK_SIZE)
        BLOCK_MID = triton.next_power_of_2(MID_SIZE)

        mid = torch.empty_strided(
            [BLOCK_MID], [1], dtype=torch.float32, device=x.device
        )
        zero_workspace_kernel[(1,)](mid, BLOCK_MID)
        out = torch.empty([], dtype=dtype, device=x.device)

        l1_norm_kernel_1[(MID_SIZE,)](
            x, mid, ord, M, BLOCK_SIZE, buffer_size_limit=2048
        )
        tail_size = M % BLOCK_SIZE
        if tail_size:
            l1_norm_tail_kernel[(1,)](
                x,
                mid,
                ord,
                M - tail_size,
                tail_size,
                MID_SIZE - 1,
                buffer_size_limit=2048,
            )
        l1_norm_kernel_2[(1,)](
            mid, out, ord, MID_SIZE, BLOCK_MID, buffer_size_limit=2048
        )
    return out
