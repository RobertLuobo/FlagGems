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
import triton.language as tl

from .max import max as _triton_max

logger = logging.getLogger(__name__)


def _infer_num_classes(tensor: torch.Tensor) -> int:
    # num_classes == -1 needs a host scalar = max(index) + 1. Use the kunlunxin
    # triton max reduction (a registered gems op) for this SIZING scalar; the
    # one-hot output itself is produced by the triton scatter kernel below.
    return int(_triton_max(tensor).item()) + 1


@triton.jit
def one_hot_scatter_kernel(
    index_ptr,
    out_ptr,
    numel,
    num_classes,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < numel
    indices = tl.load(index_ptr + offsets, mask=mask, other=0)
    out_offsets = offsets * num_classes + indices
    tl.store(out_ptr + out_offsets, 1, mask=mask)


def _scatter_block_size(numel: int) -> int:
    if numel <= 1024:
        return 256
    elif numel <= 16384:
        return 512
    return 1024


def one_hot(tensor: torch.Tensor, num_classes: int = -1) -> torch.Tensor:
    logger.debug("GEMS_KUNLUNXIN ONE_HOT")
    if not tensor.is_cuda:
        # Defensive host path (this overlay is dispatched only for device
        # tensors): decompose one_hot into zeros + scatter instead of the
        # native F.one_hot so there is no whole-op fallback.
        nc = num_classes
        if nc == -1:
            nc = int(tensor.max().item()) + 1 if tensor.numel() else 0
        out = torch.zeros(
            *tensor.shape, nc, device=tensor.device, dtype=torch.int64
        )
        if tensor.numel() and nc > 0:
            out.scatter_(-1, tensor.unsqueeze(-1), 1)
        return out
    if not tensor.is_contiguous():
        tensor = tensor.contiguous()

    numel = tensor.numel()
    if num_classes == -1:
        if numel == 0:
            raise RuntimeError(
                "Can not infer total number of classes from empty tensor."
            )
        num_classes = _infer_num_classes(tensor)

    total = numel * num_classes
    if numel == 0 or num_classes <= 0:
        out = torch.zeros(total, device=tensor.device, dtype=torch.int64)
        return out.view(*tensor.shape, num_classes)

    BLOCK_SIZE = _scatter_block_size(numel)
    grid = (triton.cdiv(numel, BLOCK_SIZE),)

    guard = BLOCK_SIZE * num_classes
    out = torch.zeros(total + guard, device=tensor.device, dtype=torch.int64)

    one_hot_scatter_kernel[grid](
        tensor.view(-1),
        out,
        numel,
        num_classes,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return out[:total].view(*tensor.shape, num_classes)
