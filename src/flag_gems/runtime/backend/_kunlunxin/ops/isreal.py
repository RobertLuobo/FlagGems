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
import triton
import triton.language as tl

from flag_gems.ops.isreal import isreal_true_kernel
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)

_REAL_DTYPES = (
    torch.float16,
    torch.bfloat16,
    torch.float32,
    torch.float64,
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
    torch.uint8,
    torch.bool,
)
_COMPLEX_DTYPES = (torch.complex64, torch.complex128)


@libentry()
@triton.jit
def isreal_complex_kernel(in_real_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    imag = tl.load(in_real_ptr + offsets * 2 + 1, mask=mask, other=0.0)
    tl.store(out_ptr + offsets, imag == 0.0, mask=mask)


def isreal(input: torch.Tensor) -> torch.Tensor:
    logger.debug("GEMS_KUNLUNXIN ISREAL")
    if not isinstance(input, torch.Tensor):
        raise TypeError(f"isreal expects a Tensor, got {type(input).__name__}")
    if input.dtype not in _REAL_DTYPES + _COMPLEX_DTYPES:
        raise TypeError(
            "isreal only supports floating point, complex, integral and bool "
            f"inputs, got {input.dtype}"
        )

    out = torch.empty(input.size(), dtype=torch.bool, device=input.device)
    n_elements = input.numel()
    if n_elements == 0:
        return out

    if input.dtype in _COMPLEX_DTYPES:
        if input.is_conj():
            input = input.resolve_conj()
        if input.is_neg():
            input = input.resolve_neg()
        if not input.is_contiguous():
            input = input.contiguous()
        BLOCK_SIZE = 4096
        grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
        isreal_complex_kernel[grid](
            torch.view_as_real(input), out, n_elements, BLOCK_SIZE=BLOCK_SIZE
        )
    else:
        BLOCK_SIZE = 32768
        grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
        isreal_true_kernel[grid](out, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return out
