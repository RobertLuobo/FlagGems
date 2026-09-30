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
import importlib
import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)

_generic = importlib.import_module("flag_gems.ops.isreal")
isreal_true_kernel = _generic.isreal_true_kernel
_REAL_DTYPES = _generic._REAL_DTYPES
_COMPLEX_DTYPES = _generic._COMPLEX_DTYPES


@libentry()
@triton.jit
def isreal_complex_kernel(in_real_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    """Flat-1D scan of interleaved (real, imag) storage for the XPU3 backend.

    The generic implementation reshapes the pair load to ``[BLOCK, 2]`` and
    ``tl.split``s it; that 2D-tile pattern fails the ConvertTritonToTritonXPU
    pass on XPU3. Here the imaginary lane sits at odd flat offsets
    ``2*i + 1`` of ``view_as_real`` storage, so a direct strided load picks it
    out without any reshape/split.
    """
    pid = tl.program_id(0)
    idx = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = idx < n_elements
    imag = tl.load(in_real_ptr + idx * 2 + 1, mask=mask, other=0.0)
    tl.store(out_ptr + idx, imag == 0.0, mask=mask)


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
        in_real = torch.view_as_real(input).reshape(-1)
        # The imaginary lane is a stride-2 (discrete) load; a 4096-wide block
        # was measured fastest on XPU3 (1024 leaves too many programs, >=8192
        # blows up the discrete-gather tile). ~2x over the 1024 default.
        BLOCK_SIZE = 4096
        grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
        with torch_device_fn.device(input.device):
            isreal_complex_kernel[grid](
                in_real, out, n_elements, BLOCK_SIZE=BLOCK_SIZE
            )
    else:
        # Pure bool fill; a wide 32768 block minimizes program count for this
        # launch/bandwidth-bound path (measured ~2x over 4096 at 16M elements,
        # never worse at small sizes).
        BLOCK_SIZE = 32768
        grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
        with torch_device_fn.device(input.device):
            isreal_true_kernel[grid](out, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return out
