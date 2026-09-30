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
import math

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as tle

from .linalg_slogdet import linalg_slogdet as _xpu_linalg_slogdet

logger = logging.getLogger(__name__)

_REGISTER_TILE_LIMIT = 32


@libentry()
@triton.jit
def _logdet_combine_kernel(
    sign_ptr,
    lad_ptr,
    out_ptr,
    numel,
    BLOCK: tl.constexpr,
):
    pid = tle.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < numel
    sign = tl.load(sign_ptr + offs, mask=mask, other=1.0).to(tl.float32)
    lad = tl.load(lad_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    res = tl.where(
        sign > 0.0,
        lad,
        tl.where(sign < 0.0, float("nan"), float("-inf")),
    )
    tl.store(out_ptr + offs, res.to(out_ptr.dtype.element_ty), mask=mask)


def logdet(inp):
    logger.debug("GEMS LOGDET")

    if inp.dim() < 2 or inp.shape[-1] != inp.shape[-2]:
        raise RuntimeError("logdet: input must be batches of square matrices")
    if inp.is_complex():
        raise RuntimeError("logdet: complex inputs are not supported by this kernel")
    if inp.requires_grad:
        raise RuntimeError("logdet: autograd is not supported by this kernel")
    if inp.dtype not in (torch.float32, torch.float64):
        raise RuntimeError(f"logdet: unsupported dtype {inp.dtype}")

    n = inp.shape[-1]
    batch_shape = inp.shape[:-2]
    batch_count = math.prod(batch_shape)

    if n == 0:
        return torch.zeros(batch_shape, dtype=inp.dtype, device=inp.device)
    if batch_count == 0:
        return torch.empty(batch_shape, dtype=inp.dtype, device=inp.device)
    fp64_unsupported = inp.dtype == torch.float64 and n > 16
    if n > _REGISTER_TILE_LIMIT or fp64_unsupported:
        raise RuntimeError(
            f"logdet: {inp.dtype} matrices of size {n} are not supported by "
            "the Triton kernel"
        )

    sign, logabsdet = _xpu_linalg_slogdet(inp)
    sign_flat = sign.reshape(-1)
    lad_flat = logabsdet.reshape(-1)
    out = torch.empty(batch_shape, dtype=inp.dtype, device=inp.device)
    out_flat = out.reshape(-1)
    numel = out_flat.numel()
    BLOCK = 1024
    grid = (triton.cdiv(numel, BLOCK),)
    with torch_device_fn.device(inp.device):
        _logdet_combine_kernel[grid](
            sign_flat, lad_flat, out_flat, numel, BLOCK=BLOCK, num_warps=1
        )
    return out
