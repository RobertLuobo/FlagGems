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

from ..utils.pointwise_dynamic import pointwise_dynamic
from flag_gems.runtime import torch_device_fn

logger = logging.getLogger(__name__)


@pointwise_dynamic(
    promotion_methods=[
        ((0, 1), "DEFAULT"),
        ((0, 1), "DEFAULT"),
    ],
    num_outputs=2,
)
@triton.jit
def polar_kernel(abs, angle):
    real = abs * tl.cos(angle)
    imag = abs * tl.sin(angle)
    return real, imag


@triton.jit
def _polar_pack_kernel(real, imag, out, n2, BLOCK: tl.constexpr):
    # Interleave two dense real/imag streams into one contiguous complex
    # storage stream: out[2i] = real[i], out[2i+1] = imag[i].  The flat
    # lane index only multiplies/adds in the store address (stride 1) and
    # the parity test only selects a value, mirroring the conj/resolve_conj
    # flat kernels, so the stores stay one dense block DMA.
    pid = tl.program_id(0)
    j = pid * BLOCK + tl.arange(0, BLOCK)
    m = j < n2
    even = (j & 1) == 0
    half = j >> 1
    r = tl.load(real + half, mask=m & even, other=0.0)
    im = tl.load(imag + half, mask=m & (~even), other=0.0)
    v = tl.where(even, r, im)
    tl.store(out + j, v, mask=m)


def polar(abs, angle):
    logger.debug("GEMS_KUNLUNXIN POLAR")
    # XPU note: writing the two components directly into the interleaved
    # complex layout (stride-2 strided outputs) takes the slow rank-2
    # scalarized codegen path (~40s for 16M elts), and torch.complex lowers
    # to two stride-2 ATen copies on XPU.  Write two contiguous real/imag
    # buffers on the fast rank-1 path, then interleave them with one 1D
    # contiguous pack kernel (out viewed as the real dtype).
    real = torch.empty(abs.shape, dtype=abs.dtype, device=abs.device)
    imag = torch.empty(abs.shape, dtype=abs.dtype, device=abs.device)

    polar_kernel(abs, angle, out0=real, out1=imag)

    cplx_dtype = (
        torch.complex64 if abs.dtype == torch.float32 else torch.complex128
    )
    out = torch.empty(abs.shape, dtype=cplx_dtype, device=abs.device)
    fout = out.view(real.dtype).reshape(-1)
    n2 = fout.numel()

    BLOCK = 8192
    grid = (triton.cdiv(n2, BLOCK),)
    with torch_device_fn.device(abs.device):
        _polar_pack_kernel[grid](real, imag, fout, n2, BLOCK=BLOCK, num_warps=8)

    return out
