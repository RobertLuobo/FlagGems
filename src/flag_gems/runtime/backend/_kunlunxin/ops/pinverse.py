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

from flag_gems.runtime import device, torch_device_fn

from .bmm import bmm
from .svd import svd

logger = logging.getLogger(__name__)


@triton.jit
def _rcond_scale_kernel(S, OUT, rcond, K, BLOCK: tl.constexpr):
    """Scaled reciprocal singular values with a Moore-Penrose rcond cutoff.

    One program per batch row.  ``largest`` is the first (descending-sorted)
    singular value; values at or below ``rcond * largest`` are dropped to zero
    exactly like ``torch.pinverse``.  NaN ``rcond`` and non-finite tolerances
    are propagated so the resulting pseudoinverse matches the reference.
    """
    pid = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < K
    base = S + pid * K
    s = tl.load(base + offs, mask=mask, other=0.0)
    largest = tl.load(base)
    valid_rcond = rcond == rcond
    scaled_tolerance = rcond * largest
    tolerance = tl.where(
        scaled_tolerance == scaled_tolerance,
        tl.where(scaled_tolerance > 0.0, scaled_tolerance, 0.0),
        scaled_tolerance,
    )
    active = mask & valid_rcond & (s > tolerance)
    inv = tl.where(active, 1.0 / tl.where(active, s, 1.0), 0.0)
    tl.store(OUT + pid * K + offs, inv, mask=mask)


def pinverse(inp, rcond=1e-15):
    """Moore-Penrose pseudoinverse built on the vendor Triton SVD.

    The generic implementation routes through the generic one-sided Jacobi SVD,
    whose fully static-unrolled small-matrix kernel overflows the XPU3 unroll
    controller (``Failed to tune buffer size.``).  This overlay reuses the
    vendor ``svd`` kernels and assembles ``A+ = V @ diag(s_inv) @ U^T`` with the
    vendor ``bmm``; the rcond cutoff on the singular values is done in Triton.
    """
    logger.debug("GEMS_KUNLUNXIN PINVERSE")

    if inp.ndim < 2:
        raise RuntimeError("pinverse: expected a tensor with at least 2 dimensions")
    if inp.device.type != device.name or inp.dtype != torch.float32:
        raise NotImplementedError(
            f"FlagGems pinverse currently supports only float32 {device.name} tensors"
        )
    if inp.requires_grad:
        raise NotImplementedError(
            "FlagGems pinverse does not yet support autograd inputs"
        )
    if rcond < 0.0:
        raise NotImplementedError(
            "FlagGems pinverse does not yet support negative rcond values"
        )

    m, n = inp.shape[-2:]
    lead = inp.shape[:-2]
    if inp.numel() == 0:
        return torch.empty((*lead, n, m), dtype=inp.dtype, device=inp.device)

    k = min(m, n)
    batch = inp.numel() // (m * n)

    result = svd(inp, some=True, compute_uv=True)
    u = result.U.reshape(batch, m, k)
    s = result.S.reshape(batch, k).contiguous()
    v = result.V.reshape(batch, n, k)

    s_inv = torch.empty((batch, k), dtype=torch.float32, device=inp.device)
    block = triton.next_power_of_2(k)
    with torch_device_fn.device(inp.device):
        _rcond_scale_kernel[(batch,)](
            s, s_inv, rcond, k, BLOCK=block, num_warps=1
        )

    scaled_v = v * s_inv.unsqueeze(-2)
    output = bmm(scaled_v, u.transpose(-2, -1).contiguous())
    return output.reshape((*lead, n, m)).contiguous()
