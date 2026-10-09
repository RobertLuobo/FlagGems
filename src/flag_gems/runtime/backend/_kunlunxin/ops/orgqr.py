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

from flag_gems.ops.copy import copy_

from .linalg_householder_product import linalg_householder_product

logger = logging.getLogger("flag_gems").getChild(__name__.lstrip("."))


def _validate_orgqr(input, tau, out=None):
    if input.ndim < 2:
        raise RuntimeError("orgqr: input must have at least 2 dimensions")
    if tau.ndim != input.ndim - 1 or tau.shape[:-1] != input.shape[:-2]:
        raise RuntimeError("orgqr: tau batch dimensions must match input")
    if input.shape[-2] < input.shape[-1]:
        raise RuntimeError(
            "orgqr: input.shape[-2] must be greater than or equal to input.shape[-1]"
        )
    if tau.shape[-1] > input.shape[-1]:
        raise RuntimeError(
            "orgqr: input.shape[-1] must be greater than or equal to tau.shape[-1]"
        )
    if input.dtype != tau.dtype or input.device != tau.device:
        raise RuntimeError("orgqr: input and tau must have the same dtype and device")
    if input.is_complex():
        raise RuntimeError(
            "orgqr: complex dtypes are not supported by this Triton kernel"
        )
    if input.dtype not in (torch.float32, torch.float64):
        raise RuntimeError("orgqr: only float32 and float64 inputs are supported")
    if out is not None:
        if out.device != input.device:
            raise RuntimeError("orgqr: input and out must be on the same device")
        if not torch.can_cast(input.dtype, out.dtype):
            raise RuntimeError(
                f"orgqr: result dtype {input.dtype} cannot be safely cast to {out.dtype}"
            )


def _orgqr_impl(input, tau, out):
    """Reconstruct Q from Householder reflectors on Kunlunxin/XPU.

    The generic ``flag_gems.ops.orgqr`` kernel keeps a [BLOCK_M, BLOCK_N] 2-D
    register tile and drives it with ``dots = tl.sum(..., axis=0)``; the axis-0
    reduce on a 2D+ shape fails ``TritonXPULegalize`` on xpu3
    (``Pipeline failed while executing [TritonXPULegalize]``).  This overlay
    delegates the compute to the vendor ``linalg_householder_product`` -- the
    same primitive ``torch.orgqr`` is defined on (``Q = H(0)*...*H(k-1)``),
    which already lowers and runs correctly on this device via its
    column-per-program one-shot kernel -- and keeps the orgqr-specific
    out= / dtype-cast / aliasing / non-contiguous-batch contract here.
    """
    _validate_orgqr(input, tau, out)
    if out.shape != input.shape:
        out.resize_(input.shape)
    if out.numel() == 0:
        return out

    # Vendor kernel computes Q in input precision into a fresh buffer (it only
    # reads A/tau), so out aliasing input is safe: input is fully consumed
    # before the copy_ below writes back.  copy_ performs the single final cast
    # to out.dtype and respects arbitrary (non-contiguous) out strides.
    q = linalg_householder_product(input, tau)
    copy_(out, q)
    return out


def orgqr(input, tau):
    """Compute real float32/float64 Q from Householder reflectors.

    Complex inputs are intentionally unsupported by this implementation.
    """
    logger.debug("GEMS_KUNLUNXIN ORGQR")
    _validate_orgqr(input, tau)
    out = torch.empty_like(input)
    return _orgqr_impl(input, tau, out)


def orgqr_out(input, tau, *, out):
    """Write Q into ``out``, preserving its storage and arbitrary batch strides."""
    logger.debug("GEMS_KUNLUNXIN ORGQR_OUT")
    return _orgqr_impl(input, tau, out)
