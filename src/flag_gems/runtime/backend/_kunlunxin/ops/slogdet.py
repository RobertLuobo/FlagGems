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

from flag_gems.ops.slogdet import _check_input, _real_dtype_of
from flag_gems.runtime import torch_device_fn

from .linalg_slogdet import (
    _kg_copy_pad,
    _kg_slogdet_fused,
)

logger = logging.getLogger(__name__)

_SMALL_FUSED_MAX = 0


@triton.jit
def _kg_cplx_copy_pad(A_ptr, Wr_ptr, Wi_ptr, n: tl.constexpr, P: tl.constexpr):
    pid = tl.program_id(0)
    aoff = pid.to(tl.int64) * (n * n)
    woff = pid.to(tl.int64) * (P * P)
    r = tl.arange(0, P)
    rowm = r < n
    for i in tl.static_range(0, n):
        re = tl.load(A_ptr + 2 * (aoff + i * n + r), mask=rowm, other=0.0).to(tl.float32)
        im = tl.load(
            A_ptr + 2 * (aoff + i * n + r) + 1, mask=rowm, other=0.0
        ).to(tl.float32)
        tl.store(Wr_ptr + woff + i * P + r, re, mask=rowm)
        tl.store(Wi_ptr + woff + i * P + r, im, mask=rowm)


@triton.jit
def _kg_cplx_elim_step(
    Wr_ptr,
    Wi_ptr,
    n: tl.constexpr,
    k: tl.constexpr,
    rlo: tl.constexpr,
    rhi: tl.constexpr,
    P: tl.constexpr,
):
    pid = tl.program_id(0)
    wi = pid.to(tl.int64) * (P * P)
    r = tl.arange(0, P)
    rowm = r < n
    pr = tl.load(Wr_ptr + wi + (k * P + k))
    pi = tl.load(Wi_ptr + wi + (k * P + k))
    denom = pr * pr + pi * pi
    safe = tl.where(denom == 0.0, 1.0, denom)
    rowk_r = tl.load(Wr_ptr + wi + (k * P + r), mask=rowm, other=0.0)
    rowk_i = tl.load(Wi_ptr + wi + (k * P + r), mask=rowm, other=0.0)
    for i in tl.static_range(rlo, rhi):
        ar = tl.load(Wr_ptr + wi + (i * P + k))
        ai = tl.load(Wi_ptr + wi + (i * P + k))
        mr = tl.where(denom == 0.0, 0.0, (ar * pr + ai * pi) / safe)
        mi = tl.where(denom == 0.0, 0.0, (ai * pr - ar * pi) / safe)
        rowi_r = tl.load(Wr_ptr + wi + (i * P + r), mask=rowm, other=0.0)
        rowi_i = tl.load(Wi_ptr + wi + (i * P + r), mask=rowm, other=0.0)
        rowi_r = rowi_r - (mr * rowk_r - mi * rowk_i)
        rowi_i = rowi_i - (mr * rowk_i + mi * rowk_r)
        tl.store(Wr_ptr + wi + (i * P + r), rowi_r, mask=rowm)
        tl.store(Wi_ptr + wi + (i * P + r), rowi_i, mask=rowm)


@triton.jit
def _kg_cplx_post(
    Wr_ptr,
    Wi_ptr,
    sr_ptr,
    si_ptr,
    lad_ptr,
    n: tl.constexpr,
    P: tl.constexpr,
):
    pid = tl.program_id(0)
    off = pid.to(tl.int64) * (P * P)
    r = tl.arange(0, P)
    rowm = r < n
    dr = tl.load(Wr_ptr + off + (r * P + r), mask=rowm, other=1.0)
    di = tl.load(Wi_ptr + off + (r * P + r), mask=rowm, other=0.0)
    mod = tl.sqrt(dr * dr + di * di)
    zero = rowm & (mod == 0.0)
    singular = tl.max(tl.where(zero, 1.0, 0.0))
    nan_cnt = tl.sum(tl.where(rowm & (mod != mod), 1.0, 0.0))
    safe_mod = tl.where(mod == 0.0, 1.0, mod)
    lad = tl.sum(tl.where(zero, 0.0, tl.log(safe_mod)))
    acc_r = tl.full((), 1.0, tl.float32)
    acc_i = tl.full((), 0.0, tl.float32)
    for i in tl.static_range(0, n):
        zr = tl.sum(tl.where(r == i, dr / safe_mod, 0.0))
        zi = tl.sum(tl.where(r == i, di / safe_mod, 0.0))
        nr = acc_r * zr - acc_i * zi
        ni = acc_r * zi + acc_i * zr
        acc_r = nr
        acc_i = ni
    bad = (singular == 1.0) | (nan_cnt > 0.0)
    acc_r = tl.where(bad, 0.0, acc_r)
    acc_i = tl.where(bad, 0.0, acc_i)
    lad = tl.where(singular == 1.0, float("-inf"), lad)
    tl.store(sr_ptr + pid, acc_r.to(sr_ptr.dtype.element_ty))
    tl.store(si_ptr + pid, acc_i.to(si_ptr.dtype.element_ty))
    tl.store(lad_ptr + pid, lad.to(lad_ptr.dtype.element_ty))


@triton.jit
def _kg_pivot(
    W_ptr,
    par_ptr,
    n,
    k,
    P: tl.constexpr,
):
    """Partial pivoting for column k. Selects the largest-magnitude entry at or
    below the diagonal and exchanges that row onto the diagonal, counting the
    swap so the determinant sign can be flipped once per swap. Partial pivoting
    is required for numerical reliability at larger n, where a no-pivot
    factorization can flip the sign.

    ``k`` is a runtime scalar (not constexpr) so this kernel compiles once and
    is reused for every pivot column."""
    pid = tl.program_id(0)
    wi = pid.to(tl.int64) * (P * P)
    r = tl.arange(0, P)
    rowm = r < n
    valid = (r >= k) & (r < n)
    col = tl.load(W_ptr + wi + (r * P + k), mask=valid, other=0.0)
    acol = tl.where(valid, tl.abs(col), -1.0)
    pivval = tl.max(acol)
    pivrow = tl.min(tl.where(acol == pivval, r, P))
    rk = tl.load(W_ptr + wi + (k * P + r), mask=rowm, other=0.0)
    rp = tl.load(W_ptr + wi + (pivrow.to(tl.int64) * P + r), mask=rowm, other=0.0)
    tl.store(W_ptr + wi + (k * P + r), rp, mask=rowm)
    tl.store(W_ptr + wi + (pivrow.to(tl.int64) * P + r), rk, mask=rowm)
    prev = tl.load(par_ptr + pid)
    tl.store(par_ptr + pid, prev + tl.where(pivrow != k, 1, 0))


@triton.jit
def _kg_elim(
    W_ptr,
    n,
    k,
    P: tl.constexpr,
):
    """Eliminate every row below the pivot in column k. ``k`` is a runtime
    scalar and the row sweep is a runtime-bounded loop so the kernel compiles
    once and is reused for every pivot column."""
    pid = tl.program_id(0)
    wi = pid.to(tl.int64) * (P * P)
    r = tl.arange(0, P)
    rowm = r < n
    pv = tl.load(W_ptr + wi + (k * P + k))
    rowk = tl.load(W_ptr + wi + (k * P + r), mask=rowm, other=0.0)
    for i in range(k + 1, n):
        ri = tl.load(W_ptr + wi + (i * P + r), mask=rowm, other=0.0)
        a = tl.sum(tl.where(r == k, ri, 0.0))
        mult = tl.where(pv == 0.0, 0.0, a / pv)
        ri = ri - mult * rowk
        tl.store(W_ptr + wi + (i * P + r), ri, mask=rowm)


@triton.jit
def _kg_post_pivot(
    W_ptr,
    par_ptr,
    sign_ptr,
    lad_ptr,
    n: tl.constexpr,
    P: tl.constexpr,
):
    pid = tl.program_id(0)
    off = pid.to(tl.int64) * (P * P)
    r = tl.arange(0, P)
    rowm = r < n
    d = tl.load(W_ptr + off + (r * P + r), mask=rowm, other=0.0)
    da = tl.abs(d)
    zero = rowm & (da == 0.0)
    singular = tl.max(tl.where(zero, 1.0, 0.0))
    nan_cnt = tl.sum(tl.where(rowm & (d != d), 1.0, 0.0))
    neg = tl.sum(tl.where(rowm & (d < 0.0), 1.0, 0.0))
    lad = tl.sum(tl.where(zero, 0.0, tl.log(tl.where(zero, 1.0, da))))
    swaps = tl.load(par_ptr + pid)
    parity = (neg.to(tl.int32) + swaps) % 2
    sign = tl.where(parity == 1, -1.0, 1.0)
    sign = tl.where(nan_cnt > 0.0, 0.0, sign)
    sign = tl.where(singular == 1.0, 0.0, sign)
    lad = tl.where(singular == 1.0, float("-inf"), lad)
    tl.store(sign_ptr + pid, sign.to(sign_ptr.dtype.element_ty))
    tl.store(lad_ptr + pid, lad.to(lad_ptr.dtype.element_ty))


def _pad_size(n):
    if n <= 17:
        p = triton.next_power_of_2(n)
        return 128 if n == 16 else p
    return 128


def _slogdet_real(A, batch_shape, n, M):
    sign = torch.empty(batch_shape, dtype=A.dtype, device=A.device)
    logabsdet = torch.empty(batch_shape, dtype=A.dtype, device=A.device)
    sign_flat = sign.reshape(-1)
    lad_flat = logabsdet.reshape(-1)
    if n <= _SMALL_FUSED_MAX:
        P = triton.next_power_of_2(n)
        W = torch.empty((M * n * n,), dtype=torch.float32, device=A.device)
        _kg_slogdet_fused[(M,)](A, W, sign_flat, lad_flat, n, P=P, num_warps=1)
        return sign, logabsdet
    P = _pad_size(n)
    W = torch.zeros((M * P * P,), dtype=torch.float32, device=A.device)
    par = torch.zeros((M,), dtype=torch.int32, device=A.device)
    _kg_copy_pad[(M,)](A, W, n=n, P=P, num_warps=1)
    for k in range(n):
        _kg_pivot[(M,)](W, par, n, k, P=P, num_warps=1)
        _kg_elim[(M,)](W, n, k, P=P, num_warps=1)
    _kg_post_pivot[(M,)](W, par, sign_flat, lad_flat, n=n, P=P, num_warps=1)
    return sign, logabsdet


def _slogdet_complex(A, batch_shape, n, M):
    real_dtype = _real_dtype_of(A.dtype)
    sign_re = torch.empty(batch_shape, dtype=real_dtype, device=A.device)
    sign_im = torch.empty(batch_shape, dtype=real_dtype, device=A.device)
    logabsdet = torch.empty(batch_shape, dtype=real_dtype, device=A.device)
    sr = sign_re.reshape(-1)
    si = sign_im.reshape(-1)
    lad = logabsdet.reshape(-1)
    P = _pad_size(n)
    Wr = torch.zeros((M * P * P,), dtype=torch.float32, device=A.device)
    Wi = torch.zeros((M * P * P,), dtype=torch.float32, device=A.device)
    A_real = torch.view_as_real(A)
    _kg_cplx_copy_pad[(M,)](A_real, Wr, Wi, n=n, P=P, num_warps=1)
    RS = 16
    for k in range(n):
        rlo = k + 1
        while rlo < n:
            rhi = min(n, rlo + RS)
            _kg_cplx_elim_step[(M,)](
                Wr, Wi, n=n, k=k, rlo=rlo, rhi=rhi, P=P, num_warps=1
            )
            rlo = rhi
    _kg_cplx_post[(M,)](Wr, Wi, sr, si, lad, n=n, P=P, num_warps=1)
    sign = torch.complex(sign_re, sign_im)
    return sign, logabsdet


def _slogdet_forward(A):
    _check_input(A)
    batch_shape = A.shape[:-2]
    n = A.shape[-1]
    real_dtype = _real_dtype_of(A.dtype)

    M = 1
    for d in batch_shape:
        M *= d
    if M == 0:
        return (
            torch.empty(batch_shape, dtype=A.dtype, device=A.device),
            torch.empty(batch_shape, dtype=real_dtype, device=A.device),
        )
    if n == 0:
        return (
            torch.ones(batch_shape, dtype=A.dtype, device=A.device),
            torch.zeros(batch_shape, dtype=real_dtype, device=A.device),
        )

    A_work = A.contiguous().clone()
    with torch_device_fn.device(A.device):
        if A.dtype in (torch.complex64, torch.complex128):
            return _slogdet_complex(A_work, batch_shape, n, M)
        return _slogdet_real(A_work, batch_shape, n, M)


class _Slogdet(torch.autograd.Function):
    @staticmethod
    def forward(ctx, A):
        sign, logabsdet = _slogdet_forward(A)
        ctx.save_for_backward(A)
        return sign, logabsdet

    @staticmethod
    def backward(ctx, grad_sign, grad_logabsdet):
        (A,) = ctx.saved_tensors
        if grad_logabsdet is None:
            return None
        inv_h = torch.linalg.inv(A).mH
        return grad_logabsdet[..., None, None] * inv_h


def slogdet(A):
    logger.debug("GEMS SLOGDET")
    if torch.is_grad_enabled() and A.requires_grad:
        return _Slogdet.apply(A)
    return _slogdet_forward(A)
