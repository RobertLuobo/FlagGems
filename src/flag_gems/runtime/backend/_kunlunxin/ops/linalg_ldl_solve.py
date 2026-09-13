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
#
# Triton implementation of linalg_ldl_solve.
import logging
import math

import torch
import triton
import triton.language as tl

from flag_gems.utils import libentry

logger = logging.getLogger(__name__)

LDL_SOLVE_RHS_BLOCK = 32
LDL_SOLVE_KS = 64  # RHS column-slice width (fixed, bounded)
REAL_DTYPES = (torch.float32, torch.float64)
COMPLEX_DTYPES = (torch.complex64, torch.complex128)


@triton.jit
def _cmul(ar, ai, br, bi):
    return ar * br - ai * bi, ar * bi + ai * br


@triton.jit
def _cconj_mul(ar, ai, br, bi):
    return ar * br + ai * bi, ar * bi - ai * br


@triton.jit
def _cdiv(nr, ni, dr, di):
    denom = dr * dr + di * di
    return (nr * dr + ni * di) / denom, (ni * dr - nr * di) / denom


@libentry()
@triton.jit(
    do_not_specialize=[
        "n",
        "nrhs",
        "ld_batch_stride",
        "ld_row_stride",
        "ld_col_stride",
        "piv_batch_stride",
        "piv_row_stride",
        "x_batch_stride",
        "x_row_stride",
        "x_col_stride",
    ]
)
def linalg_ldl_solve_real_kernel(
    LD,
    pivots,
    X,
    n,
    nrhs,
    ld_batch_stride,
    ld_row_stride,
    ld_col_stride,
    piv_batch_stride,
    piv_row_stride,
    x_batch_stride,
    x_row_stride,
    x_col_stride,
    BLOCK_NRH: tl.constexpr,
    NUM_RHS_BLOCKS: tl.constexpr,
):
    program_id = tl.program_id(0)
    batch_idx = program_id // NUM_RHS_BLOCKS
    block_idx = program_id % NUM_RHS_BLOCKS
    col_start = block_idx * BLOCK_NRH
    cols = col_start + tl.arange(0, BLOCK_NRH)
    col_mask = cols < nrhs

    ld_base = LD + batch_idx * ld_batch_stride
    piv_base = pivots + batch_idx * piv_batch_stride
    x_base = X + batch_idx * x_batch_stride
    col_offsets = cols * x_col_stride

    k = 0
    while k < n:
        ip = tl.load(piv_base + k * piv_row_stride).to(tl.int32)
        if ip > 0:
            kp = ip - 1
            row_k_ptr = x_base + k * x_row_stride + col_offsets
            if kp != k:
                row_kp_ptr = x_base + kp * x_row_stride + col_offsets
                xk = tl.load(row_k_ptr, mask=col_mask, other=0.0)
                xkp = tl.load(row_kp_ptr, mask=col_mask, other=0.0)
                tl.store(row_k_ptr, xkp, mask=col_mask)
                tl.store(row_kp_ptr, xk, mask=col_mask)
            xk = tl.load(row_k_ptr, mask=col_mask, other=0.0)

            i = k + 1
            while i < n:
                lij = tl.load(ld_base + i * ld_row_stride + k * ld_col_stride)
                row_i_ptr = x_base + i * x_row_stride + col_offsets
                xi = tl.load(row_i_ptr, mask=col_mask, other=0.0)
                xi -= lij * xk
                tl.store(row_i_ptr, xi, mask=col_mask)
                i += 1

            d = tl.load(ld_base + k * ld_row_stride + k * ld_col_stride)
            xk = xk / d
            tl.store(row_k_ptr, xk, mask=col_mask)
            k += 1
        else:
            kp = -ip - 1
            row_k_ptr = x_base + k * x_row_stride + col_offsets
            row_k1_ptr = x_base + (k + 1) * x_row_stride + col_offsets
            if kp != k + 1:
                row_kp_ptr = x_base + kp * x_row_stride + col_offsets
                xk1 = tl.load(row_k1_ptr, mask=col_mask, other=0.0)
                xkp = tl.load(row_kp_ptr, mask=col_mask, other=0.0)
                tl.store(row_k1_ptr, xkp, mask=col_mask)
                tl.store(row_kp_ptr, xk1, mask=col_mask)
            xk = tl.load(row_k_ptr, mask=col_mask, other=0.0)
            xk1 = tl.load(row_k1_ptr, mask=col_mask, other=0.0)

            i = k + 2
            while i < n:
                l0 = tl.load(ld_base + i * ld_row_stride + k * ld_col_stride)
                l1 = tl.load(ld_base + i * ld_row_stride + (k + 1) * ld_col_stride)
                row_i_ptr = x_base + i * x_row_stride + col_offsets
                xi = tl.load(row_i_ptr, mask=col_mask, other=0.0)
                xi -= l0 * xk + l1 * xk1
                tl.store(row_i_ptr, xi, mask=col_mask)
                i += 1

            b = tl.load(ld_base + (k + 1) * ld_row_stride + k * ld_col_stride)
            a = tl.load(ld_base + k * ld_row_stride + k * ld_col_stride)
            c = tl.load(ld_base + (k + 1) * ld_row_stride + (k + 1) * ld_col_stride)
            akm1 = a / b
            ak = c / b
            denom = akm1 * ak - 1
            bkm1 = xk / b
            bk = xk1 / b
            xk = (ak * bkm1 - bk) / denom
            xk1 = (akm1 * bk - bkm1) / denom
            tl.store(row_k_ptr, xk, mask=col_mask)
            tl.store(row_k1_ptr, xk1, mask=col_mask)
            k += 2

    k = n - 1
    while k >= 0:
        ip = tl.load(piv_base + k * piv_row_stride).to(tl.int32)
        if ip > 0:
            row_k_ptr = x_base + k * x_row_stride + col_offsets
            xk = tl.load(row_k_ptr, mask=col_mask, other=0.0)

            i = k + 1
            while i < n:
                lij = tl.load(ld_base + i * ld_row_stride + k * ld_col_stride)
                row_i_ptr = x_base + i * x_row_stride + col_offsets
                xi = tl.load(row_i_ptr, mask=col_mask, other=0.0)
                xk -= lij * xi
                i += 1

            kp = ip - 1
            if kp != k:
                row_kp_ptr = x_base + kp * x_row_stride + col_offsets
                xkp = tl.load(row_kp_ptr, mask=col_mask, other=0.0)
                tl.store(row_k_ptr, xkp, mask=col_mask)
                tl.store(row_kp_ptr, xk, mask=col_mask)
            else:
                tl.store(row_k_ptr, xk, mask=col_mask)
            k -= 1
        else:
            row_km1_ptr = x_base + (k - 1) * x_row_stride + col_offsets
            row_k_ptr = x_base + k * x_row_stride + col_offsets
            xkm1 = tl.load(row_km1_ptr, mask=col_mask, other=0.0)
            xk = tl.load(row_k_ptr, mask=col_mask, other=0.0)

            i = k + 1
            while i < n:
                l0 = tl.load(ld_base + i * ld_row_stride + (k - 1) * ld_col_stride)
                l1 = tl.load(ld_base + i * ld_row_stride + k * ld_col_stride)
                row_i_ptr = x_base + i * x_row_stride + col_offsets
                xi = tl.load(row_i_ptr, mask=col_mask, other=0.0)
                xkm1 -= l0 * xi
                xk -= l1 * xi
                i += 1

            kp = -ip - 1
            if kp != k:
                row_kp_ptr = x_base + kp * x_row_stride + col_offsets
                xkp = tl.load(row_kp_ptr, mask=col_mask, other=0.0)
                tl.store(row_k_ptr, xkp, mask=col_mask)
                tl.store(row_kp_ptr, xk, mask=col_mask)
            else:
                tl.store(row_k_ptr, xk, mask=col_mask)
            tl.store(row_km1_ptr, xkm1, mask=col_mask)
            k -= 2


@libentry()
@triton.jit(
    do_not_specialize=[
        "n",
        "nrhs",
        "ld_batch_stride",
        "ld_row_stride",
        "ld_col_stride",
        "piv_batch_stride",
        "piv_row_stride",
        "x_batch_stride",
        "x_row_stride",
        "x_col_stride",
    ]
)
def linalg_ldl_solve_complex_kernel(
    LD,
    pivots,
    X,
    n,
    nrhs,
    ld_batch_stride,
    ld_row_stride,
    ld_col_stride,
    piv_batch_stride,
    piv_row_stride,
    x_batch_stride,
    x_row_stride,
    x_col_stride,
    HERM: tl.constexpr,
    BLOCK_NRH: tl.constexpr,
):
    batch_idx = tl.program_id(0)
    block_idx = tl.program_id(1)
    col_start = block_idx * BLOCK_NRH
    cols = col_start + tl.arange(0, BLOCK_NRH)
    col_mask = cols < nrhs

    ld_base = LD + batch_idx * ld_batch_stride
    piv_base = pivots + batch_idx * piv_batch_stride
    x_base = X + batch_idx * x_batch_stride
    col_offsets = cols * x_col_stride

    k = 0
    while k < n:
        ip = tl.load(piv_base + k * piv_row_stride).to(tl.int32)
        if ip > 0:
            kp = ip - 1
            row_k_ptr = x_base + k * x_row_stride + col_offsets
            row_k_ptr_i = row_k_ptr + 1
            if kp != k:
                row_kp_ptr = x_base + kp * x_row_stride + col_offsets
                row_kp_ptr_i = row_kp_ptr + 1
                xk_r = tl.load(row_k_ptr, mask=col_mask, other=0.0)
                xk_i = tl.load(row_k_ptr_i, mask=col_mask, other=0.0)
                xkp_r = tl.load(row_kp_ptr, mask=col_mask, other=0.0)
                xkp_i = tl.load(row_kp_ptr_i, mask=col_mask, other=0.0)
                tl.store(row_k_ptr, xkp_r, mask=col_mask)
                tl.store(row_k_ptr_i, xkp_i, mask=col_mask)
                tl.store(row_kp_ptr, xk_r, mask=col_mask)
                tl.store(row_kp_ptr_i, xk_i, mask=col_mask)
            xk_r = tl.load(row_k_ptr, mask=col_mask, other=0.0)
            xk_i = tl.load(row_k_ptr_i, mask=col_mask, other=0.0)

            i = k + 1
            while i < n:
                l_r = tl.load(ld_base + i * ld_row_stride + k * ld_col_stride)
                l_i = tl.load(ld_base + i * ld_row_stride + k * ld_col_stride + 1)
                row_i_ptr = x_base + i * x_row_stride + col_offsets
                row_i_ptr_i = row_i_ptr + 1
                xi_r = tl.load(row_i_ptr, mask=col_mask, other=0.0)
                xi_i = tl.load(row_i_ptr_i, mask=col_mask, other=0.0)
                prod_r, prod_i = _cmul(l_r, l_i, xk_r, xk_i)
                xi_r -= prod_r
                xi_i -= prod_i
                tl.store(row_i_ptr, xi_r, mask=col_mask)
                tl.store(row_i_ptr_i, xi_i, mask=col_mask)
                i += 1

            d_r = tl.load(ld_base + k * ld_row_stride + k * ld_col_stride)
            if HERM:
                inv_d = 1.0 / d_r
                xk_r = xk_r * inv_d
                xk_i = xk_i * inv_d
            else:
                d_i = tl.load(ld_base + k * ld_row_stride + k * ld_col_stride + 1)
                xk_r, xk_i = _cdiv(xk_r, xk_i, d_r, d_i)
            tl.store(row_k_ptr, xk_r, mask=col_mask)
            tl.store(row_k_ptr_i, xk_i, mask=col_mask)
            k += 1
        else:
            kp = -ip - 1
            row_k_ptr = x_base + k * x_row_stride + col_offsets
            row_k_ptr_i = row_k_ptr + 1
            row_k1_ptr = x_base + (k + 1) * x_row_stride + col_offsets
            row_k1_ptr_i = row_k1_ptr + 1
            if kp != k + 1:
                row_kp_ptr = x_base + kp * x_row_stride + col_offsets
                row_kp_ptr_i = row_kp_ptr + 1
                xk1_r = tl.load(row_k1_ptr, mask=col_mask, other=0.0)
                xk1_i = tl.load(row_k1_ptr_i, mask=col_mask, other=0.0)
                xkp_r = tl.load(row_kp_ptr, mask=col_mask, other=0.0)
                xkp_i = tl.load(row_kp_ptr_i, mask=col_mask, other=0.0)
                tl.store(row_k1_ptr, xkp_r, mask=col_mask)
                tl.store(row_k1_ptr_i, xkp_i, mask=col_mask)
                tl.store(row_kp_ptr, xk1_r, mask=col_mask)
                tl.store(row_kp_ptr_i, xk1_i, mask=col_mask)
            xk_r = tl.load(row_k_ptr, mask=col_mask, other=0.0)
            xk_i = tl.load(row_k_ptr_i, mask=col_mask, other=0.0)
            xk1_r = tl.load(row_k1_ptr, mask=col_mask, other=0.0)
            xk1_i = tl.load(row_k1_ptr_i, mask=col_mask, other=0.0)

            i = k + 2
            while i < n:
                l0_r = tl.load(ld_base + i * ld_row_stride + k * ld_col_stride)
                l0_i = tl.load(ld_base + i * ld_row_stride + k * ld_col_stride + 1)
                l1_r = tl.load(ld_base + i * ld_row_stride + (k + 1) * ld_col_stride)
                l1_i = tl.load(
                    ld_base + i * ld_row_stride + (k + 1) * ld_col_stride + 1
                )
                row_i_ptr = x_base + i * x_row_stride + col_offsets
                row_i_ptr_i = row_i_ptr + 1
                xi_r = tl.load(row_i_ptr, mask=col_mask, other=0.0)
                xi_i = tl.load(row_i_ptr_i, mask=col_mask, other=0.0)
                prod0_r, prod0_i = _cmul(l0_r, l0_i, xk_r, xk_i)
                prod1_r, prod1_i = _cmul(l1_r, l1_i, xk1_r, xk1_i)
                xi_r -= prod0_r + prod1_r
                xi_i -= prod0_i + prod1_i
                tl.store(row_i_ptr, xi_r, mask=col_mask)
                tl.store(row_i_ptr_i, xi_i, mask=col_mask)
                i += 1

            b_r = tl.load(ld_base + (k + 1) * ld_row_stride + k * ld_col_stride)
            b_i = tl.load(ld_base + (k + 1) * ld_row_stride + k * ld_col_stride + 1)
            a_r = tl.load(ld_base + k * ld_row_stride + k * ld_col_stride)
            a_i = tl.load(ld_base + k * ld_row_stride + k * ld_col_stride + 1)
            c_r = tl.load(ld_base + (k + 1) * ld_row_stride + (k + 1) * ld_col_stride)
            c_i = tl.load(
                ld_base + (k + 1) * ld_row_stride + (k + 1) * ld_col_stride + 1
            )
            if HERM:
                akm1_r, akm1_i = _cdiv(a_r, a_i, b_r, -b_i)
                ak_r, ak_i = _cdiv(c_r, c_i, b_r, b_i)
                denom_r, denom_i = _cmul(akm1_r, akm1_i, ak_r, ak_i)
                denom_r -= 1
                bkm1_r, bkm1_i = _cdiv(xk_r, xk_i, b_r, -b_i)
                bk_r, bk_i = _cdiv(xk1_r, xk1_i, b_r, b_i)
                tmp_r, tmp_i = _cmul(ak_r, ak_i, bkm1_r, bkm1_i)
                xk_r, xk_i = _cdiv(tmp_r - bk_r, tmp_i - bk_i, denom_r, denom_i)
                tmp_r, tmp_i = _cmul(akm1_r, akm1_i, bk_r, bk_i)
                xk1_r, xk1_i = _cdiv(tmp_r - bkm1_r, tmp_i - bkm1_i, denom_r, denom_i)
            else:
                akm1_r, akm1_i = _cdiv(a_r, a_i, b_r, b_i)
                ak_r, ak_i = _cdiv(c_r, c_i, b_r, b_i)
                denom_r, denom_i = _cmul(akm1_r, akm1_i, ak_r, ak_i)
                denom_r -= 1
                bkm1_r, bkm1_i = _cdiv(xk_r, xk_i, b_r, b_i)
                bk_r, bk_i = _cdiv(xk1_r, xk1_i, b_r, b_i)
                tmp_r, tmp_i = _cmul(ak_r, ak_i, bkm1_r, bkm1_i)
                xk_r, xk_i = _cdiv(tmp_r - bk_r, tmp_i - bk_i, denom_r, denom_i)
                tmp_r, tmp_i = _cmul(akm1_r, akm1_i, bk_r, bk_i)
                xk1_r, xk1_i = _cdiv(tmp_r - bkm1_r, tmp_i - bkm1_i, denom_r, denom_i)
            tl.store(row_k_ptr, xk_r, mask=col_mask)
            tl.store(row_k_ptr_i, xk_i, mask=col_mask)
            tl.store(row_k1_ptr, xk1_r, mask=col_mask)
            tl.store(row_k1_ptr_i, xk1_i, mask=col_mask)
            k += 2

    k = n - 1
    while k >= 0:
        ip = tl.load(piv_base + k * piv_row_stride).to(tl.int32)
        if ip > 0:
            row_k_ptr = x_base + k * x_row_stride + col_offsets
            row_k_ptr_i = row_k_ptr + 1
            xk_r = tl.load(row_k_ptr, mask=col_mask, other=0.0)
            xk_i = tl.load(row_k_ptr_i, mask=col_mask, other=0.0)

            i = k + 1
            while i < n:
                l_r = tl.load(ld_base + i * ld_row_stride + k * ld_col_stride)
                l_i = tl.load(ld_base + i * ld_row_stride + k * ld_col_stride + 1)
                row_i_ptr = x_base + i * x_row_stride + col_offsets
                row_i_ptr_i = row_i_ptr + 1
                xi_r = tl.load(row_i_ptr, mask=col_mask, other=0.0)
                xi_i = tl.load(row_i_ptr_i, mask=col_mask, other=0.0)
                if HERM:
                    prod_r, prod_i = _cconj_mul(l_r, l_i, xi_r, xi_i)
                else:
                    prod_r, prod_i = _cmul(l_r, l_i, xi_r, xi_i)
                xk_r -= prod_r
                xk_i -= prod_i
                i += 1

            kp = ip - 1
            if kp != k:
                row_kp_ptr = x_base + kp * x_row_stride + col_offsets
                row_kp_ptr_i = row_kp_ptr + 1
                xkp_r = tl.load(row_kp_ptr, mask=col_mask, other=0.0)
                xkp_i = tl.load(row_kp_ptr_i, mask=col_mask, other=0.0)
                tl.store(row_k_ptr, xkp_r, mask=col_mask)
                tl.store(row_k_ptr_i, xkp_i, mask=col_mask)
                tl.store(row_kp_ptr, xk_r, mask=col_mask)
                tl.store(row_kp_ptr_i, xk_i, mask=col_mask)
            else:
                tl.store(row_k_ptr, xk_r, mask=col_mask)
                tl.store(row_k_ptr_i, xk_i, mask=col_mask)
            k -= 1
        else:
            row_km1_ptr = x_base + (k - 1) * x_row_stride + col_offsets
            row_km1_ptr_i = row_km1_ptr + 1
            row_k_ptr = x_base + k * x_row_stride + col_offsets
            row_k_ptr_i = row_k_ptr + 1
            xkm1_r = tl.load(row_km1_ptr, mask=col_mask, other=0.0)
            xkm1_i = tl.load(row_km1_ptr_i, mask=col_mask, other=0.0)
            xk_r = tl.load(row_k_ptr, mask=col_mask, other=0.0)
            xk_i = tl.load(row_k_ptr_i, mask=col_mask, other=0.0)

            i = k + 1
            while i < n:
                l0_r = tl.load(ld_base + i * ld_row_stride + (k - 1) * ld_col_stride)
                l0_i = tl.load(
                    ld_base + i * ld_row_stride + (k - 1) * ld_col_stride + 1
                )
                l1_r = tl.load(ld_base + i * ld_row_stride + k * ld_col_stride)
                l1_i = tl.load(ld_base + i * ld_row_stride + k * ld_col_stride + 1)
                row_i_ptr = x_base + i * x_row_stride + col_offsets
                row_i_ptr_i = row_i_ptr + 1
                xi_r = tl.load(row_i_ptr, mask=col_mask, other=0.0)
                xi_i = tl.load(row_i_ptr_i, mask=col_mask, other=0.0)
                if HERM:
                    prod0_r, prod0_i = _cconj_mul(l0_r, l0_i, xi_r, xi_i)
                    prod1_r, prod1_i = _cconj_mul(l1_r, l1_i, xi_r, xi_i)
                else:
                    prod0_r, prod0_i = _cmul(l0_r, l0_i, xi_r, xi_i)
                    prod1_r, prod1_i = _cmul(l1_r, l1_i, xi_r, xi_i)
                xkm1_r -= prod0_r
                xkm1_i -= prod0_i
                xk_r -= prod1_r
                xk_i -= prod1_i
                i += 1

            kp = -ip - 1
            if kp != k:
                row_kp_ptr = x_base + kp * x_row_stride + col_offsets
                row_kp_ptr_i = row_kp_ptr + 1
                xkp_r = tl.load(row_kp_ptr, mask=col_mask, other=0.0)
                xkp_i = tl.load(row_kp_ptr_i, mask=col_mask, other=0.0)
                tl.store(row_k_ptr, xkp_r, mask=col_mask)
                tl.store(row_k_ptr_i, xkp_i, mask=col_mask)
                tl.store(row_kp_ptr, xk_r, mask=col_mask)
                tl.store(row_kp_ptr_i, xk_i, mask=col_mask)
            else:
                tl.store(row_k_ptr, xk_r, mask=col_mask)
                tl.store(row_k_ptr_i, xk_i, mask=col_mask)
            tl.store(row_km1_ptr, xkm1_r, mask=col_mask)
            tl.store(row_km1_ptr_i, xkm1_i, mask=col_mask)
            k -= 2


@libentry()
@triton.jit(
    do_not_specialize=[
        "n",
        "nrhs",
        "ld_batch_stride",
        "ld_row_stride",
        "ld_col_stride",
        "piv_batch_stride",
        "piv_row_stride",
        "x_batch_stride",
        "x_row_stride",
        "x_col_stride",
    ]
)
def linalg_ldl_solve_real_fast_kernel(
    LD,
    pivots,
    X,
    n,
    nrhs,
    ld_batch_stride,
    ld_row_stride,
    ld_col_stride,
    piv_batch_stride,
    piv_row_stride,
    x_batch_stride,
    x_row_stride,
    x_col_stride,
    KS: tl.constexpr,
    NS: tl.constexpr,
):
    program_id = tl.program_id(0)
    batch_idx = program_id // NS
    block_idx = program_id % NS
    col_start = block_idx * KS
    cols = col_start + tl.arange(0, KS)
    col_mask = cols < nrhs

    ld_base = LD + batch_idx * ld_batch_stride
    piv_base = pivots + batch_idx * piv_batch_stride
    x_base = X + batch_idx * x_batch_stride
    col_offsets = cols * x_col_stride

    # Forward substitution: solve L * z = P * B  (P applied via row swaps).
    # Row k of X holds the raw (possibly pre-swapped) RHS until it is
    # finalized; rows < k hold the solved z and are re-read by the dot product.
    k = 0
    while k < n:
        ip = tl.load(piv_base + k * piv_row_stride).to(tl.int32)
        if ip > 0:
            kp = ip - 1
            if kp != k:
                row_k_ptr = x_base + k * x_row_stride + col_offsets
                row_kp_ptr = x_base + kp * x_row_stride + col_offsets
                xk = tl.load(row_k_ptr, mask=col_mask, other=0.0)
                xkp = tl.load(row_kp_ptr, mask=col_mask, other=0.0)
                tl.store(row_k_ptr, xkp, mask=col_mask)
                tl.store(row_kp_ptr, xk, mask=col_mask)

            acc = tl.load(
                x_base + k * x_row_stride + col_offsets, mask=col_mask, other=0.0
            )
            for j in range(k):
                lij = tl.load(ld_base + k * ld_row_stride + j * ld_col_stride)
                zj = tl.load(
                    x_base + j * x_row_stride + col_offsets, mask=col_mask, other=0.0
                )
                acc -= lij * zj
            # Store the undivided z; D^-1 is applied at the start of the
            # backward sweep (TritonXPU has no separate register-path for the
            # in-place rows read back by later dot products).
            tl.store(
                x_base + k * x_row_stride + col_offsets, acc, mask=col_mask
            )
            k += 1
        else:
            kp = -ip - 1
            if kp != k + 1:
                row_k1_ptr = x_base + (k + 1) * x_row_stride + col_offsets
                row_kp_ptr = x_base + kp * x_row_stride + col_offsets
                xk1 = tl.load(row_k1_ptr, mask=col_mask, other=0.0)
                xkp = tl.load(row_kp_ptr, mask=col_mask, other=0.0)
                tl.store(row_k1_ptr, xkp, mask=col_mask)
                tl.store(row_kp_ptr, xk1, mask=col_mask)

            zk = tl.load(
                x_base + k * x_row_stride + col_offsets, mask=col_mask, other=0.0
            )
            zk1 = tl.load(
                x_base + (k + 1) * x_row_stride + col_offsets,
                mask=col_mask,
                other=0.0,
            )
            for j in range(k):
                l0 = tl.load(ld_base + k * ld_row_stride + j * ld_col_stride)
                l1 = tl.load(
                    ld_base + (k + 1) * ld_row_stride + j * ld_col_stride
                )
                zj = tl.load(
                    x_base + j * x_row_stride + col_offsets, mask=col_mask, other=0.0
                )
                zk -= l0 * zj
                zk1 -= l1 * zj

            b = tl.load(ld_base + (k + 1) * ld_row_stride + k * ld_col_stride)
            a = tl.load(ld_base + k * ld_row_stride + k * ld_col_stride)
            c = tl.load(
                ld_base + (k + 1) * ld_row_stride + (k + 1) * ld_col_stride
            )
            akm1 = a / b
            ak = c / b
            denom = akm1 * ak - 1
            bkm1 = zk / b
            bk = zk1 / b
            xk = (ak * bkm1 - bk) / denom
            xk1 = (akm1 * bk - bkm1) / denom
            tl.store(
                x_base + k * x_row_stride + col_offsets, xk, mask=col_mask
            )
            tl.store(
                x_base + (k + 1) * x_row_stride + col_offsets, xk1, mask=col_mask
            )
            k += 2

    # Backward substitution: solve L^T * x = D^-1 * z.
    k = n - 1
    while k >= 0:
        ip = tl.load(piv_base + k * piv_row_stride).to(tl.int32)
        if ip > 0:
            kp = ip - 1
            acc = tl.load(
                x_base + k * x_row_stride + col_offsets, mask=col_mask, other=0.0
            )
            # Apply D^-1 (rows hold the undivided z from the forward sweep):
            # w_k = z_k / D[k], then L^T y = w is a plain unit-lower solve.
            d = tl.load(ld_base + k * ld_row_stride + k * ld_col_stride)
            acc = acc * (1.0 / d)
            for j in range(k + 1, n):
                ljk = tl.load(ld_base + j * ld_row_stride + k * ld_col_stride)
                xj = tl.load(
                    x_base + j * x_row_stride + col_offsets, mask=col_mask, other=0.0
                )
                acc -= ljk * xj
            if kp != k:
                row_k_ptr = x_base + k * x_row_stride + col_offsets
                row_kp_ptr = x_base + kp * x_row_stride + col_offsets
                xkp = tl.load(row_kp_ptr, mask=col_mask, other=0.0)
                tl.store(row_k_ptr, xkp, mask=col_mask)
                tl.store(row_kp_ptr, acc, mask=col_mask)
            else:
                tl.store(
                    x_base + k * x_row_stride + col_offsets, acc, mask=col_mask
                )
            k -= 1
        else:
            kp = -ip - 1
            zkm1 = tl.load(
                x_base + (k - 1) * x_row_stride + col_offsets,
                mask=col_mask,
                other=0.0,
            )
            zk = tl.load(
                x_base + k * x_row_stride + col_offsets, mask=col_mask, other=0.0
            )
            for j in range(k + 1, n):
                l0 = tl.load(
                    ld_base + j * ld_row_stride + (k - 1) * ld_col_stride
                )
                l1 = tl.load(ld_base + j * ld_row_stride + k * ld_col_stride)
                xj = tl.load(
                    x_base + j * x_row_stride + col_offsets, mask=col_mask, other=0.0
                )
                zkm1 -= l0 * xj
                zk -= l1 * xj
            if kp != k:
                row_k_ptr = x_base + k * x_row_stride + col_offsets
                row_kp_ptr = x_base + kp * x_row_stride + col_offsets
                xkp = tl.load(row_kp_ptr, mask=col_mask, other=0.0)
                tl.store(row_k_ptr, xkp, mask=col_mask)
                tl.store(row_kp_ptr, zk, mask=col_mask)
            else:
                tl.store(
                    x_base + k * x_row_stride + col_offsets, zk, mask=col_mask
                )
            tl.store(
                x_base + (k - 1) * x_row_stride + col_offsets,
                zkm1,
                mask=col_mask,
            )
            k -= 2


@libentry()
@triton.jit(
    do_not_specialize=[
        "n",
        "nrhs",
        "ld_batch_stride",
        "ld_row_stride",
        "ld_col_stride",
        "piv_batch_stride",
        "piv_row_stride",
        "x_batch_stride",
        "x_row_stride",
        "x_col_stride",
    ]
)
def linalg_ldl_solve_complex_fast_kernel(
    LD,
    pivots,
    X,
    n,
    nrhs,
    ld_batch_stride,
    ld_row_stride,
    ld_col_stride,
    piv_batch_stride,
    piv_row_stride,
    x_batch_stride,
    x_row_stride,
    x_col_stride,
    HERM: tl.constexpr,
    KS: tl.constexpr,
    NS: tl.constexpr,
):
    program_id = tl.program_id(0)
    batch_idx = program_id // NS
    block_idx = program_id % NS
    col_start = block_idx * KS
    cols = col_start + tl.arange(0, KS)
    col_mask = cols < nrhs

    ld_base = LD + batch_idx * ld_batch_stride
    piv_base = pivots + batch_idx * piv_batch_stride
    x_base = X + batch_idx * x_batch_stride
    col_offsets = cols * x_col_stride

    k = 0
    while k < n:
        ip = tl.load(piv_base + k * piv_row_stride).to(tl.int32)
        if ip > 0:
            kp = ip - 1
            row_k_ptr = x_base + k * x_row_stride + col_offsets
            row_k_ptr_i = row_k_ptr + 1
            if kp != k:
                row_kp_ptr = x_base + kp * x_row_stride + col_offsets
                row_kp_ptr_i = row_kp_ptr + 1
                xk_r = tl.load(row_k_ptr, mask=col_mask, other=0.0)
                xk_i = tl.load(row_k_ptr_i, mask=col_mask, other=0.0)
                xkp_r = tl.load(row_kp_ptr, mask=col_mask, other=0.0)
                xkp_i = tl.load(row_kp_ptr_i, mask=col_mask, other=0.0)
                tl.store(row_k_ptr, xkp_r, mask=col_mask)
                tl.store(row_k_ptr_i, xkp_i, mask=col_mask)
                tl.store(row_kp_ptr, xk_r, mask=col_mask)
                tl.store(row_kp_ptr_i, xk_i, mask=col_mask)

            zk_r = tl.load(row_k_ptr, mask=col_mask, other=0.0)
            zk_i = tl.load(row_k_ptr_i, mask=col_mask, other=0.0)
            for j in range(k):
                l_r = tl.load(ld_base + k * ld_row_stride + j * ld_col_stride)
                l_i = tl.load(
                    ld_base + k * ld_row_stride + j * ld_col_stride + 1
                )
                zj_r = tl.load(
                    x_base + j * x_row_stride + col_offsets, mask=col_mask, other=0.0
                )
                zj_i = tl.load(
                    x_base + j * x_row_stride + col_offsets + 1,
                    mask=col_mask,
                    other=0.0,
                )
                prod_r, prod_i = _cmul(l_r, l_i, zj_r, zj_i)
                zk_r -= prod_r
                zk_i -= prod_i

            # Store the undivided z; D^-1 is applied at the start of the
            # backward sweep (same convention as the real kernel).
            tl.store(row_k_ptr, zk_r, mask=col_mask)
            tl.store(row_k_ptr_i, zk_i, mask=col_mask)
            k += 1
        else:
            kp = -ip - 1
            row_k_ptr = x_base + k * x_row_stride + col_offsets
            row_k_ptr_i = row_k_ptr + 1
            row_k1_ptr = x_base + (k + 1) * x_row_stride + col_offsets
            row_k1_ptr_i = row_k1_ptr + 1
            if kp != k + 1:
                row_kp_ptr = x_base + kp * x_row_stride + col_offsets
                row_kp_ptr_i = row_kp_ptr + 1
                xk1_r = tl.load(row_k1_ptr, mask=col_mask, other=0.0)
                xk1_i = tl.load(row_k1_ptr_i, mask=col_mask, other=0.0)
                xkp_r = tl.load(row_kp_ptr, mask=col_mask, other=0.0)
                xkp_i = tl.load(row_kp_ptr_i, mask=col_mask, other=0.0)
                tl.store(row_k1_ptr, xkp_r, mask=col_mask)
                tl.store(row_k1_ptr_i, xkp_i, mask=col_mask)
                tl.store(row_kp_ptr, xk1_r, mask=col_mask)
                tl.store(row_kp_ptr_i, xk1_i, mask=col_mask)

            zk_r = tl.load(row_k_ptr, mask=col_mask, other=0.0)
            zk_i = tl.load(row_k_ptr_i, mask=col_mask, other=0.0)
            zk1_r = tl.load(row_k1_ptr, mask=col_mask, other=0.0)
            zk1_i = tl.load(row_k1_ptr_i, mask=col_mask, other=0.0)
            for j in range(k):
                l0_r = tl.load(ld_base + k * ld_row_stride + j * ld_col_stride)
                l0_i = tl.load(
                    ld_base + k * ld_row_stride + j * ld_col_stride + 1
                )
                l1_r = tl.load(
                    ld_base + (k + 1) * ld_row_stride + j * ld_col_stride
                )
                l1_i = tl.load(
                    ld_base + (k + 1) * ld_row_stride + j * ld_col_stride + 1
                )
                zj_r = tl.load(
                    x_base + j * x_row_stride + col_offsets, mask=col_mask, other=0.0
                )
                zj_i = tl.load(
                    x_base + j * x_row_stride + col_offsets + 1,
                    mask=col_mask,
                    other=0.0,
                )
                prod0_r, prod0_i = _cmul(l0_r, l0_i, zj_r, zj_i)
                prod1_r, prod1_i = _cmul(l1_r, l1_i, zj_r, zj_i)
                zk_r -= prod0_r
                zk_i -= prod0_i
                zk1_r -= prod1_r
                zk1_i -= prod1_i

            b_r = tl.load(ld_base + (k + 1) * ld_row_stride + k * ld_col_stride)
            b_i = tl.load(
                ld_base + (k + 1) * ld_row_stride + k * ld_col_stride + 1
            )
            a_r = tl.load(ld_base + k * ld_row_stride + k * ld_col_stride)
            a_i = tl.load(ld_base + k * ld_row_stride + k * ld_col_stride + 1)
            c_r = tl.load(
                ld_base + (k + 1) * ld_row_stride + (k + 1) * ld_col_stride
            )
            c_i = tl.load(
                ld_base + (k + 1) * ld_row_stride + (k + 1) * ld_col_stride + 1
            )
            if HERM:
                akm1_r, akm1_i = _cdiv(a_r, a_i, b_r, -b_i)
                ak_r, ak_i = _cdiv(c_r, c_i, b_r, b_i)
                denom_r, denom_i = _cmul(akm1_r, akm1_i, ak_r, ak_i)
                denom_r -= 1
                bkm1_r, bkm1_i = _cdiv(zk_r, zk_i, b_r, -b_i)
                bk_r, bk_i = _cdiv(zk1_r, zk1_i, b_r, b_i)
                tmp_r, tmp_i = _cmul(ak_r, ak_i, bkm1_r, bkm1_i)
                xk_r, xk_i = _cdiv(tmp_r - bk_r, tmp_i - bk_i, denom_r, denom_i)
                tmp_r, tmp_i = _cmul(akm1_r, akm1_i, bk_r, bk_i)
                xk1_r, xk1_i = _cdiv(tmp_r - bkm1_r, tmp_i - bkm1_i, denom_r, denom_i)
            else:
                akm1_r, akm1_i = _cdiv(a_r, a_i, b_r, b_i)
                ak_r, ak_i = _cdiv(c_r, c_i, b_r, b_i)
                denom_r, denom_i = _cmul(akm1_r, akm1_i, ak_r, ak_i)
                denom_r -= 1
                bkm1_r, bkm1_i = _cdiv(zk_r, zk_i, b_r, b_i)
                bk_r, bk_i = _cdiv(zk1_r, zk1_i, b_r, b_i)
                tmp_r, tmp_i = _cmul(ak_r, ak_i, bkm1_r, bkm1_i)
                xk_r, xk_i = _cdiv(tmp_r - bk_r, tmp_i - bk_i, denom_r, denom_i)
                tmp_r, tmp_i = _cmul(akm1_r, akm1_i, bk_r, bk_i)
                xk1_r, xk1_i = _cdiv(tmp_r - bkm1_r, tmp_i - bkm1_i, denom_r, denom_i)
            tl.store(row_k_ptr, xk_r, mask=col_mask)
            tl.store(row_k_ptr_i, xk_i, mask=col_mask)
            tl.store(row_k1_ptr, xk1_r, mask=col_mask)
            tl.store(row_k1_ptr_i, xk1_i, mask=col_mask)
            k += 2

    # Backward substitution: solve L^H * x = D^-1 * z.
    k = n - 1
    while k >= 0:
        ip = tl.load(piv_base + k * piv_row_stride).to(tl.int32)
        if ip > 0:
            kp = ip - 1
            row_k_ptr = x_base + k * x_row_stride + col_offsets
            row_k_ptr_i = row_k_ptr + 1
            acc_r = tl.load(row_k_ptr, mask=col_mask, other=0.0)
            acc_i = tl.load(row_k_ptr_i, mask=col_mask, other=0.0)
            # Apply D^-1 (rows hold the undivided z from the forward sweep).
            d_r = tl.load(ld_base + k * ld_row_stride + k * ld_col_stride)
            if HERM:
                inv_d = 1.0 / d_r
                acc_r = acc_r * inv_d
                acc_i = acc_i * inv_d
            else:
                d_i = tl.load(ld_base + k * ld_row_stride + k * ld_col_stride + 1)
                acc_r, acc_i = _cdiv(acc_r, acc_i, d_r, d_i)
            for j in range(k + 1, n):
                l_r = tl.load(ld_base + j * ld_row_stride + k * ld_col_stride)
                l_i = tl.load(ld_base + j * ld_row_stride + k * ld_col_stride + 1)
                xj_r = tl.load(
                    x_base + j * x_row_stride + col_offsets, mask=col_mask, other=0.0
                )
                xj_i = tl.load(
                    x_base + j * x_row_stride + col_offsets + 1,
                    mask=col_mask,
                    other=0.0,
                )
                if HERM:
                    prod_r, prod_i = _cconj_mul(l_r, l_i, xj_r, xj_i)
                else:
                    prod_r, prod_i = _cmul(l_r, l_i, xj_r, xj_i)
                acc_r -= prod_r
                acc_i -= prod_i
            if kp != k:
                row_kp_ptr = x_base + kp * x_row_stride + col_offsets
                row_kp_ptr_i = row_kp_ptr + 1
                xkp_r = tl.load(row_kp_ptr, mask=col_mask, other=0.0)
                xkp_i = tl.load(row_kp_ptr_i, mask=col_mask, other=0.0)
                tl.store(row_k_ptr, xkp_r, mask=col_mask)
                tl.store(row_k_ptr_i, xkp_i, mask=col_mask)
                tl.store(row_kp_ptr, acc_r, mask=col_mask)
                tl.store(row_kp_ptr_i, acc_i, mask=col_mask)
            else:
                tl.store(row_k_ptr, acc_r, mask=col_mask)
                tl.store(row_k_ptr_i, acc_i, mask=col_mask)
            k -= 1
        else:
            row_km1_ptr = x_base + (k - 1) * x_row_stride + col_offsets
            row_km1_ptr_i = row_km1_ptr + 1
            row_k_ptr = x_base + k * x_row_stride + col_offsets
            row_k_ptr_i = row_k_ptr + 1
            zkm1_r = tl.load(row_km1_ptr, mask=col_mask, other=0.0)
            zkm1_i = tl.load(row_km1_ptr_i, mask=col_mask, other=0.0)
            zk_r = tl.load(row_k_ptr, mask=col_mask, other=0.0)
            zk_i = tl.load(row_k_ptr_i, mask=col_mask, other=0.0)
            for j in range(k + 1, n):
                l0_r = tl.load(ld_base + j * ld_row_stride + (k - 1) * ld_col_stride)
                l0_i = tl.load(
                    ld_base + j * ld_row_stride + (k - 1) * ld_col_stride + 1
                )
                l1_r = tl.load(ld_base + j * ld_row_stride + k * ld_col_stride)
                l1_i = tl.load(ld_base + j * ld_row_stride + k * ld_col_stride + 1)
                xj_r = tl.load(
                    x_base + j * x_row_stride + col_offsets, mask=col_mask, other=0.0
                )
                xj_i = tl.load(
                    x_base + j * x_row_stride + col_offsets + 1,
                    mask=col_mask,
                    other=0.0,
                )
                if HERM:
                    prod0_r, prod0_i = _cconj_mul(l0_r, l0_i, xj_r, xj_i)
                    prod1_r, prod1_i = _cconj_mul(l1_r, l1_i, xj_r, xj_i)
                else:
                    prod0_r, prod0_i = _cmul(l0_r, l0_i, xj_r, xj_i)
                    prod1_r, prod1_i = _cmul(l1_r, l1_i, xj_r, xj_i)
                zkm1_r -= prod0_r
                zkm1_i -= prod0_i
                zk_r -= prod1_r
                zk_i -= prod1_i
            kp = -ip - 1
            if kp != k:
                row_kp_ptr = x_base + kp * x_row_stride + col_offsets
                row_kp_ptr_i = row_kp_ptr + 1
                xkp_r = tl.load(row_kp_ptr, mask=col_mask, other=0.0)
                xkp_i = tl.load(row_kp_ptr_i, mask=col_mask, other=0.0)
                tl.store(row_k_ptr, xkp_r, mask=col_mask)
                tl.store(row_k_ptr_i, xkp_i, mask=col_mask)
                tl.store(row_kp_ptr, zk_r, mask=col_mask)
                tl.store(row_kp_ptr_i, zk_i, mask=col_mask)
            else:
                tl.store(row_k_ptr, zk_r, mask=col_mask)
                tl.store(row_k_ptr_i, zk_i, mask=col_mask)
            tl.store(row_km1_ptr, zkm1_r, mask=col_mask)
            tl.store(row_km1_ptr_i, zkm1_i, mask=col_mask)
            k -= 2


def _validate_inputs(LD, pivots, B):
    if LD.device != B.device or LD.device != pivots.device:
        raise ValueError("LD, pivots, and B must be on the same device")
    if LD.dtype != B.dtype:
        raise TypeError("LD and B must have the same dtype")
    if LD.ndim < 2 or B.ndim < 2:
        raise ValueError("LD and B must be at least 2D")
    if LD.shape[-1] != LD.shape[-2]:
        raise ValueError("LD must be a square matrix or a batch of square matrices")
    if B.shape[-2] != LD.shape[-1]:
        raise ValueError("B must have shape (*, n, k) with the same n as LD")
    if pivots.shape != LD.shape[:-1]:
        raise ValueError("pivots must have shape (*, n) matching LD")
    if LD.shape[:-2] != B.shape[:-2]:
        raise ValueError("LD, pivots, and B must share the same batch dimensions")


def linalg_ldl_solve(LD, pivots, B, *, hermitian=False):
    """
    Solve a linear system using the compact LDL factorization produced by
    torch.linalg.ldl_factor_ex.

    Kunlunxin (batch3, 2026-09-10):
      * Two kernel paths:
        - fast (pull-based, register-accumulated): used when every pivot is
          identity (piv[k] == k+1).  This covers the SPD inputs produced by
          ldl_factor_ex for the tested cases; the row-dependency chain is an
          in-register dot product over the already-solved rows (no per-step
          load/store of the RHS rows), which is the fastest structure the
          XPU Triton backend accepts.
        - general (vendor push-based): used when any pivot performs a row
          interchange or a 2x2 block.  The interleaved-interchange semantics
          (the multiplier row used for a swap-carried value changes along
          its position journey) cannot be expressed by the pull form, so the
          vendor algorithm is kept verbatim for these inputs.
      * All masked loads use other=0.0.
    """
    logger.debug("GEMS LINALG_LDL_SOLVE")
    _validate_inputs(LD, pivots, B)

    if LD.dtype not in REAL_DTYPES + COMPLEX_DTYPES:
        raise TypeError(
            "linalg_ldl_solve supports only float32, float64, complex64, and complex128 inputs"
        )
    if LD.numel() == 0 or B.numel() == 0:
        return B.clone()

    batch = math.prod(LD.shape[:-2])
    n = LD.shape[-1]
    nrhs = B.shape[-1]
    piv_work = pivots.reshape(batch, n).contiguous()
    # Fast path requires identity pivots only (no interchanges, no 2x2 blocks).
    # The identity check costs ~0.1ms (one device->host sync); for tiny
    # matrices it exceeds the whole vendor kernel, so the vendor path is used
    # directly there (it is correct for every pivot pattern, so no check is
    # needed).  For n > 8 the fast path's savings outweigh the check.
    if n <= 8:
        identity_pivots = False
    else:
        identity_pivots = piv_work.flatten().cpu().tolist() == list(
            range(1, n + 1)
        ) * max(1, batch)

    with torch.no_grad():
        if LD.dtype in REAL_DTYPES:
            LD_work = LD.reshape(batch, n, n).contiguous()
            X = B.reshape(batch, n, nrhs).clone()
            if identity_pivots:
                num_slices = (nrhs + LDL_SOLVE_KS - 1) // LDL_SOLVE_KS
                grid = (batch * num_slices,)
                linalg_ldl_solve_real_fast_kernel[grid](
                    LD_work,
                    piv_work,
                    X,
                    n,
                    nrhs,
                    LD_work.stride(0),
                    LD_work.stride(1),
                    LD_work.stride(2),
                    piv_work.stride(0),
                    piv_work.stride(1),
                    X.stride(0),
                    X.stride(1),
                    X.stride(2),
                    KS=LDL_SOLVE_KS,
                    NS=num_slices,
                    num_warps=4,
                    isCloseVectorization=True,
                    buffer_size_limit=2048,
                )
            else:
                for col_start in range(0, nrhs, 64):
                    X_block = X[:, :, col_start : col_start + 64]
                    block_width = min(triton.next_power_of_2(X_block.shape[-1]), 64)
                    linalg_ldl_solve_real_kernel[(batch,)](
                        LD_work,
                        piv_work,
                        X_block,
                        n,
                        X_block.shape[-1],
                        LD_work.stride(0),
                        LD_work.stride(1),
                        LD_work.stride(2),
                        piv_work.stride(0),
                        piv_work.stride(1),
                        X_block.stride(0),
                        X_block.stride(1),
                        X_block.stride(2),
                        BLOCK_NRH=block_width,
                        NUM_RHS_BLOCKS=1,
                        num_warps=4,
                        isCloseVectorization=True,
                        buffer_size_limit=2048,
                    )
            return X.reshape(B.shape)

        # NOTE(Kunlunxin): .contiguous()/.clone() on a complex tensor dispatch
        # to the vendor copy_, which rejects complex dtypes
        # (NotImplementedError: copy_ for complex tensors is not supported).
        # Convert to the real view first (float) so the copy goes through the
        # float path of vendor copy_, which is supported.
        LD_work = torch.view_as_real(LD).reshape(batch, n, n, 2).contiguous()
        X = torch.view_as_real(B).reshape(batch, n, nrhs, 2).contiguous()
        if identity_pivots:
            num_slices = (nrhs + LDL_SOLVE_KS - 1) // LDL_SOLVE_KS
            grid = (batch * num_slices,)
            linalg_ldl_solve_complex_fast_kernel[grid](
                LD_work,
                piv_work,
                X,
                n,
                nrhs,
                LD_work.stride(0),
                LD_work.stride(1),
                LD_work.stride(2),
                piv_work.stride(0),
                piv_work.stride(1),
                X.stride(0),
                X.stride(1),
                X.stride(2),
                HERM=hermitian,
                KS=LDL_SOLVE_KS,
                NS=num_slices,
                num_warps=4,
                isCloseVectorization=True,
                buffer_size_limit=2048,
            )
        else:
            for col_start in range(0, nrhs, 64):
                X_block = X[:, :, col_start : col_start + 64]
                block_width = min(triton.next_power_of_2(X_block.shape[-2]), 64)
                linalg_ldl_solve_complex_kernel[(batch, 1)](
                    LD_work,
                    piv_work,
                    X_block,
                    n,
                    X_block.shape[-2],
                    LD_work.stride(0),
                    LD_work.stride(1),
                    LD_work.stride(2),
                    piv_work.stride(0),
                    piv_work.stride(1),
                    X_block.stride(0),
                    X_block.stride(1),
                    X_block.stride(2),
                    HERM=hermitian,
                    BLOCK_NRH=block_width,
                    num_warps=4,
                    isCloseVectorization=True,
                    buffer_size_limit=2048,
                )
        return torch.view_as_complex(X).reshape(B.shape)

