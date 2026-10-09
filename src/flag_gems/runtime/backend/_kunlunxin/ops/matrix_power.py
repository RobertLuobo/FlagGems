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

from .linalg_matrix_power import (
    _ensure_contiguous,
    _gems_copy,
    _inverse,
    _make_identity_like,
    _matmul,
)

logger = logging.getLogger(__name__)

_SUPPORTED_DTYPES = (
    torch.float16,
    torch.bfloat16,
    torch.float32,
    torch.float64,
)

# Low-precision dtypes are promoted to fp32 for the whole repeated-squaring
# chain so that intermediate products are not rounded back to bf16/fp16 after
# every matmul. Rounding each squaring step compounds multiplicatively and
# diverges from the reference several times faster than the reference's own
# per-step bf16/fp16 rounding; keeping the chain in fp32 brings the gems result
# as close to the exact value as the vendor GEMM allows (verified).
_PROMOTE_DTYPES = (torch.float16, torch.bfloat16)


def _cast(t: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """dtype conversion via the gems pointwise copy kernel (no torch fallback)."""
    if t.dtype == dtype:
        return t
    dst = torch.empty(t.shape, dtype=dtype, device=t.device)
    return _gems_copy(dst, t)


def _validate(A, n):
    if A.ndim < 2:
        raise RuntimeError(
            f"matrix_power: A must be at least 2-D, got shape {tuple(A.shape)}"
        )
    if A.shape[-1] != A.shape[-2]:
        raise RuntimeError(
            f"matrix_power: A must be square, got ({A.shape[-2]}, {A.shape[-1]})"
        )
    if not isinstance(n, int):
        raise TypeError(f"matrix_power: n must be int, got {type(n).__name__}")
    if A.dtype not in _SUPPORTED_DTYPES:
        raise RuntimeError(
            f"matrix_power only supports float16, bfloat16, float32, float64, "
            f"got {A.dtype}"
        )


def _power(A, n):
    """A ** n for a batched square matrix routed entirely through vendor
    GEMM / LU kernels (no torch numeric fallback). ``A`` is already a
    contiguous (batch, M, M) tensor and ``n`` has been validated."""
    if n < 0:
        if A.dim() == 2:
            A = _inverse(A.unsqueeze(0)).squeeze(0)
        else:
            A = _inverse(A)
        n = -n

    result = None
    base = A
    remaining = n
    while remaining > 0:
        if remaining & 1:
            result = base if result is None else _matmul(result, base)
        remaining >>= 1
        if remaining > 0:
            base = _matmul(base, base)
    return result


def matrix_power(A, n):
    logger.debug("GEMS_KUNLUNXIN MATRIX_POWER")
    _validate(A, n)

    M = A.shape[-1]
    batch_shape = A.shape[:-2]

    if n == 0:
        eye = _make_identity_like(A.reshape(-1, M, M) if A.dim() != 2 else A)
        return eye.reshape(*batch_shape, M, M)

    if A.dtype in (torch.float16, torch.bfloat16) and n < 0:
        # Matrix inversion is only numerically defined via the fp32/fp64 LU
        # path; torch itself refuses half-precision negative powers.
        raise RuntimeError(
            f"matrix_power: negative powers require float32/float64, got {A.dtype}"
        )

    A3 = _ensure_contiguous(A.reshape(-1, M, M) if A.dim() != 2 else A)

    if n == 1:
        res = torch.empty(A3.shape, dtype=A3.dtype, device=A3.device)
        res = _gems_copy(res, A3)
        return res.reshape(*batch_shape, M, M)

    orig_dtype = A3.dtype
    compute = A3
    if orig_dtype in _PROMOTE_DTYPES:
        compute = _cast(A3, torch.float32)

    out = _power(compute, n)

    if out.dtype != orig_dtype:
        out = _cast(_ensure_contiguous(out), orig_dtype)
    return out.reshape(*batch_shape, M, M)


def matrix_power_out(A, n, *, out):
    logger.debug("GEMS_KUNLUNXIN MATRIX_POWER_OUT")
    result = matrix_power(A, n)
    return _gems_copy(out, result)
