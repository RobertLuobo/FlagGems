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
from _kunlunxin.utils.codegen_config_utils import CodeGenConfig

from ..utils.pointwise_dynamic import pointwise_dynamic

logger = logging.getLogger(__name__)

config_ = CodeGenConfig(
    512,
    (65536, 65536, 65536),
    32,
    True,
    prefer_1d_tile=True,
    isCloseMemoryAsync=False,
    kunlunAutoGrid=True,
    unroll_num=8,
)


@pointwise_dynamic(promotion_methods=[(0, 1, "DEFAULT")], config=config_)
@triton.jit
def bitwise_or_func(x, y):
    return x | y


def bitwise_or_tensor(A, B):
    logger.debug("GEMS_KUNLUNXIN BITWISE_OR")
    return bitwise_or_func(A, B)


def bitwise_or_tensor_(A, B):
    logger.debug("GEMS_KUNLUNXIN BITWISE_OR_")
    return bitwise_or_func(A, B, out0=A)


@pointwise_dynamic(is_tensor=[True, False], promotion_methods=[(0, 1, "DEFAULT")], config=config_)
@triton.jit
def bitwise_or_func_scalar(x, y):
    return x | y


def bitwise_or_scalar(A, B):
    logger.debug("GEMS_KUNLUNXIN BITWISE_OR_SCALAR")
    # int32-word packing fast path for bool/int16, mirroring the (verified)
    # bitwise_or_scalar_ in-place recipe below / bitwise_or_scalar_tensor: a
    # byte / 16-bit load-store pays a heavy per-byte penalty on XPU, while 4
    # bools (or 2 int16s) fit one int32 word whose bytes / 16-bit lanes are
    # uniformly OR-ed with the replicated scalar (bool takes the low bit of
    # the two's-complement scalar, int16 the low 16 bits -- both match torch).
    # Out-of-place: the OR-ed int32 words go to a freshly allocated tensor
    # (the kernel treats the int32 view of A as read-only input), which is
    # then viewed back to A's dtype/shape -- functional semantics, the
    # result never aliases A.
    # Gating: the bool lane only packs for a Python *bool* scalar; a Python
    # int with a bool tensor type-promotes to Long in torch (2|x != (2&1)|x),
    # so such inputs keep the generic scalar kernel (long-faithful, unchanged
    # behavior). int16 packs for any int/bool scalar (torch truncates to low
    # 16 bits). Restricted to contiguous inputs with a full int32-aligned
    # byte count; anything else (non-contiguous, tail bytes, 0-dim/empty,
    # int32/int64) falls back to the generic scalar kernel.
    if (
        A.dtype in (torch.bool, torch.int16)
        and A.is_contiguous()
        and isinstance(B, (int, bool))
    ):
        nbytes = A.numel() * A.element_size()
        if nbytes > 0 and nbytes % 4 == 0:
            scalar = int(B)
            if A.dtype == torch.bool:
                if type(B) is not bool:
                    return bitwise_or_func_scalar(A, B)
                mask = 0x01010101 if (scalar & 1) else 0
            else:
                s = scalar & 0xFFFF
                mask = s | (s << 16)
            try:
                in_view = A.reshape(-1).view(torch.int32)
            except RuntimeError:  # e.g. unaligned storage offset
                return bitwise_or_func_scalar(A, B)
            res = bitwise_or_func_scalar(in_view, mask)
            return res.view(A.dtype).reshape(A.shape)
    return bitwise_or_func_scalar(A, B)


def bitwise_or_scalar_(A, B):
    logger.debug("GEMS_KUNLUNXIN BITWISE_OR_SCALAR_")
    # int32-word packing fast path for bool/int16 in-place, mirroring the
    # (verified) bitwise_or_scalar_tensor recipe of the same file: a byte /
    # 16-bit load-store pays a heavy per-byte penalty on XPU, while 4 bools
    # (or 2 int16s) fit one int32 word whose bytes / 16-bit lanes are
    # uniformly OR-ed with the replicated scalar (bool takes the low bit of
    # the two's-complement scalar, int16 the low 16 bits -- both match torch).
    # In-place: the OR-ed int32 words are written directly back into A's own
    # storage (out0=in_view), preserving alias / mutation semantics.
    # Gating: the bool lane only packs for a Python *bool* scalar; a Python
    # int with a bool tensor type-promotes to Long in torch and the in-place
    # op even raises (result type Long can't be cast to Bool), so such inputs
    # keep the generic scalar kernel (long-faithful, unchanged behavior).
    # int16 packs for any int/bool scalar (torch truncates to low 16 bits).
    # Restricted to contiguous inputs with a full int32-aligned byte count;
    # anything else (non-contiguous, tail bytes, 0-dim/empty, int32/int64)
    # falls back to the generic scalar kernel.
    if (
        A.dtype in (torch.bool, torch.int16)
        and A.is_contiguous()
        and isinstance(B, (int, bool))
    ):
        nbytes = A.numel() * A.element_size()
        if nbytes > 0 and nbytes % 4 == 0:
            scalar = int(B)
            if A.dtype == torch.bool:
                if type(B) is not bool:
                    return bitwise_or_func_scalar(A, B, out0=A)
                mask = 0x01010101 if (scalar & 1) else 0
            else:
                s = scalar & 0xFFFF
                mask = s | (s << 16)
            try:
                in_view = A.reshape(-1).view(torch.int32)
            except RuntimeError:  # e.g. unaligned storage offset
                return bitwise_or_func_scalar(A, B, out0=A)
            bitwise_or_func_scalar(in_view, mask, out0=in_view)
            return A
    return bitwise_or_func_scalar(A, B, out0=A)


def bitwise_or_scalar_tensor(A, B):
    logger.debug("GEMS_KUNLUNXIN BITWISE_OR_SCALAR_TENSOR")
    # Fast path for sub-32-bit dtypes, mirroring bitwise_and_scalar_tensor.
    # On XPU a byte (bool) / 16-bit (int16) load/store pays a heavy per-byte
    # penalty vs an int32 word load. Bitwise OR with a *uniform* scalar is
    # trivially packable: 4 bools (or 2 int16s) fit one int32 word and the
    # scalar only has to be replicated to every byte / 16-bit lane of the word
    # (scalar truncation to 1 / 16 bits matches torch: bool takes the low bit
    # of the two's-complement scalar, int16 takes the low 16 bits). The
    # int32-view kernel then runs at the full int32 load/store bandwidth.
    # Gating: the bool lane is only packed when the scalar is a Python *bool*.
    # torch's `bitwise_or(int, bool_tensor)` type-promotes to int64 (the full
    # int is OR-ed, e.g. 2|x -> 2/3), which a per-byte mask cannot reproduce;
    # such inputs fall back to the generic scalar kernel (int64-faithful).
    # int16 packs for any int/bool scalar (torch truncates to low 16 bits).
    # Restricted to contiguous inputs with a full int32-aligned byte count;
    # anything else (non-contiguous, tail bytes, 0-dim/empty, int32/int64)
    # falls back to the generic scalar kernel.
    if (
        B.dtype in (torch.bool, torch.int16)
        and B.is_contiguous()
        and isinstance(A, (int, bool))
    ):
        nbytes = B.numel() * B.element_size()
        if nbytes > 0 and nbytes % 4 == 0:
            scalar = int(A)
            if B.dtype == torch.bool:
                if type(A) is not bool:
                    return bitwise_or_func_scalar(B, A)
                # torch bool conversion of a scalar = low bit (verified:
                # 2->False, 3->True, 5->True, -2->False on reference).
                mask = 0x01010101 if (scalar & 1) else 0
            else:
                s = scalar & 0xFFFF
                mask = s | (s << 16)
            n_words = nbytes // 4
            out = torch.empty_strided(
                (n_words,), (1,), dtype=torch.int32, device=B.device
            )
            try:
                in_view = B.reshape(-1).view(torch.int32)
            except RuntimeError:  # e.g. unaligned storage offset
                return bitwise_or_func_scalar(B, A)
            bitwise_or_func_scalar(in_view, mask, out0=out)
            return out.view(B.dtype).reshape(B.shape)
    return bitwise_or_func_scalar(B, A)
