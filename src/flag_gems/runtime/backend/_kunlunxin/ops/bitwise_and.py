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
    # isCloseMemoryAsync must stay at its default (True = async copy closed).
    # Enabling async copy (=False) together with unroll_num=8 makes the LLVM
    # lowering materialize a ~478-pointer local-buffer struct that is re-printed
    # on every insertvalue, blowing the compiled IR up to ~9GB (see
    # benchmark/ir_dump/ir-bitwise_and_tensor-dev5.log). unroll_num/autoGrid are
    # kept for the #1277 speedup; only the async pipeline is dropped.
    kunlunAutoGrid=True,
    unroll_num=8,
)


@pointwise_dynamic(promotion_methods=[(0, 1, "DEFAULT")], config=config_)
@triton.jit
def bitwise_and_func(x, y):
    return x & y


def bitwise_and_tensor(A, B):
    logger.debug("GEMS_KUNLUNXIN BITWISE_AND")
    # Hand pointwise_dynamic a pre-allocated result as `out0`.  Without it the
    # wrapper re-derives the promoted dtype on every call (type_promotion) and
    # allocates the output itself through the *registered* torch.empty_like; at
    # the (64,64) launch-bound shape that host work dominates the op (measured
    # ~15-18us -> ~5.4us, single-variable A/B, harness/solution/
    # bitwise_tensor_family/README_out0.md).  The (4096,4096) cases are device
    # bound and are unaffected.
    #
    # Gate: only take the fast path when the promoted result is provably
    # identical to a fresh tensor of A's own dtype/shape/layout, i.e.
    #   * A.dtype == B.dtype          -> promotion picks that very dtype
    #   * A.shape == B.shape          -> broadcast shape == A.shape
    #   * both contiguous             -> empty_strided(A.shape, A.stride())
    #                                    reproduces exactly the empty_like(A)
    #                                    buffer the wrapper would have built
    # Anything else (mixed dtype, broadcast shapes, non-contiguous operands)
    # keeps the generic path, bit-for-bit unchanged.
    if (
        A.dtype == B.dtype
        and A.shape == B.shape
        and A.is_contiguous()
        and B.is_contiguous()
    ):
        out = torch.empty_strided(
            A.shape, A.stride(), dtype=A.dtype, device=A.device
        )
        return bitwise_and_func(A, B, out0=out)
    return bitwise_and_func(A, B)


def bitwise_and_tensor_(A, B):
    logger.debug("GEMS_KUNLUNXIN BITWISE_AND_")
    return bitwise_and_func(A, B, out0=A)


# Scalar (tensor-vs-scalar) path.
@pointwise_dynamic(
    is_tensor=[True, False], promotion_methods=[(0, 1, "DEFAULT")], config=config_
)
@triton.jit
def bitwise_and_func_scalar(x, y):
    return x & y


def bitwise_and_scalar(A, B):
    logger.debug("GEMS_KUNLUNXIN BITWISE_AND_SCALAR")
    # Same int32-word packing as bitwise_and_scalar_/bitwise_and_scalar_tensor
    # below: the int16 element-wise kernel is pathological on this backend
    # (same element count, 3.7x slower than int32 -- see
    # harness/meta/TODO.md#bitwise_and_scalar), while the int32-word kernel
    # runs at the int32 rate. bool/int16 lane semantics are preserved exactly:
    # `int16_tensor & s` wraps the scalar into int16, and AND on a 16-bit lane
    # is interpretation-independent, so the duplicated word mask is bit-exact.
    #
    # Size gate (measured, see harness/solution/bitwise_and_scalar/
    # README_pack_gate.md): the crossover is dtype-dependent, so it is NOT a
    # single nbytes threshold.
    #   * int16: packing never loses -- 0.91x/0.80x/0.81x at 512B/4KiB/8KiB and
    #     0.35x at >=64MiB vs the int16-lane kernel (9 rounds, official
    #     do_bench caliber, cross-round median). Ungated.
    #   * bool : packing costs ~+2us of host time and LOSES at <=2KiB
    #     (1.10-1.17x over 2 independent runs) and at 4KiB in one 9-round run
    #     (1.33x, while a 21-round run says 0.99x -- the 4KiB cell is the noisy
    #     one); from 8KiB up it wins in 2 of 3 runs and is never >1.01x
    #     (8KiB 0.94-1.00, 16KiB 0.86-1.00, 32KiB 0.82-0.97, 64KiB 0.53,
    #     128KiB 0.59, 128MiB 0.15). Gate at 8KiB.
    # NOTE: the benchmark's bool case feeds an int scalar (0x3F), which promotes
    # to int64 and never reaches this packing at all -- this gate is for real
    # bool-scalar callers (`t & True`), not for the benchmark number.
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
                    # bool_tensor & int promotes to int64 on reference
                    # (verified), i.e. not a bool-lane op at all -> no packing.
                    return bitwise_and_func_scalar(A, B)
                if nbytes < 8 * 1024:
                    # measured bool crossover (3 independent runs, 9/15/21
                    # rounds): packing is a consistent loss at <=2KiB and a
                    # 1.33x loss in one 9-round run at 4KiB, while 8KiB+ wins in
                    # 2 of 3 runs and never loses >1.01x. 8KiB is a 4x margin
                    # over the agreed 2KiB loss point.
                    return bitwise_and_func_scalar(A, B)
                # torch bool conversion of a scalar = low bit (verified:
                # 2->False, 3->True, 5->True, -2->False on reference).
                mask = 0x01010101 if (scalar & 1) else 0
            else:
                if not -0x8000 <= scalar <= 0x7FFF:
                    return bitwise_and_func_scalar(A, B)
                s = scalar & 0xFFFF
                mask = s | (s << 16)
            try:
                in_view = A.reshape(-1).view(torch.int32)
            except RuntimeError:  # e.g. unaligned storage offset
                return bitwise_and_func_scalar(A, B)
            n_words = nbytes // 4
            out = torch.empty_strided(
                (n_words,), (1,), dtype=torch.int32, device=A.device
            )
            bitwise_and_func_scalar(in_view, mask, out0=out)
            return out.view(A.dtype).reshape(A.shape)
    return bitwise_and_func_scalar(A, B)


def bitwise_and_scalar_(A, B):
    logger.debug("GEMS_KUNLUNXIN BITWISE_AND_SCALAR_")
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
                    return bitwise_and_func_scalar(A, B, out0=A)
                # torch bool conversion of a scalar = low bit (verified:
                # 2->False, 3->True, 5->True, -2->False on reference).
                mask = 0x01010101 if (scalar & 1) else 0
            else:
                if not -0x8000 <= scalar <= 0x7FFF:
                    return bitwise_and_func_scalar(A, B, out0=A)
                s = scalar & 0xFFFF
                mask = s | (s << 16)
            try:
                in_view = A.reshape(-1).view(torch.int32)
            except RuntimeError:  # e.g. unaligned storage offset
                return bitwise_and_func_scalar(A, B, out0=A)
            bitwise_and_func_scalar(in_view, mask, out0=in_view)
            return A
    return bitwise_and_func_scalar(A, B, out0=A)


def bitwise_and_scalar_tensor(A, B):
    logger.debug("GEMS_KUNLUNXIN BITWISE_AND_SCALAR_TENSOR")
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
                    return bitwise_and_func_scalar(B, A)
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
                return bitwise_and_func_scalar(B, A)
            bitwise_and_func_scalar(in_view, mask, out0=out)
            return out.view(B.dtype).reshape(B.shape)
    return bitwise_and_func_scalar(B, A)
