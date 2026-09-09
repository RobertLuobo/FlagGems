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

import functools
import logging
import math
import os

import torch
import triton
import triton.language as tl
from _kunlunxin.utils.codegen_config_utils import CodeGenConfig
from flag_gems.runtime import torch_device_fn

from ..utils.pointwise_dynamic import pointwise_dynamic

# ---------------------------------------------------------------------------
# tle.raw fast path for the tensor-vs-scalar less compare (P800 xpu3,
# cluster C payload in lt_raw.xpu). The compiler cannot vectorize a
# tensor-vs-scalar float compare (CoreTiling hands the scalar kernel 8
# elements/core/iteration, below the f32 vector width of 16, so `arith.cmpf`
# stays scalarized -- ~370us on 16M elements; TRITONXPU_VEC_REPORT shows the
# closure vetoed with keyState=Conflict). The hand-written payload streams
# the input once per core with pipelined GM2LM/LM2GM DMA and compares with
# the hardware vector-lt intrinsics (x < s with the vector on the LEFT:
# vvlt(x, splat(s)), the vendor xdnn SWAP=false recipe), with the same memory
# footprint as the ATen reference. It also matches ATen's scalar-dtype
# promotion exactly. Mirrors the not_equal/greater scalar payloads
# (ne_raw.xpu / gt_raw.xpu); see lt_raw.xpu for the full analysis.
try:
    import triton.experimental.tle as tle

    _TLE_OK = True
except ImportError:
    tle = None
    _TLE_OK = False

_HERE = os.path.dirname(os.path.abspath(__file__))
_NCLUSTER = 12  # P800 (xpu3): one Triton program == one cluster of 64 cores
# Payload scalars are i32 (do_not_specialize); guard the byte range.
_RAW_MAX_ELEMS = 2**31 - 1
# Must match CHUNK_BYTES in lt_raw.xpu (the chunk-grid partition contract).
_RAW_CHUNK_BYTES = 2048
# Below this element count the op is launch-bound and the bare pointwise
# scalar kernel (single launch, no extra host work) is fine; the payload wins
# from ~64K elements up (same crossover as the not_equal/greater payloads).
_RAW_SMALL_SCALAR_LIMIT = 65536

_RAW_TYPE_CODE = {
    torch.float32: 0,
    torch.float16: 1,
    torch.bfloat16: 2,
}

if _TLE_OK:

    @tle.raw.dialect("xpu3", file=os.path.join(_HERE, "lt_raw.xpu"))
    def lt_scalar_raw(in_, out, numel, esz, type_code, scalar_bits,
                      chunk_start, chunk_count):
        ...

    @triton.jit(do_not_specialize=["numel", "esz", "type_code", "scalar_bits",
                                   "chunk_count"])
    def lt_scalar_raw_kernel(In, Out, numel, esz, type_code, scalar_bits,
                             chunk_count):
        pid = tl.program_id(0)
        tle.raw.call(lt_scalar_raw, (In, Out, numel, esz, type_code,
                                     scalar_bits, pid * chunk_count,
                                     chunk_count))


def _view_u8(t):
    """Byte view of a tensor; works for 0-dim tensors too."""
    if t.dim() == 0:
        return t.view(1).view(torch.uint8)
    return t.view(torch.uint8)


@functools.lru_cache(maxsize=1024)
def _scalar_bits(B, dtype):
    """The scalar promoted to `dtype`, as a sign-extended int32 bit pattern.

    Matches torch.less's type promotion: the python float scalar is converted
    to the tensor's dtype and compared in that dtype. Cached: the dtype
    conversion costs a couple of microseconds on the host, which is directly
    visible on launch-bound small shapes.
    """
    if dtype == torch.float32:
        return int(torch.tensor(B, dtype=torch.float32).view(torch.int32).item())
    # fp16 / bf16: the payload only reads the low 16 bits.
    return int(torch.tensor(B, dtype=dtype).view(torch.int16).item())


def _raw_lt_scalar(A, B):
    """less(A, scalar) via the raw payload, or None when it does not apply.

    Only the contiguous case in the supported float dtypes is handled;
    anything else falls back to the pointwise scalar kernel below.
    """
    if not _TLE_OK or not A.is_contiguous():
        return None
    type_code = _RAW_TYPE_CODE.get(A.dtype)
    if type_code is None:
        return None
    M = A.numel()
    if M == 0 or M > _RAW_MAX_ELEMS:
        return None
    esz = A.element_size()
    s_bits = _scalar_bits(B, A.dtype)
    out = torch.empty(A.shape, dtype=torch.bool, device=A.device)
    # partition by payload chunks (CHUNK_BYTES/esz elements each): every
    # program and core gets whole chunks so all GM2LM/LM2GM transfers are
    # CHUNK_BYTES-aligned in global memory.
    chunk_elems = _RAW_CHUNK_BYTES // esz
    total_chunks = (M + chunk_elems - 1) // chunk_elems
    per = (total_chunks + _NCLUSTER - 1) // _NCLUSTER
    with torch_device_fn.device(A.device):
        lt_scalar_raw_kernel[(_NCLUSTER,)](
            _view_u8(A), _view_u8(out), M, esz, type_code, s_bits, per)
    return out

logger = logging.getLogger(__name__)

config_ = CodeGenConfig(
    512,
    (65536, 65536, 65536),
    32,
    True,
    prefer_1d_tile=True,
    isCloseMemoryAsync=False,
    unroll_num=8,
    kunlunAutoGrid=True,
)


@pointwise_dynamic(
    promotion_methods=[(0, 1, "ALWAYS_BOOL")],
    config=config_,
)
@triton.jit
def lt_func(x, y):
    return x.to(tl.float32) < y


def lt(A, B):
    logger.debug("GEMS_KUNLUNXIN LT")
    os.environ["TRITONXPU_COMPARE_FUSION"] = "1"
    os.environ["TRITONXPU_FP16_FAST"] = "1"
    res = lt_func(A, B)
    del os.environ["TRITONXPU_COMPARE_FUSION"]
    del os.environ["TRITONXPU_FP16_FAST"]
    return res


@pointwise_dynamic(
    is_tensor=[True, False],
    promotion_methods=[(0, 1, "ALWAYS_BOOL")],
    config=config_,
)
@triton.jit
def lt_func_scalar(x, y):
    return x.to(tl.float32) < y


def lt_scalar(A, B):
    logger.debug("GEMS_KUNLUNXIN LT_SCALAR")
# Fast path: hand-written cluster-C payload (lt_raw.xpu). The compiler
    # scalarizes the tensor-vs-scalar compare (~370us on 16M elements, see the
    # header comment); the payload streams A once with per-core pipelined DMA
    # and the hardware vector-lt intrinsics at the same memory footprint as
    # ATen, and matches ATen's scalar-dtype promotion exactly.
    if A.numel() >= _RAW_SMALL_SCALAR_LIMIT:
        raw_out = _raw_lt_scalar(A, B)
        if raw_out is not None:
            return raw_out
    numel = A.numel()
    dtype = A.dtype
    if (
        A.is_contiguous()
        and dtype in (torch.float16, torch.float32, torch.bfloat16)
        and float(B) == float(torch.tensor(float(B), dtype=dtype).item())
    ):
        if (
            numel >= _LT_SCALAR_FAST_TILE * _LT_SCALAR_MIN_GRID
            and numel % _LT_SCALAR_FAST_TILE == 0
        ):
            # exact-multiple flat tiles (grid = numel / TILE >= MIN_GRID): no
            # mask, no i1 -- a saturating fp32 store + vendor bool conversion.
            return _lt_scalar_fast(A, float(B), (numel // _LT_SCALAR_FAST_TILE,))
        if numel >= _LT_SCALAR_MASKED_MIN and numel % _LT_SCALAR_FAST_TILE != 0:
            # non-multiple mid sizes (e.g. 2.56M): flat tiles with a real tail
            # mask. The mask is genuine (tail elements), so the masked-memory
            # path is the only penalty and the i1/bool-store catastrophe is
            # still avoided.
            return _lt_scalar_fast_masked(A, float(B), numel)
    # Generic path: match torch's type promotion for the scalar too. torch
    # wraps the scalar into the input dtype first (e.g. fp16 tensor vs 0.1
    # compares against (fp16)0.1 = 0.0999756), so a raw fp32 compare against
    # the unwrapped scalar would diverge for non-representable values. This
    # only matters for the generic path: the fast paths are gated on exact
    # representability above.
    if dtype in (torch.float16, torch.float32, torch.bfloat16):
        B = float(torch.tensor(float(B), dtype=dtype).item())
    res = lt_func_scalar(A, B)
    return res


# ---------------------------------------------------------------------------
# lt_scalar fast paths (fp16/fp32/bf16, contiguous) -- same two-stage recipe
# as gt_scalar (sibling file): no i1 is ever materialized in Triton.
#
# Why: the generic scalar-compare path (pointwise_dynamic 1d-tile codegen)
# always emits `mask = tid < num_tasks` and materializes
# `arith.cmpf -> i1 -> bool store` per lane. On XPU both are slow: the
# always-true runtime mask goes through the masked-memory path, and the
# i1/bool store lowers to a per-lane slow path (~10x). Previous lt_scalar
# fast kernel stored `x < scalar` (an i1) directly into a bool tensor -- it
# only removed the always-true mask, so the big shapes stayed at
# ~5.4 ms / ~13.2 ms on [268435456] / [10000, 65536] (speedup 0.083-0.18).
#
#   1. saturating fp arithmetic on the fp32-upcast value:
#      t = (scalar - x) * 1e30; max(0,t); min(1,t) -> exactly 0.0/1.0 written
#      into a fp32 buffer (M = 1e30 saturates every representable x != s gap
#      of fp16/bf16/fp32; eq/gap exact; for x < s with subnormal gap the
#      product stays a positive value in (0, 1), still nonzero -> True).
#      NaN -> 0 (max/min prefer the non-NaN operand; torch: NaN < s == False).
#      +-0, +-inf are exact.
#   2. fp32 -> bool via `torch.ops.aten._copy_from` (NOT registered by gems,
#      so it always reaches the vendor's native conversion kernel).
#
# The scalar gate `float(B) == float(torch.tensor(B, dtype=A.dtype).item())`
# only admits scalars exactly representable in A.dtype: torch compares against
# the scalar rounded to the input dtype (wrapped scalar), and restricting to
# representable scalars (benchmark 0, test 0) makes the fp32 compare
# bit-identical to torch semantics. Anything else (e.g. fp16 scalar 1e-10,
# 0.1 non-representable) keeps the generic path, unchanged behavior.
#
# fp16/bf16 intermediate buffers were probed and rejected (see gt_scalar): the
# vendor's fp16/bf16 -> bool conversion is ~3x slower than fp32 -> bool, so
# the fp32 intermediate wins despite the extra bytes.
_LT_SCALAR_FAST_TILE = 131072
_LT_SCALAR_MIN_GRID = 128
_LT_SCALAR_MASKED_MIN = 1 << 20


@triton.jit
def lt_scalar_fast_kernel(out_ptr, x_ptr, scalar, TILE: tl.constexpr):
    pid = tl.program_id(0)
    tid = pid * TILE + tl.arange(0, TILE)
    x = tl.load(x_ptr + tid).to(tl.float32)
    t = (scalar - x) * 1.0e30
    t = tl.maximum(0.0, t)
    t = tl.minimum(1.0, t)
    tl.store(out_ptr + tid, t)


def _lt_scalar_fast(A, scalar, grid):
    out32 = torch.empty_like(A, dtype=torch.float32)
    lt_scalar_fast_kernel[grid](
        out32,
        A,
        scalar,
        TILE=_LT_SCALAR_FAST_TILE,
        num_warps=4,
        buffer_size_limit=8192,
        unroll_num=16,
        isCloseMemoryAsync=False,
    )
    out = torch.empty_like(A, dtype=torch.bool)
    torch.ops.aten._copy_from(out32, out, False)
    return out


@triton.jit
def lt_scalar_fast_masked_kernel(out_ptr, x_ptr, scalar, numel, TILE: tl.constexpr):
    pid = tl.program_id(0)
    tid = pid * TILE + tl.arange(0, TILE)
    mask = tid < numel
    x = tl.load(x_ptr + tid, mask=mask).to(tl.float32)
    t = (scalar - x) * 1.0e30
    t = tl.maximum(0.0, t)
    t = tl.minimum(1.0, t)
    tl.store(out_ptr + tid, t, mask=mask)


def _lt_scalar_fast_masked(A, scalar, numel):
    out32 = torch.empty_like(A, dtype=torch.float32)
    grid = (math.ceil(numel / _LT_SCALAR_FAST_TILE),)
    lt_scalar_fast_masked_kernel[grid](
        out32,
        A,
        scalar,
        numel,
        TILE=_LT_SCALAR_FAST_TILE,
        num_warps=4,
        buffer_size_limit=8192,
        unroll_num=16,
        isCloseMemoryAsync=False,
    )
    out = torch.empty_like(A, dtype=torch.bool)
    torch.ops.aten._copy_from(out32, out, False)
    return out


# lt_ / lt_scalar_ are the in-place aliases of lt (out = (x < y) written back
# into x). They were NOT overridden by kunlunxin, so they fell to the generic
# ops/lt_.py -- a bare `@pointwise_dynamic` with NO CodeGenConfig -> discrete /
# launch-bound slow path. Baseline IR (ir-lt_-dev0 / ir-lt_scalar_-dev1) shows
# 512-wide discrete masked stores (`tt.ptr<f16, 0>` + per-element i32 column
# offsets) -> catastrophic latency on large shapes.
#
# Fix: a dedicated tuned pointwise_dynamic that writes in place (out0=A).
# CRITICAL: this in-place variant must NOT reuse lt's config_ -- config_ has
# `isCloseMemoryAsync=False` (async memory copy ON), and with in-place aliasing
# (input tensor == output tensor) the async double-buffered copy path deadlocks
# the device ("noc idle timeout" hang). The out-of-place lt is fine because its
# output is a fresh bool tensor (no aliasing). So use a config with the DEFAULT
# isCloseMemoryAsync (True = async closed), mirroring the proven in-place op
# greater_equal_. Body returns tl.where(...,1,0) (int 0/1) which stores cleanly
# into A's original fp16/bf16/fp32 dtype. The scalar path additionally must NOT
# set the TRITONXPU_COMPARE_FUSION / FP16_FAST fusion env vars (tensor-vs-scalar
# fp16 compare trips `arith.cmpf same-type` -> uni_sram overflow compile fail).
config_inplace_ = CodeGenConfig(
    512,
    (65536, 65536, 65536),
    32,
    True,
    prefer_1d_tile=True,
    kunlunAutoGrid=True,
    unroll_num=8,
)


@pointwise_dynamic(promotion_methods=[(0, 1, "DEFAULT")], config=config_inplace_)
@triton.jit
def lt_func_(x, y):
    # XPU codegen penalty: a direct tensor-tensor fp compare (`x < y`) followed
    # by ANY use of the i1 result (cast/select) lowers to a per-lane slow path
    # (~10.5ms for 268M fp16 vs ~0.92ms for a plain `x - y`); a scalar-compare
    # with fp constants is fast. Emulate `x < y` with fast fp arithmetic only:
    #   t = (y - x) * M      -> +inf when x < y, -inf/0 when x >= y
    #   t = max(0, t)        -> 0 for x >= y, keeps +inf for x < y
    #   t = min(1, t)        -> 1 for x < y, 0 for x >= y
    # M = 1e30 saturates every representable x<y gap of fp16 (min gap 6e-8)
    # to +inf. max/min on this backend prefer the non-NaN operand, so NaN
    # inputs (y - x = NaN) resolve to 0, matching torch lt semantics; +-0 and
    # +-inf pairs also match IEEE. The slow i1-path is never materialized.
    t = (y - x) * 1.0e30
    t = tl.maximum(0.0, t)
    t = tl.minimum(1.0, t)
    return t


def lt_(A, B):
    logger.debug("GEMS_KUNLUNXIN LT_")
    if A.device != B.device:
        B = B.to(A.device)
    lt_func_(A, B, out0=A)
    return A


@pointwise_dynamic(
    is_tensor=[True, False],
    promotion_methods=[(0, 1, "DEFAULT")],
    config=config_inplace_,
)
@triton.jit
def lt_func_scalar_(x, y):
    t = (y - x) * 1.0e30
    t = tl.maximum(0.0, t)
    t = tl.minimum(1.0, t)
    return t


def lt_scalar_(A, B):
    logger.debug("GEMS_KUNLUNXIN LT_SCALAR_")
    numel = A.numel()
    if (
        A.is_contiguous()
        and A.dtype in (torch.float16, torch.float32)
        and numel >= _LT_SCALAR_INPLACE_FAST_TILE
        and numel % _LT_SCALAR_INPLACE_FAST_TILE == 0
        and numel // _LT_SCALAR_INPLACE_FAST_TILE >= _LT_SCALAR_INPLACE_MIN_GRID
        and float(B) == 0.0
    ):
        return _lt_scalar_inplace_fast(A)
    lt_func_scalar_(A, B, out0=A)
    return A


# ---------------------------------------------------------------------------
# lt_scalar_ fast path (fp16/fp32 only): same unmasked flat-tile idea as the
# out-of-place lt_scalar fast path above, but writing back in place. The
# generic in-place codegen path always emits `mask = tid < num_tasks` and the
# XPU backend lowers even an always-true runtime mask through the slow
# masked-memory path. An unmasked kernel with a fixed pow2 TILE (grid =
# numel / TILE, only applied when numel is exactly divisible) restores the
# fast DMA path. Swept on XPU 3 (2026-08-13) for [4096,4096]/[10000,65536]:
#   fp16 1.77 -> 1.51 ms (10000x65536) and ~0.057 -> ~0.046 ms (4096^2);
#   fp32 2.93 -> 2.88 ms; bf16 gets *slower* (1.94 -> 4.90 ms), so bf16
#   stays on the generic saturating-arithmetic path.
# As in the out-of-place path, only scalar == 0.0 is eligible: torch rounds a
# non-zero scalar to the input dtype first, so a raw fp32 scalar compare
# would diverge for non-zero values. Semantics are bit-identical to the
# generic in-place body for scalar 0 (same `(0 - x) * 1e30` saturate trick,
# and 1.0/0.0 are exactly representable in fp16/fp32). The kernel writes into
# the SAME tensor it reads, so it must keep the DEFAULT isCloseMemoryAsync
# (True = async copy closed); passing False with in-place aliasing is the
# documented "noc idle timeout" deadlock, same as config_inplace_ above.
_LT_SCALAR_INPLACE_FAST_TILE = 131072
_LT_SCALAR_INPLACE_MIN_GRID = 128


@triton.jit
def lt_scalar_inplace_fast_kernel(x_ptr, TILE: tl.constexpr):
    pid = tl.program_id(0)
    tid = pid * TILE + tl.arange(0, TILE)
    x = tl.load(x_ptr + tid)
    t = (0.0 - x) * 1.0e30
    t = tl.maximum(0.0, t)
    t = tl.minimum(1.0, t)
    tl.store(x_ptr + tid, t)


def _lt_scalar_inplace_fast(A):
    grid = (A.numel() // _LT_SCALAR_INPLACE_FAST_TILE,)
    lt_scalar_inplace_fast_kernel[grid](
        A,
        TILE=_LT_SCALAR_INPLACE_FAST_TILE,
        num_warps=4,
        buffer_size_limit=8192,
        unroll_num=16,
        isCloseMemoryAsync=True,
    )
    return A


# ---------------------------------------------------------------------------
# less_ / less_scalar_ are the PyTorch ALIAS names of lt_ / lt_scalar_
# (torch.ops.aten.less_.Tensor == torch.ops.aten.lt_.Tensor). They were NOT
# overridden by kunlunxin, so `less_` / `less_scalar_` stayed bound to the
# generic flag_gems/ops/less_.py -- a bare @pointwise_dynamic with NO
# CodeGenConfig -> discrete/launch-bound slow path on XPU (measured baseline
# [4096,4096] fp16: 49.1 ms gem vs 0.056 ms torch; [64,64,65536] fp16:
# 809 ms vs 0.80 ms; [10000,65536] less_scalar_ fp16: 1692 ms vs 1.21 ms;
# ~1000x regression, while the same-op lt_/lt_scalar_ are already tuned).
#
# Fix: export less_ / less_scalar_ from the kunlunxin ops module so
# SpecOpRegistrar shadows the generic names (same mechanism as le_/gt_/...).
# They simply delegate to the tuned in-place lt_ / lt_scalar_ recipes
# (config_inplace_ + unmasked flat-tile fast paths); kernel body/algorithm
# unchanged (zero correctness risk, aliases of the exact same ATen op).
def less_(A, B):
    return lt_(A, B)


def less_scalar_(A, B):
    return lt_scalar_(A, B)
