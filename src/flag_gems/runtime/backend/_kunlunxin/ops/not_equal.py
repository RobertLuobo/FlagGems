import logging
import math
import os

import torch
import triton
import triton.language as tl
from _kunlunxin.utils.codegen_config_utils import CodeGenConfig

from ..utils.pointwise_dynamic import pointwise_dynamic

logger = logging.getLogger(__name__)


# NOTE: `not_equal` / `not_equal_scalar` are aliases of `ne` / `ne_scalar`.
# kunlunxin overrides `ne` with the tuned config below but previously left
# `not_equal` UNCOVERED, so it fell to the generic `ops/not_equal.py` bare
# `@pointwise_dynamic` (no CodeGenConfig, no kunlunAutoGrid / unroll_num) and was
# stuck at the launch-bound / narrow-DMA baseline (IR
# `harness/perf_ir_3/ir-not_equal-dev6.log`). Mirroring the sibling `ne` recipe
# verbatim (tuned config + kunlunAutoGrid + unroll_num) lifts throughput with
# zero algorithm change.
#
# `buffer_size_limit=4096` bounds the per-core DMA tile (same lever proven on
# acos/isfinite). On the large benchmark shapes (268M / 65536-wide) it shaves a
# consistent ~4% off fp16/bf16 and ~10% off fp32 gems latency (fp32 268M
# 1.853->1.661ms, fp32 65536-wide 4.474->4.006ms) with no change on small
# shapes; the default launch path used buffer_size_limit=2048.
config_ = CodeGenConfig(
    512,
    (65536, 65536, 65536),
    32,
    True,
    prefer_1d_tile=True,
    isCloseMemoryAsync=False,
    kunlunAutoGrid=True,
    unroll_num=8,
    buffer_size_limit=4096,
)


@pointwise_dynamic(
    promotion_methods=[(0, 1, "ALWAYS_BOOL")],
    config=config_,
)
@triton.jit
def not_equal_func(x, y):
    return x.to(tl.float32) != y.to(tl.float32)


def not_equal(A, B):
    logger.debug("GEMS_KUNLUNXIN NOT_EQUAL")
    # Fast path (numel <= 65536, same-dtype/contiguous/same-shape floats): the
    # generic pointwise path below is a 1-CTA monolith (tile = next_pow2(numel),
    # sum(shape) <= 131072 forces num_ctas=1 in the generated wrapper), which is
    # launch/issue-bound at ~9-12us on small shapes. A many-small-tile flat
    # fused compare+bool-store kernel (same `x != y` body, no i1 ever stored)
    # measures 5.9-8.9us there -- see harness/solution/performance/not_equal_perf.md.
    # Tile buckets sweep-measured on XPU 5 (2026-09-04, harness do_bench
    # warmup=1000/rep=100 median):
    #   numel <= 16384   -> 2048-lane  (5.95-6.20us vs 1-CTA 9.6-12.0us; e.g.
    #                       (1024,16) 6.02 vs 11.05, torch 5.69 -> 0.95x)
    #   16384 < n <= 65536 -> 8192-lane (8.83-8.90us vs 1-CTA 8.65-10.25us)
    # The masked variant is used only for non-multiples (e.g. (1024,1): one
    # real-tail block); exact multiples run the unmasked kernel (the masked
    # memory channel costs ~2x, and every benchmark/tile size here divides its
    # bucket).
    numel = A.numel()
    if (
        A.dtype in (torch.float16, torch.float32, torch.bfloat16)
        and A.dtype == B.dtype
        and A.is_contiguous()
        and B.is_contiguous()
        and A.shape == B.shape
        and 0 < numel <= _NOT_EQUAL_TENSOR_FAST_MAX
    ):
        if numel <= _NOT_EQUAL_TENSOR_SMALL_MAX:
            return _not_equal_tensor_fast(A, B, numel, _NOT_EQUAL_TENSOR_TILE_SMALL)
        return _not_equal_tensor_fast(A, B, numel, _NOT_EQUAL_TENSOR_TILE_MID)
    os.environ["TRITONXPU_COMPARE_FUSION"] = "1"
    os.environ["TRITONXPU_FP16_FAST"] = "1"
    res = not_equal_func(A, B)
    del os.environ["TRITONXPU_COMPARE_FUSION"]
    del os.environ["TRITONXPU_FP16_FAST"]
    return res


# ---------------------------------------------------------------------------
# not_equal tensor-tensor fast paths (fp16/fp32/bf16, both contiguous, same
# shape, numel <= 65536). Same mechanism as the sibling less_equal/ne small
# path: a fused `cmpf != 0` + bool store under TRITONXPU_COMPARE_FUSION /
# TRITONXPU_FP16_FAST (set by `not_equal` for every entry, so the first
# compile of either kernel sees them), no i1 ever materialized outside the
# vendor's fused compare store. `x.to(tl.float32) != y.to(tl.float32)` is
# bit-identical to the generic path (fp16/bf16 -> fp32 is exact), so NaN
# (ne(NaN, y) == True), +-0 (+-0 != +-0 is False), equal +-inf and every
# subnormal/normal gap behave exactly like torch.
#
# Measured crossover (same-matrix A/B, 2026-09-04): on (1024,256)=262144 the
# 1-CTA monolith (12.5-15.4us) is already below every flat multi-CTA bucket
# (T=8192: 16.97us), so the fast path stops at 65536; the twostage
# saturating+_copy_from recipe is 12.3-16.5us everywhere in this range and
# never wins for tensor-tensor (the fused 5B/elem path beats its 13B/elem).
_NOT_EQUAL_TENSOR_TILE_SMALL = 2048
_NOT_EQUAL_TENSOR_SMALL_MAX = 16384
_NOT_EQUAL_TENSOR_TILE_MID = 8192
_NOT_EQUAL_TENSOR_FAST_MAX = 65536


@triton.jit
def not_equal_tensor_fast_kernel(x_ptr, y_ptr, out_ptr, n_elements, TILE: tl.constexpr):
    pid = tl.program_id(0)
    offset = pid * TILE + tl.arange(0, TILE)
    mask = offset < n_elements
    x = tl.load(x_ptr + offset, mask=mask, other=0).to(tl.float32)
    y = tl.load(y_ptr + offset, mask=mask, other=0).to(tl.float32)
    tl.store(out_ptr + offset, x != y, mask=mask)


@triton.jit
def not_equal_tensor_fast_unmasked_kernel(x_ptr, y_ptr, out_ptr, TILE: tl.constexpr):
    pid = tl.program_id(0)
    offset = pid * TILE + tl.arange(0, TILE)
    x = tl.load(x_ptr + offset).to(tl.float32)
    y = tl.load(y_ptr + offset).to(tl.float32)
    tl.store(out_ptr + offset, x != y)


def _not_equal_tensor_fast(A, B, numel, TILE):
    os.environ["TRITONXPU_COMPARE_FUSION"] = "1"
    os.environ["TRITONXPU_FP16_FAST"] = "1"
    out = torch.empty_like(A, dtype=torch.bool)
    try:
        if numel % TILE == 0:
            not_equal_tensor_fast_unmasked_kernel[(numel // TILE,)](
                A,
                B,
                out,
                TILE=TILE,
                num_warps=4,
                buffer_size_limit=8192,
                unroll_num=16,
                isCloseMemoryAsync=False,
            )
        else:
            # non-multiple of the bucket (e.g. (1024,1) with TILE=2048): a
            # single block with a real tail mask. The mask covers genuine
            # elements only.
            not_equal_tensor_fast_kernel[(triton.cdiv(numel, TILE),)](
                A,
                B,
                out,
                numel,
                TILE=TILE,
                num_warps=4,
                buffer_size_limit=8192,
                unroll_num=16,
                isCloseMemoryAsync=False,
            )
        return out
    finally:
        del os.environ["TRITONXPU_COMPARE_FUSION"]
        del os.environ["TRITONXPU_FP16_FAST"]


@pointwise_dynamic(
    is_tensor=[True, False],
    promotion_methods=[(0, 1, "ALWAYS_BOOL")],
    config=config_,
)
@triton.jit
def not_equal_func_scalar(x, y):
    return x.to(tl.float32) != y


def not_equal_scalar(A, B):
    logger.debug("GEMS_KUNLUNXIN NOT_EQUAL_SCALAR")
    # not_equal.Scalar is an exact alias of ne.Scalar (`torch.not_equal` ==
    # `torch.ne`; same ATen semantics: a != b element-wise, NaN-aware). The
    # generic scalar-compare path (not_equal_func_scalar) materializes
    # `arith.cmpf -> i1 -> bool store` per lane, which the XPU backend lowers
    # to the same i1 slow path that doomed the closed ne_scalar (baseline
    # 2026-08-14, XPU 7: 17.6ms vs 1.09ms on [10000,65536]). Take the closed
    # ne_scalar fast path below (same two-stage saturating recipe,
    # `harness/solution/performance/not_equal_scalar_perf.md`) whenever
    # applicable; generic path otherwise, behavior unchanged.
    numel = A.numel()
    dtype = A.dtype
    if (
        A.is_contiguous()
        and dtype in (torch.float16, torch.float32, torch.bfloat16)
        and numel >= _NOT_EQUAL_SCALAR_MASKED_MIN
    ):
        # Only build the wrapped scalar (a `torch.tensor(...).item()`
        # roundtrip, ~3us host) for candidate sizes; the small-shape generic
        # path below must stay free of host overhead (measured 0.49-0.82x
        # regression on (64,64)/(10000,1)/(100,1,100) when it ran every call).
        s = float(B)
        wrapped = torch.tensor(s, dtype=dtype).item()
        if math.isfinite(wrapped):
            if (
                numel % _NOT_EQUAL_SCALAR_FAST_TILE == 0
                and numel >= _NOT_EQUAL_SCALAR_FAST_TILE * _NOT_EQUAL_SCALAR_MIN_GRID
            ):
                # exact-multiple flat tiles (grid >= MIN_GRID): no mask, no
                # i1 -- a saturating fp32 store + vendor bool conversion.
                return _not_equal_scalar_fast(
                    A, float(wrapped), (numel // _NOT_EQUAL_SCALAR_FAST_TILE,)
                )
            if numel % _NOT_EQUAL_SCALAR_FAST_TILE != 0:
                # non-multiple mid sizes (e.g. 2.56M, [10000,256]): flat
                # tiles with a real tail mask. The mask is genuine (tail
                # elements), so the masked-memory path is the only penalty.
                return _not_equal_scalar_fast_masked(A, float(wrapped), numel)
    # Like ne_scalar / gt_scalar, the scalar path must NOT set
    # TRITONXPU_COMPARE_FUSION / TRITONXPU_FP16_FAST: for tensor-vs-scalar the
    # fusion env vars make the compiler emit an fp16 compare that trips
    # `arith.cmpf same-type` and overflows uni_sram -> compile failure.
    res = not_equal_func_scalar(A, B)
    return res


# ---------------------------------------------------------------------------
# not_equal_scalar fast paths (fp16/fp32/bf16, contiguous, finite wrapped
# scalar). Exact copy of the closed ne_scalar recipe (identical ATen op).
#
# Why: like the rest of the scalar-compare family, the generic scalar path
# always materializes `arith.cmpf -> i1 -> bool store` per lane, which the
# XPU backend lowers to a per-lane slow path (~10-20x). The saturating fp32
# store + vendor `_copy_from` fp32->bool conversion never materializes i1.
#
# not_equal(x, s) is exactly the logical complement of eq(x, s) but with
# OPPOSITE NaN semantics (ne(NaN, s) == True while eq(NaN, s) == False). The
# eq formula's saturating distance stores directly (no negation):
#   t = min(1, |x - s| * 1e30 * 1e15)  -> 0.0 when x == s, 1.0 otherwise
# SCALE = 1e30 * 1e15: every representable fp16/bf16/fp32 gap saturates t to
# exactly 1.0 while a zero difference stays exactly 0.0; +-0 != +-0 -> False.
# NaN input -> t = min(1, NaN): fp16/fp32 min prefers the non-NaN operand
# (1.0), bf16 yields NaN; both convert to bool True via the vendor conversion
# -- exactly ne(NaN, s) == True (the eq path wraps the NaN away with
# max(0, .), ne must NOT).
#
# The tensored scalar passed to the kernel is float(wrapped) -- the scalar
# rounded to the input dtype -- bit-identical to torch's wrapped-scalar
# comparison. The +/-inf corner (x = s = +/-inf -> False) needs a wrapped
# scalar of +/-inf: rejected above by math.isfinite, keeping the exact
# generic compare path. NaN scalars also stay generic.
#
# Stage two: fp32 -> bool via `torch.ops.aten._copy_from` (NOT registered by
# gems, so it always reaches the vendor's native conversion kernel).
_NOT_EQUAL_SCALAR_FAST_TILE = 131072
_NOT_EQUAL_SCALAR_MIN_GRID = 128
_NOT_EQUAL_SCALAR_MASKED_MIN = 1 << 20


@triton.jit
def not_equal_scalar_fast_kernel(out_ptr, x_ptr, scalar, TILE: tl.constexpr):
    pid = tl.program_id(0)
    tid = pid * TILE + tl.arange(0, TILE)
    x = tl.load(x_ptr + tid).to(tl.float32)
    d = tl.abs(x - scalar)
    t = tl.minimum(1.0, d * 1.0e30 * 1.0e15)
    tl.store(out_ptr + tid, t)


def _not_equal_scalar_fast(A, scalar, grid):
    out32 = torch.empty_like(A, dtype=torch.float32)
    not_equal_scalar_fast_kernel[grid](
        out32,
        A,
        scalar,
        TILE=_NOT_EQUAL_SCALAR_FAST_TILE,
        num_warps=4,
        buffer_size_limit=8192,
        unroll_num=16,
        isCloseMemoryAsync=False,
    )
    out_bool = torch.empty_like(A, dtype=torch.bool)
    torch.ops.aten._copy_from(out32, out_bool, False)
    return out_bool


@triton.jit
def not_equal_scalar_fast_masked_kernel(
    out_ptr, y_ptr, scalar, numel, TILE: tl.constexpr
):
    pid = tl.program_id(0)
    tid = pid * TILE + tl.arange(0, TILE)
    mask = tid < numel
    y = tl.load(y_ptr + tid, mask=mask).to(tl.float32)
    d = tl.abs(y - scalar)
    t = tl.minimum(1.0, d * 1.0e30 * 1.0e15)
    tl.store(out_ptr + tid, t, mask=mask)


def _not_equal_scalar_fast_masked(A, scalar, numel):
    out32 = torch.empty_like(A, dtype=torch.float32)
    grid = (math.ceil(numel / _NOT_EQUAL_SCALAR_FAST_TILE),)
    not_equal_scalar_fast_masked_kernel[grid](
        out32,
        A,
        scalar,
        numel,
        TILE=_NOT_EQUAL_SCALAR_FAST_TILE,
        num_warps=4,
        buffer_size_limit=8192,
        unroll_num=16,
        isCloseMemoryAsync=False,
    )
    out_bool = torch.empty_like(A, dtype=torch.bool)
    torch.ops.aten._copy_from(out32, out_bool, False)
    return out_bool