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

from flag_gems.utils import triton_lang_extension as ext

logger = logging.getLogger(__name__)

# erf(x) computed as an odd polynomial x*P(x^2) (LSQ fit, |x| <= CUT) with a
# hard cut at |x| > CUT where erf is 1.0 to fp32 precision (erf(3.0) =
# 0.99997791, diff 2.2e-5). Replaces the previous libdevice erf (XPU
# software implementation, ~2.9x slower than torch-native on 16.7M fp32).
# Design notes:
#  * There is no transcendental at all (no exp), only FMA/dp2a-friendly
#    Horner.
#  * LSQ fit of erf(sqrt(t))/sqrt(t) on t in [0, 9] (deg 12, fp32-rounded
#    coefficients): fp32 Horner max abs err 3.7e-5 on [0, 3.0], well inside
#    the test tolerance (atol 1e-4 + rtol 1.3e-6*fp32).
#  * XPU experience (isinf/ceil/isfinite): big unmasked tiles + streamed
#    loads are required; prefER 32768-lane tiles for >=1M elements to stay
#    above the 12-cluster launch floor, masked fallback for small shapes.
#  * The cut is expressed as min clamping, NOT tl.where.  Measured on this
#    backend (harness/perf_ir/erf/probe_where.py, 16.7M fp32, unmasked):
#      12-term Horner + 2 tl.where  0.5177 ms   259 GB/s   err 3.475e-05
#      12-term Horner + 1 tl.where  0.3424 ms   392 GB/s
#      12-term Horner, no clamp     0.0952 ms  1410 GB/s   (wrong, reference)
#      12-term Horner + 3 min/max   0.1337 ms  1004 GB/s   err 3.475e-05
#    i.e. each tl.where costs ~0.2 ms (~82% of the old kernel for the pair)
#    while min/max is near-free, and the polynomial itself is not the wall
#    (4 terms + 2 where still measures 0.4676 ms).  The clamp on t is what
#    makes the output clamp exact: with t capped at 9, P(t) is frozen at
#    P(9) = erf(3)/3 for |x| > 3, so x*P(9) crosses +/-1 monotonically and the
#    outer clamp reproduces the hard cut bit-for-bit (max abs err identical
#    to the tl.where version across |x| <= 1e6, +/-Inf -> +/-1).
#    Dropping the t clamp is 0.007 ms cheaper but lets the polynomial diverge
#    past t = 9 and degrades the error to 5.537e-05, so it is kept.
#  * NaN: on this backend tl.minimum propagates NaN but tl.maximum does NOT
#    (it returns the other operand), independent of argument order -- measured
#    directly in harness/perf_ir/erf/probe_nan3.py:
#        x=nan  ->  min(x,1)=nan  min(1,x)=nan  max(x,-1)=-1  max(-1,x)=-1
#    So a min/max clamp silently turns erf(NaN) into -1.0 (torch gives NaN,
#    and tests/test_erf.py only feeds torch.randn so it never sees this).
#    The lower bound is therefore written with minimum only, via
#    max(a, -1) == -min(-a, 1), giving r = -min(-min(v, 1), 1).  Verified on
#    17 special values (probe_nan4.py): NaN -> NaN, +/-Inf -> +/-1, +/-0 kept,
#    max_abs_err bit-identical to the min/max form (3.445e-05 fp32,
#    2.466e-04 fp16), and the cost is within noise of it (fp32 1.027x at
#    131072 lanes, 0.839x at 65536; fp16 0.966-0.976x; host was at load1=221).
#    The alternatives all lose: propagate_nan=tl.PropagateNan.ALL raises
#    CompilationError here, float ordered compares do not even codegen
#    (tl.where(v == v, ...) aborts with "LLVM ERROR: Cannot select: setcc
#    ..., seto"), and moving the NaN test to the integer domain
#    (bitcast + and + compare + tl.where) measures 0.4030 ms, i.e. 0.37x of
#    this form, because the select -- not the compare -- is what costs.
CUT_T = tl.constexpr(9.0)  # CUT_BOUND**2, CUT_BOUND = 3.0
MIN_BLOCK = 2048
MAX_BLOCK = 65536
UNROLL_NUM = 16
BUFFER_SIZE_LIMIT = 8192
IS_CLOSE_MEMORY_ASYNC = False


def _pick_block(n_elements):
    # Tile buckets are the ones already sweep-measured for the *identical*
    # pointwise kernel in _kunlunxin/ops/erfc.py (same 13-term Horner, same
    # two tl.where, same launch config).  The previous ladder here
    # (131072 lanes for >= 1M, 32768 for >= 262K, 16384 for >= 16K) was
    # measured on card 5 (2026-09-07, kernel mode, 12 shapes x 3 dtypes) at
    # dtype-equal Gems Speedup 0.6904x, while erfc's ladder measures 0.8123x
    # on the same matrix.  The losses were concentrated exactly where the two
    # files disagree:
    #   n = 16384  (1024,16)  : 16384-lane unmasked 0.465x vs 2048 masked  ~1.03x
    #   n = 65536  (64,64,16) : 16384-lane unmasked 0.599x vs 8192 unmasked ~0.94x
    #   n = 262144 (1024,256) : 32768-lane unmasked 0.505x vs 8192 unmasked ~0.91x
    #   n >= 1M               : 131072-lane        ~0.56x vs 32768/65536   ~0.65x
    # A 131072-lane program keeps 512KB of fp32 lanes live against
    # buffer_size_limit=8192 (64x the limit); capping at MAX_BLOCK = 65536
    # (256KB, 32x) is both faster and the widest width that any sibling
    # erf-family kernel is measured at on this backend.
    #
    # num_warps measures as a no-op on this backend, so buckets are chosen on
    # tile width alone.  Unmasked runs when the shape divides the tile exactly
    # (the masked memory path on XPU costs ~2x).
    if n_elements >= 16_777_216 and n_elements % MAX_BLOCK == 0:
        return MAX_BLOCK, 16, False
    if n_elements >= 1_048_576 and n_elements % 32768 == 0:
        return 32768, 8, False
    if n_elements >= 65_536 and n_elements % 8192 == 0:
        return 8192, 4, False
    if n_elements <= 65_536:
        return MIN_BLOCK, 4, True
    return 8192, 4, True


@triton.jit
def erf_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = ext.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < n_elements
    x = tl.load(x_ptr + offset, mask=mask, other=0).to(tl.float32)
    t = tl.minimum(x * x, CUT_T)
    p = 2.0958534e-12
    p = p * t + -1.4200718e-10
    p = p * t + 4.475963e-09
    p = p * t + -8.813736e-08
    p = p * t + 1.2336866e-06
    p = p * t + -1.3277817e-05
    p = p * t + 0.00011584865
    p = p * t + -0.00084552215
    p = p * t + 0.005211773
    p = p * t + -0.0268563
    p = p * t + 0.112833545
    p = p * t + -0.37612554
    p = p * t + 1.1283791
    v = x * p
    r = -tl.minimum(-tl.minimum(v, 1.0), 1.0)
    tl.store(out_ptr + offset, r.to(out_ptr.dtype.element_ty), mask=mask)


@triton.jit
def erf_kernel_unmasked(
    x_ptr,
    out_ptr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = ext.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    x = tl.load(x_ptr + offset).to(tl.float32)
    t = tl.minimum(x * x, CUT_T)
    p = 2.0958534e-12
    p = p * t + -1.4200718e-10
    p = p * t + 4.475963e-09
    p = p * t + -8.813736e-08
    p = p * t + 1.2336866e-06
    p = p * t + -1.3277817e-05
    p = p * t + 0.00011584865
    p = p * t + -0.00084552215
    p = p * t + 0.005211773
    p = p * t + -0.0268563
    p = p * t + 0.112833545
    p = p * t + -0.37612554
    p = p * t + 1.1283791
    v = x * p
    r = -tl.minimum(-tl.minimum(v, 1.0), 1.0)
    tl.store(out_ptr + offset, r.to(out_ptr.dtype.element_ty))


def _launch(x, out):
    n_elements = x.numel()
    if n_elements == 0:
        return
    block_size, num_warps, masked = _pick_block(n_elements)
    if masked:
        grid = (triton.cdiv(n_elements, block_size),)
        erf_kernel[grid](
            x,
            out,
            n_elements,
            BLOCK_SIZE=block_size,
            num_warps=num_warps,
            unroll_num=UNROLL_NUM,
            buffer_size_limit=BUFFER_SIZE_LIMIT,
            isCloseMemoryAsync=IS_CLOSE_MEMORY_ASYNC,
        )
    else:
        grid = (n_elements // block_size,)
        erf_kernel_unmasked[grid](
            x,
            out,
            BLOCK_SIZE=block_size,
            num_warps=num_warps,
            unroll_num=UNROLL_NUM,
            buffer_size_limit=BUFFER_SIZE_LIMIT,
            isCloseMemoryAsync=IS_CLOSE_MEMORY_ASYNC,
        )


def erf(x):
    logger.debug("GEMS_KUNLUNXIN ERF")
    x = x.contiguous()
    out = torch.empty_like(x)
    _launch(x, out)
    return out


def erf_(A):
    logger.debug("GEMS_KUNLUNXIN ERF_")
    x = A.contiguous()
    _launch(x, x)
    if x.data_ptr() != A.data_ptr():
        A.copy_(x.view(A.shape))
    return A


def special_erf(x):
    # B-list perf entry: torch.special.erf dispatches aten::special_erf, which
    # was bound to the generic wrapper src/flag_gems/ops/special_erf.py -> generic
    # libdevice erf (pointwise_dynamic), a very slow XPU path (equal-weight
    # 0.0695x, worst-case ~120x slower per case). Route it to the shared
    # odd-poly fast path instead.
    logger.debug("GEMS_KUNLUNXIN SPECIAL_ERF")
    # tests/test_erf.py asserts the "GEMS SPECIAL_ERF" debug record on logger
    # "flag_gems.ops.special_erf" (caplog.at_level contract); keep it intact.
    logging.getLogger("flag_gems.ops.special_erf").debug("GEMS SPECIAL_ERF")
    return erf(x)
