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
import math

import torch
import triton
import triton.language as tl
from _kunlunxin.utils.codegen_config_utils import CodeGenConfig

from flag_gems.utils import tl_extra_shim

from ..utils.pointwise_dynamic import pointwise_dynamic

logger = logging.getLogger(__name__)

atan2 = tl_extra_shim.atan2

# XPU-tuned CodeGenConfig, one SEPARATE instance per signature (a CodeGenConfig is
# bound to the signature it was first compiled for; sharing one instance across
# signatures is the known `addcmul` cross-signature side effect).
#
# Without an explicit config both pointwise kernels fall back to the platform
# default (`CodeGenConfig(512,(65536,)*3,32,True,prefer_1d_tile=True)` with
# kunlunAutoGrid=False / buffer_size_limit=0), which is measurably worse here:
#   * kunlunAutoGrid=True picks num_ctas=1 for small tasks instead of the fixed
#     12-CTA launch (removes cluster overhead at tiny shapes);
#   * buffer_size_limit=4096 lifts the per-core DMA staging buffer for the large
#     shapes.
config_complex_ = CodeGenConfig(
    512,
    (65536, 65536, 65536),
    32,
    True,
    prefer_1d_tile=True,
    buffer_size_limit=4096,
    kunlunAutoGrid=True,
)

config_float_int_ = CodeGenConfig(
    512,
    (65536, 65536, 65536),
    32,
    True,
    prefer_1d_tile=True,
    buffer_size_limit=4096,
    kunlunAutoGrid=True,
)

# Third instance for the packed-complex64 kernel: same (one tensor, INT_TO_FLOAT)
# signature shape but a different operand dtype (int64), and a CodeGenConfig is
# bound to the signature it was first compiled for.
config_complex_packed_ = CodeGenConfig(
    512,
    (65536, 65536, 65536),
    32,
    True,
    prefer_1d_tile=True,
    buffer_size_limit=4096,
    kunlunAutoGrid=True,
)


@pointwise_dynamic(
    is_tensor=[True, True],
    promotion_methods=[(0, "DEFAULT")],
    config=config_complex_,
)
@triton.jit
def angle_func(real, imag):
    real_last, imag_last = (
        (real.to(tl.float32), imag.to(tl.float32))
        if real.dtype == tl.float16
        else (real, imag)
    )
    result = atan2(imag_last, real_last)
    return result


@pointwise_dynamic(
    is_tensor=[True],
    promotion_methods=[(0, "INT_TO_FLOAT")],
    config=config_float_int_,
)
@triton.jit
def angle_float_and_int(real):
    # angle(x) == pi if x < 0 else 0.
    #
    # Upstream body is `tl.where(real >= 0.0, 0.0, math.pi)`.  On this backend the
    # i1 select itself is not the problem -- it is the *widening f32 conversion*
    # that the float-typed compare drags in, which collapses the vectorized
    # load/store datapath for >=32-bit element loads.  Measured on 16M elements
    # (official caliber, median of 7 in-process samples):
    #
    #   dtype   select+float-cmp   select+int-cmp   float clamp (below)
    #   bool        118 us             47 us              n/a
    #   int16       294 us            352 us            2.9 ms
    #   int32       270 us             n/a               71 us
    #   f32         270 us             n/a               70 us
    #
    # So the body is chosen per element width.
    #
    # >=32-bit floats and ints: a pure-float arithmetic clamp keeps the vectorized
    # datapath:
    #
    #     pi * clamp(-real * BIG, 0, 1)     with BIG > 1/FLT_MIN
    #
    # |real| >= FLT_MIN makes -real*BIG >= 1 for every negative real and <= 0 for
    # every non-negative real, so the clamp reproduces `real >= 0` exactly for all
    # finite normals; the `max(..., 0)` also sends -0.0 (and NaN) to 0, matching
    # `real >= 0` on -0.0.  Over +-0, +-inf, +-FLT_MIN, +-FLT_MAX, +-1e30,
    # +-1e-8 and 16M randn it is bit-identical to the `tl.where` body, and it
    # equals that body for every finite value with |real| >= 1/BIG = 1.1628e-38
    # (all finite normals, plus the top sliver of subnormals).  The deviation
    # domain is exactly two sets: (a) negative real with |real| < 1/BIG -- all
    # but that sliver of the negative subnormals -- where -real*BIG < 1 makes the
    # clamp return ~0, not pi; and (b) NaN, which the maxnum `max(..., 0)` folds
    # to 0, not pi.
    if real.dtype == tl.int1:
        # bool: comparing in the i1 domain keeps the load 1-byte (no widening), the
        # fastest body by 2.5x.
        return tl.where(real >= 0, 0.0, math.pi)
    if real.dtype == tl.int8 or real.dtype == tl.int16:
        # Sub-32-bit integers must keep the upstream float-typed compare: i16->f32
        # in this two-operand form vectorizes, whereas the i1-domain compare is 20%
        # slower and an explicit `.to(tl.float32)` does not even compile ("size
        # mismatch when packing elements for LLVM struct expected 4 but got 2" in
        # ConvertTritonXPUToLLVM); routing through int32 compiles but scalarizes
        # the load (16M int16: 2.9 ms).
        zero = 0.0
        pi = math.pi
        real_positive = real >= zero
        return tl.where(real_positive, zero, pi)
    real_last = real.to(tl.float32)
    pi = math.pi
    big = 8.6e37
    return pi * tl.minimum(tl.maximum(real_last * (0.0 - big), 0.0), 1.0)


@pointwise_dynamic(
    is_tensor=[True],
    promotion_methods=[(0, "INT_TO_FLOAT")],
    config=config_complex_packed_,
)
@triton.jit
def angle_complex64_packed(packed):
    # `packed` is one int64 per complex64 element: the raw little-endian 8-byte
    # storage of the (real, imag) pair, loaded as a single *contiguous* stream.
    #
    # `input.real` / `input.imag` are stride-2 plane views, and this backend's
    # pointwise path cannot vectorize those: measured 16M complex64, the same
    # atan2 kernel costs 7.3ms on the stride-2 views vs 5.6ms on contiguous
    # planes.  Materializing the planes is not an option under `use_gems` (the
    # vendor copy_ is 147us -> 26.9ms for a stride-2 source, and even the fast
    # `tle_copy` costs 1.94ms/plane), so the packing is done for free by reading
    # the interleaved storage as one contiguous int64 and splitting the halves in
    # registers.  Bit arithmetic, so the result is bit-identical to the plane
    # kernel (verified with torch.equal over 16M randn).
    #   * `.to(tl.int32)` truncates -> exactly the low 32 bits (real)
    #   * `>> 32` then truncate    -> exactly the high 32 bits (imag)
    real = packed.to(tl.int32).to(tl.float32, bitcast=True)
    imag = (packed >> 32).to(tl.int32).to(tl.float32, bitcast=True)
    return atan2(imag, real)


def angle(input_tensor: torch.Tensor) -> torch.Tensor:
    logger.debug("GEMS_KUNLUNXIN ANGLE")
    if input_tensor.dtype == torch.complex32 or input_tensor.dtype == torch.complex64:
        # complex64 has an exact 8-byte integer supertype, so the interleaved
        # storage can be read as one contiguous int64 element per complex value
        # (see angle_complex64_packed).  complex32 has no 4-byte integer dtype on
        # this backend, and a non-contiguous complex tensor cannot be viewed, so
        # both fall back to the stride-2 plane kernel below.
        if input_tensor.dtype == torch.complex64 and input_tensor.is_contiguous():
            packed = input_tensor.view(torch.int64)
            return angle_complex64_packed(
                packed,
                out0=torch.empty(
                    packed.shape, dtype=torch.float32, device=packed.device
                ),
            )
        real = input_tensor.real
        imag = input_tensor.imag
        # Host-side pre-allocation: passing `out0` makes pointwise_dynamic skip the
        # per-call dtype-promotion walk + `empty_like` inside prepare_args, which is
        # the dominant cost at launch-bound shapes. Valid only while promotion
        # provably cannot change the shape/dtype: two same-dtype float operands of
        # equal shape promote (DEFAULT) to exactly that dtype.
        if (
            real.dtype == imag.dtype
            and real.dtype in (torch.float32, torch.float16)
            and real.shape == imag.shape
        ):
            return angle_func(
                real,
                imag,
                out0=torch.empty(real.shape, dtype=real.dtype, device=real.device),
            )
        return angle_func(real, imag)
    else:
        real = input_tensor
        # Same host-side trick. The promoted result dtype of an INT_TO_FLOAT
        # signature is the input dtype for floating inputs and float32 for every
        # integer/bool input, so the guard below is exact rather than heuristic.
        if real.is_floating_point():
            out0 = torch.empty(real.shape, dtype=real.dtype, device=real.device)
        else:
            out0 = torch.empty(real.shape, dtype=torch.float32, device=real.device)
        return angle_float_and_int(real, out0=out0)
