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

import triton
import triton.language as tl

from flag_gems.ops import _upsample_lanczos2d_aa_backward as _generic

logger = logging.getLogger(__name__)


@triton.jit
def _lanczos_aa_filter(x):
    pix = math.pi * x
    sinc = tl.where(x == 0.0, 1.0, tl.sin(pix) / pix)
    sinc_three = tl.where(x == 0.0, 1.0, tl.sin(pix / 3.0) / (pix / 3.0))
    return tl.where(x < 3.0, sinc * sinc_three, 0.0)


# The generic Lanczos filter calls tl_extra_shim.sinpi, which lowers to an
# undefined extern symbol ("Unsupported") on the XPU3 linker. All the generic
# weight kernels resolve _lanczos_aa_filter from the generic module globals at
# JIT-compile time, so replacing that attribute with a tl.sin-based equivalent
# fixes every kernel while reusing the entire generic driver and scaffolding.
_generic._lanczos_aa_filter = _lanczos_aa_filter


# The generic fused single-pass kernel unrolls a MAX_OW x MAX_OH nested
# tl.static_range (often >100 iterations for the CI shapes). On XPU3 that
# exceeds the local-memory budget and aborts compilation with
# "Failed to tune buffer size." The separable two-pass path produces identical
# results and compiles cleanly, so force it on this backend.
def _should_use_fused_path(*args, **kwargs):
    return False


_generic._should_use_fused_path = _should_use_fused_path


def upsample_lanczos2d_aa_backward(
    grad_output,
    output_size,
    input_size,
    align_corners,
    scales_h=None,
    scales_w=None,
):
    logger.debug("GEMS_KUNLUNXIN UPSAMPLE_LANCZOS2D_AA_BACKWARD")
    logger.debug("GEMS UPSAMPLE_LANCZOS2D_AA_BACKWARD")
    return _generic.upsample_lanczos2d_aa_backward(
        grad_output,
        output_size,
        input_size,
        align_corners,
        scales_h,
        scales_w,
    )


def upsample_lanczos2d_aa_backward_grad_input(
    grad_output,
    output_size,
    input_size,
    align_corners,
    scales_h=None,
    scales_w=None,
    *,
    grad_input,
):
    logger.debug("GEMS_KUNLUNXIN UPSAMPLE_LANCZOS2D_AA_BACKWARD_GRAD_INPUT")
    logger.debug("GEMS UPSAMPLE_LANCZOS2D_AA_BACKWARD_GRAD_INPUT")
    return _generic.upsample_lanczos2d_aa_backward_grad_input(
        grad_output,
        output_size,
        input_size,
        align_corners,
        scales_h,
        scales_w,
        grad_input=grad_input,
    )
