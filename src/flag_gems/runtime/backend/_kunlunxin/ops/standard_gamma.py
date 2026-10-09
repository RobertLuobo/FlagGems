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

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils.random_utils import (
    philox_backend_seed_offset,
    uint_to_uniform_float,
)
from flag_gems.utils.shape_utils import volume

logger = logging.getLogger(__name__)

MAX_ITERS = 32
PHILOX_STRIDE = (MAX_ITERS + 1) * 4
BLOCK_SIZE = 1024


@triton.jit
def _boost_uniform(seed, base_counter, alpha_f32, MAX_ITERS: tl.constexpr):
    counter = base_counter + MAX_ITERS * 4
    c0 = (counter & 0xFFFFFFFF).to(tl.uint32)
    c1 = ((counter >> 32) & 0xFFFFFFFF).to(tl.uint32)
    z = c0 * 0
    r0, _, _, _ = tl.philox(seed, c0, c1, z, z)
    u = tl.maximum(uint_to_uniform_float(r0), 1e-7)
    return tl.where(alpha_f32 < 1.0, tl.exp(tl.log(u) / alpha_f32), 1.0)


@libentry()
@triton.jit(do_not_specialize=["philox_seed", "philox_offset"])
def standard_gamma_kernel(
    input_ptr,
    output_ptr,
    n_elements,
    philox_seed,
    philox_offset,
    TINY,
    BLOCK_SIZE: tl.constexpr,
    STRIDE: tl.constexpr,
    MAX_ITERS: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    alpha = tl.load(input_ptr + offsets, mask=mask, other=1.0)
    alpha_f32 = alpha.to(tl.float32)

    alpha_adj = tl.where(alpha_f32 < 1.0, alpha_f32 + 1.0, alpha_f32)
    d = alpha_adj - 0.3333333333333333
    c = 1.0 / tl.sqrt(9.0 * d)

    philox_seed = philox_seed.to(tl.int64)
    philox_offset = philox_offset.to(tl.int64)
    base_counter = philox_offset + offsets.to(tl.int64) * STRIDE

    boost = _boost_uniform(philox_seed, base_counter, alpha_f32, MAX_ITERS)

    # KUNLUNXIN override: the generic kernel tracked acceptance with a tl.int1
    # tensor and tested `accepted == 0`. On the XPU3 (KL3) triton backend that
    # i1 comparison lowers to an `arith.extui` whose tensor shape is dropped,
    # failing TritonXPU pass verification ("'arith.extui' op failed to verify
    # that input and output have the same tensor dimensions"). Track acceptance
    # in an int32 tensor and derive the "newly accepted" mask with integer
    # arithmetic so no i1->iN extension is emitted.
    accepted = tl.zeros([BLOCK_SIZE], dtype=tl.int32)
    result_f32 = tl.zeros([BLOCK_SIZE], dtype=tl.float32)

    for it in range(0, MAX_ITERS):
        counter = base_counter + it * 4
        c0 = (counter & 0xFFFFFFFF).to(tl.uint32)
        c1 = ((counter >> 32) & 0xFFFFFFFF).to(tl.uint32)
        z0 = c0 * 0
        r0, r1, r2, r3 = tl.philox(philox_seed, c0, c1, z0, z0)
        u2 = tl.maximum(uint_to_uniform_float(r1), 1e-7)
        u3 = uint_to_uniform_float(r2)
        u4 = tl.maximum(uint_to_uniform_float(r3), 1e-7)

        z = tl.sqrt(-2.0 * tl.log(u2)) * tl.cos(2.0 * 3.141592653589793 * u3)
        v = 1.0 + c * z
        v3 = v * v * v

        accept_i = tl.where(
            (v > 0.0) & (tl.log(u4) < (0.5 * z * z + d - d * v3 + d * tl.log(v3))),
            1,
            0,
        ).to(tl.int32)
        take_i = accept_i * (1 - accepted)
        result_f32 = tl.where(take_i != 0, d * v3, result_f32)
        accepted = tl.where(accept_i != 0, 1, accepted)

    result_f32 = result_f32 * boost
    result_f32 = tl.maximum(result_f32, TINY)

    result = result_f32.to(alpha.dtype)
    tl.store(output_ptr + offsets, result, mask=mask)


def standard_gamma(input, generator=None):
    logger.debug("GEMS_KUNLUNXIN STANDARD_GAMMA")
    logger.debug("GEMS _STANDARD_GAMMA")

    if not torch.is_floating_point(input):
        raise RuntimeError(f"\"standard_gamma\" not implemented for '{input.dtype}'")
    if input.dtype == torch.float64:
        raise RuntimeError("flag_gems standard_gamma does not support float64 inputs")

    input = input.contiguous()
    output = torch.empty_like(input)

    n_elements = volume(input.shape)
    if n_elements == 0:
        return output

    device = input.device
    tiny = torch.finfo(input.dtype).tiny

    # Constant grid tuple (grid depends only on numel): avoids the per-call
    # `grid = lambda` recompile that otherwise costs hundreds of ms/call on XPU.
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)

    with torch_device_fn.device(device):
        increment = triton.cdiv(n_elements * PHILOX_STRIDE, 4)
        philox_seed, philox_offset = philox_backend_seed_offset(
            increment, generator=generator
        )
        standard_gamma_kernel[grid](
            input,
            output,
            n_elements,
            philox_seed,
            philox_offset,
            tiny,
            BLOCK_SIZE=BLOCK_SIZE,
            STRIDE=PHILOX_STRIDE,
            MAX_ITERS=MAX_ITERS,
        )

    return output
