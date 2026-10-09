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

from flag_gems.ops.rrelu_with_noise import (
    DEFAULT_LOWER,
    DEFAULT_UPPER,
    _INT32_MAX,
    _check_rrelu_with_noise_args,
    _launch_contiguous_train,
    _new_output,
    _rrelu_with_noise_eval,
    _rrelu_with_noise_train,
)
from flag_gems.runtime import torch_device_fn
from flag_gems.utils.random_utils import (
    philox_backend_seed_offset,
    uint_to_uniform_float,
)

logger = logging.getLogger(__name__)

FILL_BLOCK = 1024


@triton.jit(do_not_specialize=["lower", "upper", "philox_seed", "philox_offset"])
def _rrelu_uniform_fill_kernel(
    out_ptr,
    n_elements,
    lower,
    upper,
    philox_seed,
    philox_offset,
    BLOCK: tl.constexpr,
):
    philox_seed = philox_seed.to(tl.int64)
    philox_offset = philox_offset.to(tl.int64)
    c0 = (philox_offset & 0xFFFFFFFF).to(tl.uint32)
    c1 = ((philox_offset >> 32) & 0xFFFFFFFF).to(tl.uint32)

    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    c0 += offsets.to(tl.uint32)
    _O = c0 * 0
    r0, r1, r2, r3 = tl.philox(philox_seed, c0, c1, _O, _O)
    u0 = uint_to_uniform_float(r0)

    lower_f = lower.to(tl.float32)
    upper_f = upper.to(tl.float32)
    val = lower_f + u0 * (upper_f - lower_f)

    mask = offsets < n_elements
    tl.store(out_ptr + offsets, val.to(out_ptr.dtype.element_ty), mask=mask)


def _fill_training_noise_xpu(noise, lower, upper, generator):
    if noise.is_contiguous():
        target = noise
    else:
        target = torch.empty_like(noise, memory_format=torch.contiguous_format)

    n_elements = target.numel()
    if n_elements == 0:
        return target

    philox_seed, philox_offset = philox_backend_seed_offset(
        n_elements, generator=generator
    )
    grid = (triton.cdiv(n_elements, FILL_BLOCK),)
    with torch_device_fn.device(target.device):
        _rrelu_uniform_fill_kernel[grid](
            target,
            n_elements,
            float(lower),
            float(upper),
            philox_seed,
            philox_offset,
            BLOCK=FILL_BLOCK,
        )
    return target


def _rrelu_with_noise_impl(
    self,
    noise,
    lower=DEFAULT_LOWER,
    upper=DEFAULT_UPPER,
    training=False,
    generator=None,
    out=None,
):
    _check_rrelu_with_noise_args(self, noise, lower, upper)

    if self.numel() == 0:
        return _new_output(self) if out is None else out

    if training:
        if (
            self.is_contiguous()
            and noise.is_contiguous()
            and self.numel() <= _INT32_MAX
        ):
            output = out if out is not None else _new_output(self)
            return _launch_contiguous_train(
                self, noise, output, lower, upper, generator
            )
        sampled_noise = _fill_training_noise_xpu(noise, lower, upper, generator)
        output = out if out is not None else _new_output(self)
        _rrelu_with_noise_train(self, sampled_noise, out0=output, out1=noise)
        return output
    else:
        slope = (float(lower) + float(upper)) * 0.5
        output = out if out is not None else _new_output(self)
        _rrelu_with_noise_eval(self, slope, out0=output)
        return output


def rrelu_with_noise(
    self,
    noise,
    lower=DEFAULT_LOWER,
    upper=DEFAULT_UPPER,
    training=False,
    generator=None,
):
    logger.debug("GEMS_KUNLUNXIN RRELU_WITH_NOISE")
    return _rrelu_with_noise_impl(self, noise, lower, upper, training, generator)


def rrelu_with_noise_(
    self,
    noise,
    lower=DEFAULT_LOWER,
    upper=DEFAULT_UPPER,
    training=False,
    generator=None,
):
    logger.debug("GEMS_KUNLUNXIN RRELU_WITH_NOISE_")
    return _rrelu_with_noise_impl(
        self, noise, lower, upper, training, generator, out=self
    )
