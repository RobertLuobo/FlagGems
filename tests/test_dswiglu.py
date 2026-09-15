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

import inspect

import pytest
import torch

import flag_gems

from . import accuracy_utils as utils

try:
    from transformer_engine.pytorch import cpp_extensions as tex

    TE_OP = getattr(tex, "dswiglu", None)
except ImportError:
    TE_OP = None

# TransformerEngine changed the dswiglu signature across releases: newer builds
# take (grad_output, inp, quantizer=...) while older ones only take
# (grad_output, inp). Probe the installed signature so the reference side keeps
# working on both.
_TE_PARAMS = set(inspect.signature(TE_OP).parameters) if TE_OP is not None else set()


def te_dswiglu(grad_output: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
    if "quantizer" in _TE_PARAMS:
        return TE_OP(grad_output, input_tensor, quantizer=None)
    return TE_OP(grad_output, input_tensor)


def generate_input(
    shape: tuple[int, ...], dtype: torch.dtype, device: torch.device
) -> torch.Tensor:
    return torch.randn(shape, dtype=dtype, device=device).contiguous()


def filter_valid_shapes(shapes: list[tuple[int, ...]]) -> list[tuple[int, ...]]:
    valid_shapes = []
    for shape in shapes:
        if not shape:
            continue
        if shape[-1] % 2 == 0:
            valid_shapes.append(shape)
    return valid_shapes


VALID_POINTWISE_SHAPES = filter_valid_shapes(utils.SWIGLU_SPECIAL_SHAPES)


@pytest.mark.dswiglu
@pytest.mark.skipif(TE_OP is None, reason="'dswiglu' not found in TransformerEngine")
@pytest.mark.parametrize("shape", VALID_POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_dswiglu(shape: tuple[int, ...], dtype: torch.dtype):
    torch.manual_seed(42)
    device = flag_gems.device

    input_tensor = generate_input(shape, dtype, device)

    grad_shape = list(shape)
    grad_shape[-1] = grad_shape[-1] // 2
    grad_output = generate_input(tuple(grad_shape), dtype, device)

    te_grad_input = te_dswiglu(grad_output, input_tensor).to(device)
    te_grad_input = utils.to_reference(te_grad_input)

    with flag_gems.use_gems():
        fg_grad_input = flag_gems.dswiglu(grad_output, input_tensor, quantizer=None)

    utils.gems_assert_close(fg_grad_input, te_grad_input, dtype)
