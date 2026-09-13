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

    TE_OP = getattr(tex, "dgeglu", None)
except ImportError:
    TE_OP = None

# TransformerEngine changed the dgeglu signature across releases. Newer builds
# take (grad_output, inp, quantizer=...), while older ones take the FP8-era
# signature (grad_output, inp, otype) and forward `otype` straight into the
# pybind11 `tex.gelu` binding, which rejects `None`. Probe the installed
# signature so the reference side keeps working on both.
_TE_PARAMS = list(inspect.signature(TE_OP).parameters) if TE_OP is not None else []

if "otype" in _TE_PARAMS:
    from transformer_engine.pytorch.constants import TE_DType


def te_dgeglu(grad_output, input_tensor, otype=None):
    if "otype" in _TE_PARAMS:
        return TE_OP(grad_output, input_tensor, TE_DType[input_tensor.dtype])
    if "quantizer" in _TE_PARAMS:
        return TE_OP(grad_output, input_tensor, quantizer=otype)
    return TE_OP(grad_output, input_tensor, otype)


@pytest.mark.dgeglu
@pytest.mark.skipif(TE_OP is None, reason="'dgeglu' not found in TransformerEngine")
@pytest.mark.parametrize("shape", utils.GLU_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_dgeglu(shape, dtype):
    input_tensor = torch.randn(shape, dtype=dtype, device=flag_gems.device)

    grad_output_shape = list(shape)
    grad_output_shape[-1] //= 2
    grad_output = torch.randn(
        tuple(grad_output_shape), dtype=dtype, device=flag_gems.device
    )
    ref_out = te_dgeglu(grad_output, input_tensor)
    ref_out = utils.to_reference(ref_out)
    with flag_gems.use_gems():
        res_out = flag_gems.dgeglu(grad_output, input_tensor)
    utils.gems_assert_close(res_out, ref_out, dtype)
