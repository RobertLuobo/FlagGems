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

# TransformerEngine's dgeglu takes (grad_output, inp, otype) and forwards `otype`
# straight into the pybind11 `tex.gelu` binding, which rejects `None`. Probe the
# installed signature so the reference side passes a real DType instead.
_TE_PARAMS = list(inspect.signature(TE_OP).parameters) if TE_OP is not None else []

if "otype" in _TE_PARAMS:
    from transformer_engine.pytorch.constants import TE_DType


def te_dgeglu(grad_output: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
    if "otype" in _TE_PARAMS:
        # Evaluate the reference in fp32 (same convention as
        # accuracy_utils.to_reference(upcast=True)). TE builds gelu_derivative out of
        # storage-dtype torch ops, so `sech2 = 1 - tanh(x)**2` catastrophically
        # cancels for |x| >~ 2 and the fp16/bf16 reference carries ~1e-2 absolute
        # noise; the fp32 evaluation is the identical TE formula at higher precision.
        return TE_OP(
            grad_output.float(), input_tensor.float(), TE_DType[torch.float32]
        ).to(input_tensor.dtype)
    return TE_OP(grad_output, input_tensor, None)


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
