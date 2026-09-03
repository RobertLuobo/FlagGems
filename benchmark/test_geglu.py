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

import flag_gems

from . import base, consts

# Note: Importing transformer_engine (especially in some versions like py 3.10) may automatically
# configure the Root Logger (adding handlers). This may cause subsequent `logging.basicConfig`
# calls (used by FlagGems benchmark) to be ignored/no-op, leading to missing result log files.
# See: https://github.com/NVIDIA/TransformerEngine/issues/1065
try:
    from transformer_engine.pytorch import cpp_extensions as tex

    TE_OP = getattr(tex, "geglu", None)
    TE_AVAILABLE = True
    GEMS_OP = getattr(flag_gems, "geglu", None)
except ImportError:
    TE_AVAILABLE = False
    TE_OP = None
    GEMS_OP = None

# TransformerEngine changed the geglu signature across releases. Newer builds
# take (inp, quantizer=...), while older ones take the FP8-era signature
# (inp, fp8_meta_tensor, fp8_tensor=None, otype=None, ...) and forward `otype`
# straight into the pybind11 `tex.gelu` binding, which rejects `None`. Probe the
# installed signature so the reference side keeps working on both.
_TE_PARAMS = list(inspect.signature(TE_OP).parameters) if TE_OP is not None else []

if "otype" in _TE_PARAMS:
    from transformer_engine.pytorch.constants import TE_DType


def te_geglu(input_tensor, quantizer=None):
    if "otype" in _TE_PARAMS:
        return TE_OP(input_tensor, None, None, TE_DType[input_tensor.dtype])
    if "quantizer" in _TE_PARAMS:
        return TE_OP(input_tensor, quantizer=quantizer)
    return TE_OP(input_tensor, quantizer)


class GegluForwardBenchmark(base.TexGluForwardBenchmark):
    def set_more_shapes(self):
        # base returns lists; Benchmark.init_user_config dedups the merged
        # shapes with dict.fromkeys, which needs hashable entries.
        return [tuple(shape) for shape in super().set_more_shapes()]


@pytest.mark.geglu
@pytest.mark.skipif(not TE_AVAILABLE, reason="TransformerEngine not installed")
@pytest.mark.skipif(TE_OP is None, reason="'geglu' not found in TransformerEngine")
@pytest.mark.skipif(GEMS_OP is None, reason="'geglu' not found in FlagGems")
def test_geglu():
    bench = GegluForwardBenchmark(
        op_name="geglu",
        torch_op=te_geglu,
        gems_op=GEMS_OP,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
