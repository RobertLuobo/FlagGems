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

    TE_OP = getattr(tex, "dgeglu", None)
    TE_AVAILABLE = True
except ImportError:
    TE_AVAILABLE = False
    TE_OP = None

# TransformerEngine changed the dgeglu signature across releases. Newer builds
# take (grad_output, inp, quantizer=...), while older ones take the FP8-era
# signature (grad_output, inp, otype) and forward `otype` straight into the
# pybind11 `tex.gelu` binding, which rejects `None`. Probe the installed
# signature so the reference side keeps working on both.
_TE_PARAMS = list(inspect.signature(TE_OP).parameters) if TE_OP is not None else []

if "otype" in _TE_PARAMS:
    from transformer_engine.pytorch.constants import TE_DType


def te_dgeglu(grad_output, inp, otype=None):
    if "otype" in _TE_PARAMS:
        return TE_OP(grad_output, inp, TE_DType[inp.dtype])
    if "quantizer" in _TE_PARAMS:
        return TE_OP(grad_output, inp, quantizer=otype)
    return TE_OP(grad_output, inp, otype)


class DgegluBackwardBenchmark(base.TexGluBackwardBenchmark):
    def set_more_shapes(self):
        # base returns lists; Benchmark.init_user_config dedups the merged
        # shapes with dict.fromkeys, which needs hashable entries.
        return [tuple(shape) for shape in super().set_more_shapes()]


@pytest.mark.dgeglu
@pytest.mark.skipif(not TE_AVAILABLE, reason="TransformerEngine not installed")
@pytest.mark.skipif(TE_OP is None, reason="'dgeglu' not found in TransformerEngine")
def test_dgeglu():
    bench = DgegluBackwardBenchmark(
        op_name="dgeglu",
        torch_op=te_dgeglu,
        gems_op=flag_gems.dgeglu,
        dtypes=consts.FLOAT_DTYPES,
        # TODO(Qiming): Is this flag correct?
        is_backward=False,
    )
    bench.run()
