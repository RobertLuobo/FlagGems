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

    TE_OP = getattr(tex, "dswiglu", None)
except ImportError:
    TE_OP = None

# TransformerEngine changed the dswiglu signature across releases: newer builds
# take (grad_output, inp, quantizer=...) while older ones only take
# (grad_output, inp). Probe the installed signature so the reference side keeps
# working on both.
_TE_PARAMS = set(inspect.signature(TE_OP).parameters) if TE_OP is not None else set()


def te_dswiglu(grad_output, input_tensor, quantizer=None):
    if "quantizer" in _TE_PARAMS:
        return TE_OP(grad_output, input_tensor, quantizer=quantizer)
    return TE_OP(grad_output, input_tensor)


class DswigluBackwardBenchmark(base.TexGluBackwardBenchmark):
    def set_more_shapes(self):
        # base returns lists; Benchmark.init_user_config dedups the merged
        # shapes with dict.fromkeys, which needs hashable entries.
        return [tuple(shape) for shape in super().set_more_shapes()]


@pytest.mark.dswiglu
@pytest.mark.skipif(TE_OP is None, reason="'dswiglu' not found in TransformerEngine")
def test_dswiglu():
    bench = DswigluBackwardBenchmark(
        op_name="dswiglu",
        torch_op=te_dswiglu,
        gems_op=flag_gems.dswiglu,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
