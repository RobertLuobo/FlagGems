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

import torch


def _linalg_eigvals(inp):
    # Dispatch the op with the XPU keyset so that the native torch_xmlir
    # backend fallback (``ts_eager_fallback``) handles it instead of the
    # generic FlagGems ``linalg_eig`` Triton kernel, which is registered at
    # the CUDA dispatch key inside ``use_gems()`` and cannot compile on this
    # XPU backend (``OutOfResources: uni_sram`` in TritonXPUCoreTiling).
    # ``redispatch_boxed`` returns the op result as a plain tuple for
    # schemas with multiple outputs; ``_linalg_eigvals`` has a single output.
    handle = torch.ops.aten._linalg_eigvals.default._handle
    keyset = torch._C.DispatchKeySet(torch._C.DispatchKey.XPU)
    return handle.redispatch_boxed(keyset, inp)