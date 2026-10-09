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

import importlib

import torch

_klx_mm = importlib.import_module("flag_gems.runtime.backend._kunlunxin.ops.mm")
_generic = importlib.import_module("flag_gems.ops.tensordot")


def _matmul_2d(a, b, out=None):
    a = a.contiguous()
    b = b.contiguous()
    M, K = a.shape
    N = b.shape[1]
    if K == 0:
        c_dtype = _klx_mm.get_higher_dtype(a.dtype, b.dtype)
        res = torch.zeros((M, N), device=a.device, dtype=c_dtype)
    else:
        res = _klx_mm.mm(a, b)
    if out is not None:
        out.copy_(res)
        return out
    return res


_generic._matmul_2d = _matmul_2d

tensordot = _generic.tensordot
tensordot_out = _generic.tensordot_out
