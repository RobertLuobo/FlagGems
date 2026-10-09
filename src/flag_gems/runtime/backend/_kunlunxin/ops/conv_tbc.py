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

from .addmm import addmm
from .conv1d import conv1d

logger = logging.getLogger(__name__)


def _conv_tbc(input, weight, bias, pad, out=None):
    assert input.dim() == 3, "Input must be 3-dimensional (Time, Batch, Channels)"
    assert weight.dim() == 3, "Weight must be 3-dimensional (kW, Cin, Cout)"
    # conv_tbc: out[t,b,co] = bias[co] + sum_{k,ci} inp[t+k-pad,b,ci]*w[k,ci,co]
    T, B, Cin = input.shape
    kW, _, Cout = weight.shape
    if kW == 1 and pad == 0:
        # Pure GEMM: avoids the conv2d 1x1 unroll-control compile path on XPU3.
        inp2d = input.reshape(T * B, Cin).contiguous()
        res = addmm(bias, inp2d, weight[0].contiguous()).reshape(T, B, Cout)
    else:
        # Stride-1 1-D cross-correlation once laid out as NCL/OIK.
        inp_ncl = input.permute(1, 2, 0).contiguous()  # [B, Cin, T]
        w_oik = weight.permute(2, 1, 0).contiguous()  # [Cout, Cin, kW]
        res = conv1d(inp_ncl, w_oik, bias, stride=1, padding=pad)  # [B, Cout, Tout]
        res = res.permute(2, 0, 1).contiguous()  # [Tout, B, Cout]
    if out is not None:
        out.copy_(res)
        return out
    return res


def conv_tbc(input, weight, bias, pad=0):
    logger.debug("GEMS_KUNLUNXIN CONV_TBC")
    return _conv_tbc(input, weight, bias, pad)


def conv_tbc_out(input, weight, bias, pad=0, *, out):
    logger.debug("GEMS_KUNLUNXIN CONV_TBC_OUT")
    return _conv_tbc(input, weight, bias, pad, out=out)
