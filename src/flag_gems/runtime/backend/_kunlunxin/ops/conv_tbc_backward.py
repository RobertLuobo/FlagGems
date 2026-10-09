# Copyright 2026, The FlagOS Contributors.
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

from .mm import mm

logger = logging.getLogger(__name__)


def conv_tbc_backward(grad_output, input, weight, bias, pad):
    logger.debug("GEMS_KUNLUNXIN CONV_TBC_BACKWARD")

    ilen, batch, in_c = input.shape
    kw, _, out_c = weight.shape
    olen = grad_output.shape[0]

    grad_output = grad_output.contiguous()
    input = input.contiguous()
    weight = weight.contiguous()

    # ---- grad_bias: sum grad_output over (time, batch) -> (out_c,) ----
    # Reduce over the leading axis via a (1, M) x (M, out_c) matmul so we reuse
    # the fp32-accumulating vendor mm kernel. The axis-0 vendor sum kernel both
    # miscompiles on tiny shapes (fp16) and loses bf16 accuracy here.
    gout_2d = grad_output.reshape(olen * batch, out_c)
    ones = torch.ones(
        (1, olen * batch), dtype=grad_output.dtype, device=grad_output.device
    )
    grad_bias = mm(ones, gout_2d).reshape(out_c)

    # ---- grad_weight: for each kernel tap, matmul windowed input with grad_output ----
    # lhs: (kw * in_c, olen * batch), rhs: (olen * batch, out_c)
    if pad > 0:
        input_pad = torch.nn.functional.pad(input, (0, 0, 0, 0, pad, pad))
    else:
        input_pad = input
    # windows over time: (kw, batch, in_c, olen)
    input_win = input_pad.unfold(0, olen, 1)
    lhs_w = input_win.permute(0, 2, 3, 1).reshape(kw * in_c, olen * batch).contiguous()
    rhs_w = gout_2d
    grad_weight = mm(lhs_w, rhs_w).reshape(kw, in_c, out_c)

    # ---- grad_input: full convolution of grad_output with flipped weight ----
    # lhs: (ilen * batch, kw * out_c), rhs: (kw * out_c, in_c)
    front = kw - 1 - pad
    grad_out_pad = torch.nn.functional.pad(grad_output, (0, 0, 0, 0, front, front))
    grad_out_win = grad_out_pad.unfold(0, ilen, 1)  # (kw, batch, out_c, ilen)
    lhs_i = (
        grad_out_win.permute(3, 1, 0, 2).reshape(ilen * batch, kw * out_c).contiguous()
    )
    weight_flip = weight.flip(0).permute(0, 2, 1).reshape(kw * out_c, in_c).contiguous()
    grad_input = mm(lhs_i, weight_flip).reshape(ilen, batch, in_c)

    return grad_input, grad_weight, grad_bias
