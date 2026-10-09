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

import torch

from .adaptive_avg_pool3d_backward import _fill_grad_input

logger = logging.getLogger(__name__)


def _adaptive_avg_pool2d_backward(grad_output, self):
    """Gradient of adaptive_avg_pool2d (Kunlunxin/XPU implementation).

    2D adaptive average pooling is the depth==1 special case of the 3D op, so
    this delegates to the proven 3D backward kernels (input-side gather, no
    scatter/atomic) by viewing both tensors as 5D with an inserted singleton
    depth axis. The 3D kernels assign one input element per lane and sum the
    contributions of every output cell whose window covers it.
    """
    logger.debug("GEMS_KUNLUNXIN _ADAPTIVE_AVG_POOL2D_BACKWARD")

    input_is_3d = self.dim() == 3
    inp = self.unsqueeze(0) if input_is_3d else self
    go = grad_output.unsqueeze(0) if input_is_3d else grad_output

    in_n, in_c, in_h, in_w = inp.shape
    out_h, out_w = go.shape[-2], go.shape[-1]

    grad_input = torch.empty(inp.shape, dtype=inp.dtype, device=inp.device)
    if go.numel() == 0 or inp.numel() == 0:
        grad_input.zero_()
        if input_is_3d:
            return grad_input.squeeze(0)
        return grad_input

    # View as 5D (N, C, D=1, H, W). torch.Tensor.view keeps the same storage;
    # grad_input stays contiguous so the kernel writes land in the real buffer.
    inp5 = inp.reshape(in_n, in_c, 1, in_h, in_w)
    go5 = go.reshape(in_n, in_c, 1, out_h, out_w)
    gi5 = grad_input.view(in_n, in_c, 1, in_h, in_w)

    _fill_grad_input(go5, inp5, gi5)

    if input_is_3d:
        return grad_input.squeeze(0)
    return grad_input
