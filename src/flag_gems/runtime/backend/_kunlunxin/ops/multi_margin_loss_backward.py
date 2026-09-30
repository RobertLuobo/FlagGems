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

import sys

import torch

import flag_gems.runtime.backend._kunlunxin.ops.multi_margin_loss  # noqa: F401

_generic = sys.modules["flag_gems.ops.multi_margin_loss"]


def multi_margin_loss_backward(
    grad_output: torch.Tensor,
    input: torch.Tensor,
    target: torch.Tensor,
    p,
    margin,
    weight=None,
    reduction=1,
) -> torch.Tensor:
    return _generic.multi_margin_loss_backward(
        grad_output,
        input,
        target,
        p,
        margin,
        weight,
        reduction,
    )


def multi_margin_loss_backward_out(
    grad_output: torch.Tensor,
    input: torch.Tensor,
    target: torch.Tensor,
    p,
    margin,
    weight=None,
    reduction=1,
    *,
    grad_input: torch.Tensor,
) -> torch.Tensor:
    return _generic.multi_margin_loss_backward_out(
        grad_output,
        input,
        target,
        p,
        margin,
        weight,
        reduction,
        grad_input=grad_input,
    )
