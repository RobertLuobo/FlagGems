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
import logging

import triton
import triton.language as tl

from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as tle

# ``flag_gems.ops._fused_adagrad_`` the *attribute* is the re-exported function,
# which shadows the submodule of the same name; import the module object itself.
_generic = importlib.import_module("flag_gems.ops._fused_adagrad_")

logger = logging.getLogger(__name__)


@libentry()
@triton.jit
def _fused_adagrad_kernel(
    param,
    grad,
    state_sum,
    state_step,
    n: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    ACC_DTYPE: tl.constexpr,
    lr: tl.constexpr,
    lr_decay: tl.constexpr,
    weight_decay: tl.constexpr,
    eps: tl.constexpr,
    maximize: tl.constexpr,
    has_grad_scale: tl.constexpr,
    grad_scale,
    has_found_inf: tl.constexpr,
    found_inf,
):
    pid = tle.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n

    step = tl.load(state_step).to(ACC_DTYPE)

    param_load = tl.load(param + offsets, mask=mask, other=0.0)
    grad_load = tl.load(grad + offsets, mask=mask, other=0.0)
    state_sum_load = tl.load(state_sum + offsets, mask=mask, other=0.0)

    param_f = param_load.to(ACC_DTYPE)
    grad_f = grad_load.to(ACC_DTYPE)
    state_sum_f = state_sum_load.to(ACC_DTYPE)

    if has_grad_scale:
        grad_scale_val = tl.load(grad_scale)
        grad_f = grad_f / grad_scale_val

    if maximize:
        grad_f = -grad_f

    if weight_decay > 0:
        grad_f = grad_f + param_f * weight_decay

    state_sum_f = state_sum_f + grad_f * grad_f
    corrected_lr = lr / (1.0 + (step - 1.0) * lr_decay)
    param_f = param_f - corrected_lr * grad_f / (tl.sqrt(state_sum_f) + eps)

    # ATen tests ``found_inf == 1`` exactly and, on a match, leaves the buffers
    # untouched.  The generic kernel expresses this with an early ``return``
    # placed before the masked stores; on XPU3 that mid-kernel control-flow
    # return combined with the masked ``tt.store`` crashes the ``TritonXPUMask``
    # pass at compile time.  Selecting the original value with ``tl.where`` and
    # always issuing the (masked) store is numerically identical -- writing the
    # loaded bits back is a no-op -- and keeps a single straight-line store path.
    if has_found_inf:
        found_inf_val = tl.load(found_inf)
        skip = found_inf_val == 1
        new_param = tl.where(skip, param_load, param_f.to(param_load.dtype))
        new_state = tl.where(skip, state_sum_load, state_sum_f.to(state_sum_load.dtype))
    else:
        new_param = param_f.to(param_load.dtype)
        new_state = state_sum_f.to(state_sum_load.dtype)

    tl.store(param + offsets, new_param, mask=mask)
    tl.store(state_sum + offsets, new_state, mask=mask)


# Reuse the generic front-end validation and launch driver verbatim; only the
# JIT kernel it dispatches to differs (the XPU3-safe variant above).  The driver
# looks the kernel up as a module global at call time, so rebinding it here is
# enough -- no generic source file is edited.
_generic._fused_adagrad_kernel = _fused_adagrad_kernel


def _fused_adagrad_(
    params,
    grads,
    state_sums,
    state_steps,
    *,
    lr: float = 1e-2,
    lr_decay: float = 0.0,
    weight_decay: float = 0.0,
    eps: float = 1e-10,
    maximize: bool = False,
    grad_scale=None,
    found_inf=None,
):
    logger.debug("GEMS_KUNLUNXIN FUSED_ADAGRAD_")
    _generic._fused_adagrad_run(
        params,
        grads,
        state_sums,
        state_steps,
        lr=lr,
        lr_decay=lr_decay,
        weight_decay=weight_decay,
        eps=eps,
        maximize=maximize,
        grad_scale=grad_scale,
        found_inf=found_inf,
    )
    return None
