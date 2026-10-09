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

import triton
import triton.language as tl

import torch

from flag_gems.ops._fused_adagrad_ import (
    _is_non_overlapping_and_dense,
    _storage_view,
    _strides_match,
)
from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as tle

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

    new_param = param_f.to(param_load.dtype)
    new_state_sum = state_sum_f.to(state_sum_load.dtype)

    # ATen tests ``found_inf == 1`` exactly: that value skips the step (AMP
    # recovery), any other value still applies it.  The generic kernel expresses
    # the skip as a data-dependent early ``return``; on XPU3 that lowers to a
    # ``cf.cond_br`` that the TritonXPUMask pass destroys while it still has uses
    # ('arith.cmpf' op operation destroyed but still has uses), crashing the
    # compile.  Replacing the branch with a branch-free ``tl.where`` select keeps
    # the skip semantics (store the original loaded bits unchanged, so the
    # found_inf==1 case stays bit-identical) without any control flow.  A scalar
    # bool folded into the store mask (``mask & (fv != 1)``) is *also* miscompiled
    # on XPU (the store fires anyway), so the select -- not the mask -- is the
    # correct lever here.
    if has_found_inf:
        found_inf_val = tl.load(found_inf)
        skip = found_inf_val == 1
        new_param = tl.where(skip, param_load, new_param)
        new_state_sum = tl.where(skip, state_sum_load, new_state_sum)

    tl.store(param + offsets, new_param, mask=mask)
    tl.store(state_sum + offsets, new_state_sum, mask=mask)


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
    """In-place fused Adagrad step (kunlunxin XPU overlay).

    Mirrors the generic driver's validation and launch loop verbatim (reusing
    its pure layout helpers), but launches an XPU-safe kernel whose ``found_inf``
    skip is branch-free.
    """
    logger.debug("GEMS_KUNLUNXIN FUSED_ADAGRAD_")

    has_grad_scale = grad_scale is not None
    has_found_inf = found_inf is not None
    grad_scale_ptr = grad_scale if has_grad_scale else None
    found_inf_ptr = found_inf if has_found_inf else None

    n_params = len(params)
    for name, seq in (
        ("grads", grads),
        ("state_sums", state_sums),
        ("state_steps", state_steps),
    ):
        if len(seq) != n_params:
            raise RuntimeError(
                f"_fused_adagrad_: expected {name} to have the same length as "
                f"params ({n_params}), got {len(seq)}"
            )

    if n_params > 0:
        expected_device = params[0].device
        expected_dtype = params[0].dtype
        _SUPPORTED = (torch.float16, torch.bfloat16, torch.float32, torch.float64)
        if expected_dtype not in _SUPPORTED:
            raise RuntimeError(
                "_fused_adagrad_ only supports float16/bfloat16/float32/float64 "
                f"inputs, got {expected_dtype} at index 0"
            )
        for i in range(n_params):
            param, grad, state_sum = params[i], grads[i], state_sums[i]
            for name, tensor in (
                ("param", param),
                ("grad", grad),
                ("state_sum", state_sum),
            ):
                if tensor.dtype != expected_dtype:
                    raise RuntimeError(
                        "params, grads, and state_sums must have same dtype, "
                        f"device, and layout: {name}[{i}] dtype {tensor.dtype} "
                        f"differs from params[0] dtype {expected_dtype}"
                    )
                if tensor.device != expected_device:
                    raise RuntimeError(
                        "params, grads, and state_sums must have same dtype, "
                        f"device, and layout: {name}[{i}] device {tensor.device} "
                        f"differs from params[0] device {expected_device}"
                    )
                if tensor.layout != torch.strided:
                    raise RuntimeError(
                        f"{name}[{i}] must be a strided tensor, got layout "
                        f"{tensor.layout}"
                    )
                if not _is_non_overlapping_and_dense(tensor):
                    raise RuntimeError(
                        f"_fused_adagrad_: {name}[{i}] is not "
                        "non-overlapping-and-dense (internal overlap or gappy "
                        "strides), so it has no well-defined element order"
                    )
            for name, other in (("grad", grad), ("state_sum", state_sum)):
                if other.shape != param.shape:
                    raise RuntimeError(
                        f"_fused_adagrad_: {name}[{i}] shape {tuple(other.shape)} "
                        f"does not match param shape {tuple(param.shape)}"
                    )
                if not _strides_match(param, other):
                    raise RuntimeError(
                        f"_fused_adagrad_: {name}[{i}] strides {other.stride()} "
                        f"do not match param strides {param.stride()}"
                    )

    for i, state_step in enumerate(state_steps):
        if state_step.dtype != torch.float32:
            raise RuntimeError(
                "state_steps must contain float32 scalar tensors (one element "
                f"holding the optimizer step), got dtype {state_step.dtype} at "
                f"index {i}"
            )
        if state_step.numel() != 1:
            raise RuntimeError(
                "state_steps must contain 1-element tensors, got numel "
                f"{state_step.numel()} at index {i}"
            )
        if state_step.device != params[0].device:
            raise RuntimeError(
                "Expected all tensors to be on the same device, but got "
                f"state_steps is on {state_step.device}, different from other "
                f"tensors on {params[0].device}"
            )

    for i in range(len(params)):
        param = params[i]
        grad = grads[i]
        state_sum = state_sums[i]
        state_step = state_steps[i]

        n = param.numel()
        if n == 0:
            continue

        BLOCK_SIZE = triton.next_power_of_2(n)
        BLOCK_SIZE = max(BLOCK_SIZE, 128)
        BLOCK_SIZE = min(BLOCK_SIZE, 4096)

        grid = (triton.cdiv(n, BLOCK_SIZE),)

        acc_dtype = tl.float64 if param.dtype == torch.float64 else tl.float32

        _fused_adagrad_kernel[grid](
            _storage_view(param),
            _storage_view(grad),
            _storage_view(state_sum),
            state_step,
            n,
            BLOCK_SIZE,
            acc_dtype,
            lr,
            lr_decay,
            weight_decay,
            eps,
            maximize,
            has_grad_scale,
            grad_scale_ptr,
            has_found_inf,
            found_inf_ptr,
        )

    return None
