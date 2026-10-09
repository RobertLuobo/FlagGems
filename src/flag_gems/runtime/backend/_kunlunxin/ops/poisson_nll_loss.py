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

import math
import sys
import types

import torch
import triton
import triton.language as tl

import flag_gems.ops.poisson_nll_loss  # noqa: F401  ensure module is imported
from flag_gems.runtime import torch_device_fn

# The package namespace shadows the submodule with the exported function, so
# `flag_gems.ops.poisson_nll_loss` resolves to the function; fetch the actual
# module object from sys.modules to patch its host-side globals.
_generic = sys.modules["flag_gems.ops.poisson_nll_loss"]

# On XPU3 libdevice, exp/log map to the sentinel symbol "Unsupported",
# producing `ld.lld: error: undefined symbol: Unsupported` at link time. tl.log
# / tl.exp lower to native XPU intrinsics instead. Use a real ModuleType clone
# so Triton's dependency finder skips it (it only deep-copies non-module globals)
# and resolves the two overridden attributes to tl builtins; every other symbol
# (cos, sin, atan2, ...) stays bound to the original backend shim.
_base_shim = _generic.tl_extra_shim
_xpu_shim = types.ModuleType(_base_shim.__name__ + "_poisson_nll_loss_xpu_shim")
_xpu_shim.__dict__.update(_base_shim.__dict__)
_xpu_shim.log = tl.log
_xpu_shim.exp = tl.exp
_generic.tl_extra_shim = _xpu_shim

# atan2 stays on the original backend shim (it is a working XPU3 intrinsic,
# unlike exp/log which map to the "Unsupported" sentinel).
_atan2 = _base_shim.atan2

_REDUCE_BLOCK = 4096


@triton.jit
def _nomask_reduce_partial(loss_ptr, partial_ptr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    vals = tl.load(loss_ptr + offsets)
    tl.store(partial_ptr + pid, tl.sum(vals))


@triton.jit
def _nomask_reduce_final(
    partial_ptr,
    output_ptr,
    partial_size,
    n_elements,
    reduction: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < partial_size
    total = tl.sum(tl.load(partial_ptr + offsets, mask=mask, other=0.0))
    if reduction == 1:
        total /= n_elements
    tl.store(output_ptr, total)


@triton.jit
def _complex_elem_kernel(
    xr_ptr,
    xi_ptr,
    yr_ptr,
    yi_ptr,
    real_ptr,
    imag_ptr,
    n,
    eps,
    LOG_INPUT: tl.constexpr,
    FULL: tl.constexpr,
    CPLX_IN: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    xr = tl.load(xr_ptr + offsets, mask=mask, other=0.0)
    xi = tl.load(xi_ptr + offsets, mask=mask, other=0.0)
    yr = tl.load(yr_ptr + offsets, mask=mask, other=0.0)
    yi = tl.load(yi_ptr + offsets, mask=mask, other=0.0)
    if LOG_INPUT:
        ex = tl.exp(xr)
        real = ex * tl.cos(xi) - (yr * xr - yi * xi)
        imag = ex * tl.sin(xi) - (yr * xi + yi * xr)
    else:
        shifted = xr + eps
        if CPLX_IN:
            scale = tl.maximum(tl.abs(shifted), tl.abs(xi))
            safe_scale = tl.where(scale == 0, 1.0, scale)
            a = shifted / safe_scale
            b = xi / safe_scale
            lr = tl.log(safe_scale) + 0.5 * tl.log(a * a + b * b)
            li = _atan2(xi, shifted)
        else:
            lr = tl.log(shifted)
            li = tl.zeros_like(xr)
        real = xr - (yr * lr - yi * li)
        imag = xi - (yr * li + yi * lr)
    if FULL:
        stirling = yr * tl.log(yr) - yr + 0.5 * tl.log(2.0 * math.pi * yr)
        real += tl.where(yr > 1.0, stirling, 0.0)
    tl.store(real_ptr + offsets, real, mask=mask)
    tl.store(imag_ptr + offsets, imag, mask=mask)


def _complex_elementwise(input, target, log_input, full, eps, reduction, dtype):
    # The generic complex path routes through a pointwise_dynamic kernel whose
    # 1d_tile t512 config crashes TritonXPUUnrollControl (out of uni_sram) at
    # large element counts on XPU3. This plain elementwise kernel reproduces the
    # exact generic _complex_loss math (exp/cos/sin for log_input, log/atan2 for
    # the complex-log branch, optional Stirling term) without the failing
    # compiler pass. Only taken for matching-shape complex inputs; broadcasting
    # complex forms stay on the generic pointwise_dynamic implementation.
    real_dtype = torch.float32
    xr, xi = _generic._complex_parts(input, real_dtype)
    yr, yi = _generic._complex_parts(target, real_dtype)
    out_shape = xr.shape
    n = xr.numel()
    reduce = reduction in (1, 2)
    if reduce:
        # Zero-padded buffers so the summation runs over whole blocks with no
        # per-lane mask, matching _reduce_via_pad. gems_sum miscompiles the
        # masked tail at element counts that are not a block multiple (e.g.
        # 65537), which silently corrupts the complex reduction.
        nblk = triton.cdiv(n, _REDUCE_BLOCK)
        padded = nblk * _REDUCE_BLOCK
        real_buf = torch.zeros(padded, dtype=real_dtype, device=input.device)
        imag_buf = torch.zeros(padded, dtype=real_dtype, device=input.device)
    else:
        real_buf = torch.empty(n, dtype=real_dtype, device=input.device)
        imag_buf = torch.empty(n, dtype=real_dtype, device=input.device)
    xrf, xif, yrf, yif = (t.reshape(-1) for t in (xr, xi, yr, yi))
    eps_f = float(torch.tensor(eps, dtype=real_dtype).item())
    with torch_device_fn.device(input.device):
        _complex_elem_kernel[(triton.cdiv(n, 1024),)](
            xrf,
            xif,
            yrf,
            yif,
            real_buf,
            imag_buf,
            n,
            eps_f,
            LOG_INPUT=bool(log_input),
            FULL=bool(full),
            CPLX_IN=bool(input.is_complex()),
            BLOCK=1024,
        )
        if reduce:
            output = torch.empty((), dtype=dtype, device=input.device)
            parts = torch.view_as_real(output)
            for src, dst in ((real_buf, parts[..., 0]), (imag_buf, parts[..., 1])):
                partial = torch.empty(nblk, dtype=real_dtype, device=input.device)
                _nomask_reduce_partial[(nblk,)](
                    src, partial, BLOCK_SIZE=_REDUCE_BLOCK
                )
                scalar = torch.empty((), dtype=real_dtype, device=input.device)
                _nomask_reduce_final[(1,)](
                    partial,
                    scalar,
                    nblk,
                    n,
                    reduction=reduction,
                    BLOCK_SIZE=triton.next_power_of_2(nblk),
                )
                _generic.copy_(dst, scalar)
            return output
    output = torch.empty(out_shape, dtype=dtype, device=input.device)
    parts = torch.view_as_real(output)
    _generic.copy_(parts[..., 0], real_buf.reshape(out_shape))
    _generic.copy_(parts[..., 1], imag_buf.reshape(out_shape))
    return output


def _reduce_via_pad(input, target, log_input, full, eps, reduction, dtype):
    output_shape, input_strides, target_strides = _generic._broadcast_metadata(
        input, target
    )
    n_elements = math.prod(output_shape)
    nblk = triton.cdiv(n_elements, _REDUCE_BLOCK)
    padded = nblk * _REDUCE_BLOCK
    loss_dtype = (
        torch.float32 if dtype in (torch.float16, torch.bfloat16) else dtype
    )
    # Zero-padded buffer: masked stores leave the tail at 0.0, so the reduction
    # runs over exact full blocks with no per-lane mask. This sidesteps the XPU3
    # masked-tail reduction miscompile in the generic partial/reduce kernels,
    # which over-counts padding lanes when n_elements is not a block multiple.
    loss_buf = torch.zeros(padded, dtype=loss_dtype, device=input.device)
    is_fp16 = dtype == torch.float16
    is_bf16 = dtype == torch.bfloat16
    is_fp64 = dtype == torch.float64
    eps = float(torch.tensor(eps, dtype=dtype).item())
    output = torch.empty((), dtype=dtype, device=input.device)
    with torch_device_fn.device(input.device):
        _generic._poisson_nll_loss_none_kernel[(triton.cdiv(n_elements, 1024),)](
            input,
            target,
            loss_buf,
            n_elements,
            eps,
            OUT_SHAPE=output_shape,
            INPUT_STRIDES=input_strides,
            TARGET_STRIDES=target_strides,
            LOG_INPUT=bool(log_input),
            FULL=bool(full),
            IS_FP16=is_fp16,
            IS_BF16=is_bf16,
            IS_FP64=is_fp64,
            BLOCK_SIZE=1024,
        )
        partial = torch.empty(nblk, dtype=loss_dtype, device=input.device)
        _nomask_reduce_partial[(nblk,)](loss_buf, partial, BLOCK_SIZE=_REDUCE_BLOCK)
        _nomask_reduce_final[(1,)](
            partial,
            output,
            nblk,
            n_elements,
            reduction=reduction,
            BLOCK_SIZE=triton.next_power_of_2(nblk),
        )
    return output


def poisson_nll_loss(input, target, log_input, full, eps, reduction):
    if input.device != target.device:
        raise RuntimeError("input and target must be on the same device")
    if full and target.dtype == torch.bool:
        raise RuntimeError("Subtraction with a bool target is not supported")
    if not log_input and input.dtype == torch.bool:
        raise RuntimeError("Subtraction with a bool input is not supported")
    if full and target.is_complex():
        raise RuntimeError("Comparisons with a complex target are not supported")
    dtype = _generic._result_dtype(input, target)
    # A bool input tensor loads as 0 through the XPU3 i1 -> tl.load path, so the
    # kernel would compute exp(0) instead of exp(True). Marshal it to float32
    # first (only reachable with log_input=True; non-log bool input is rejected
    # above), which matches the default-float promotion _result_dtype already
    # applies to a non-floating input and leaves every value identical.
    if input.dtype == torch.bool:
        casted = torch.empty(input.shape, dtype=torch.float32, device=input.device)
        _generic.copy_(casted, input)
        input = casted
    output_shape, _, _ = _generic._broadcast_metadata(input, target)
    n_elements = math.prod(output_shape)
    if (
        dtype == torch.complex64
        and n_elements > 0
        and tuple(input.shape) == tuple(target.shape)
    ):
        return _complex_elementwise(
            input, target, log_input, full, eps, reduction, dtype
        )
    if reduction in (1, 2) and n_elements > 0 and not dtype.is_complex:
        return _reduce_via_pad(input, target, log_input, full, eps, reduction, dtype)
    return _generic.poisson_nll_loss(input, target, log_input, full, eps, reduction)
