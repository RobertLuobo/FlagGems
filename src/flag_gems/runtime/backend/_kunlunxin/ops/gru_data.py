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
import triton
import triton.language as tl

from flag_gems import runtime
from flag_gems.utils import libentry, tl_extra_shim

_base = importlib.import_module("flag_gems.ops.gru")

_FREEZE_BLOCK = 256



@libentry()
@triton.autotune(
    configs=runtime.get_tuned_config("gru"),
    key=["input_size", "hidden_size", "batch_size"],
)
@triton.jit
def _gru_input_gemm_kernel(
    x_ptr,
    w_ih_ptr,
    b_ih_ptr,
    u_ptr,
    batch_sizes_ptr,
    input_size,
    hidden_size,
    batch_size,
    x_stride_s,
    x_stride_b,
    x_stride_f,
    w_ih_stride_r,
    w_ih_stride_c,
    b_ih_stride,
    u_stride_s,
    u_stride_b,
    u_stride_f,
    PACKED: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    BLOCK_B: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    COMPUTE_DTYPE: tl.constexpr,
):
    pid_b = tl.program_id(0)
    seq_idx = tl.program_id(1)
    pid_n = tl.program_id(2)
    offs_b = pid_b * BLOCK_B + tl.arange(0, BLOCK_B)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    b_mask = offs_b < batch_size
    n_mask = offs_n < 3 * hidden_size

    acc = tl.zeros((BLOCK_B, BLOCK_N), dtype=COMPUTE_DTYPE)
    for k_block in range(0, tl.cdiv(input_size, BLOCK_K)):
        offs_k = k_block * BLOCK_K + tl.arange(0, BLOCK_K)
        x = tl.load(
            x_ptr
            + seq_idx * x_stride_s
            + offs_b[:, None] * x_stride_b
            + offs_k[None, :] * x_stride_f,
            mask=(offs_b[:, None] < batch_size) & (offs_k[None, :] < input_size),
            other=0.0,
        )
        w = tl.load(
            w_ih_ptr
            + offs_k[:, None] * w_ih_stride_r
            + offs_n[None, :] * w_ih_stride_c,
            mask=(offs_k[:, None] < input_size) & (offs_n[None, :] < 3 * hidden_size),
            other=0.0,
        )
        acc += tl.dot(x, w, out_dtype=COMPUTE_DTYPE, allow_tf32=False)

    if HAS_BIAS:
        b = tl.load(b_ih_ptr + offs_n * b_ih_stride, mask=n_mask, other=0.0)
        acc += b[None, :]

    out_offsets = (
        seq_idx * u_stride_s
        + offs_b[:, None] * u_stride_b
        + offs_n[None, :] * u_stride_f
    )
    tl.store(u_ptr + out_offsets, acc, mask=b_mask[:, None] & n_mask[None, :])


@libentry()
@triton.jit
def _gru_step_kernel(
    u_ptr,
    h_prev_ptr,
    w_hh_ptr,
    b_hh_ptr,
    h_next_ptr,
    out_ptr,
    batch_sizes_ptr,
    seq_idx,
    out_feature_offset,
    hidden_size,
    batch_size,
    u_stride_s,
    u_stride_b,
    u_stride_f,
    w_hh_stride_r,
    w_hh_stride_c,
    b_hh_stride,
    out_stride_s,
    out_stride_b,
    out_stride_f,
    HAS_BIAS: tl.constexpr,
    BLOCK_B: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_K: tl.constexpr,
    COMPUTE_DTYPE: tl.constexpr,
    PACKED: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    offs_b = pid_b * BLOCK_B + tl.arange(0, BLOCK_B)
    offs_h = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    bh_mask = (offs_b[:, None] < batch_size) & (offs_h[None, :] < hidden_size)

    u_base = (
        seq_idx * u_stride_s
        + offs_b[:, None] * u_stride_b
        + offs_h[None, :] * u_stride_f
    )
    r_acc = tl.load(u_ptr + u_base, mask=bh_mask, other=0.0).to(COMPUTE_DTYPE)
    z_acc = tl.load(
        u_ptr + u_base + hidden_size * u_stride_f, mask=bh_mask, other=0.0
    ).to(COMPUTE_DTYPE)
    n_in = tl.load(
        u_ptr + u_base + 2 * hidden_size * u_stride_f, mask=bh_mask, other=0.0
    ).to(COMPUTE_DTYPE)
    n_h_acc = tl.zeros((BLOCK_B, BLOCK_H), dtype=COMPUTE_DTYPE)

    for k_block in range(0, tl.cdiv(hidden_size, BLOCK_K)):
        offs_k = k_block * BLOCK_K + tl.arange(0, BLOCK_K)
        h = tl.load(
            h_prev_ptr + offs_b[:, None] * hidden_size + offs_k[None, :],
            mask=(offs_b[:, None] < batch_size) & (offs_k[None, :] < hidden_size),
            other=0.0,
        )

        w_r = tl.load(
            w_hh_ptr
            + offs_k[:, None] * w_hh_stride_r
            + offs_h[None, :] * w_hh_stride_c,
            mask=(offs_k[:, None] < hidden_size) & (offs_h[None, :] < hidden_size),
            other=0.0,
        )
        w_z = tl.load(
            w_hh_ptr
            + offs_k[:, None] * w_hh_stride_r
            + (hidden_size + offs_h[None, :]) * w_hh_stride_c,
            mask=(offs_k[:, None] < hidden_size) & (offs_h[None, :] < hidden_size),
            other=0.0,
        )
        w_n = tl.load(
            w_hh_ptr
            + offs_k[:, None] * w_hh_stride_r
            + (2 * hidden_size + offs_h[None, :]) * w_hh_stride_c,
            mask=(offs_k[:, None] < hidden_size) & (offs_h[None, :] < hidden_size),
            other=0.0,
        )
        r_acc += tl.dot(h, w_r, out_dtype=COMPUTE_DTYPE, allow_tf32=False)
        z_acc += tl.dot(h, w_z, out_dtype=COMPUTE_DTYPE, allow_tf32=False)
        n_h_acc += tl.dot(h, w_n, out_dtype=COMPUTE_DTYPE, allow_tf32=False)

    if HAS_BIAS:
        b_hr = tl.load(
            b_hh_ptr + offs_h * b_hh_stride, mask=offs_h < hidden_size, other=0.0
        )
        b_hz = tl.load(
            b_hh_ptr + (hidden_size + offs_h) * b_hh_stride,
            mask=offs_h < hidden_size,
            other=0.0,
        )
        b_hn = tl.load(
            b_hh_ptr + (2 * hidden_size + offs_h) * b_hh_stride,
            mask=offs_h < hidden_size,
            other=0.0,
        )
        r_acc += b_hr[None, :]
        z_acc += b_hz[None, :]
        n_h_acc += b_hn[None, :]

    r_gate = tl.sigmoid(r_acc)
    z_gate = tl.sigmoid(z_acc)
    n_gate = tl_extra_shim.tanh(n_in + r_gate * n_h_acc)

    h_prev = tl.load(
        h_prev_ptr + offs_b[:, None] * hidden_size + offs_h[None, :],
        mask=bh_mask,
        other=0.0,
    ).to(COMPUTE_DTYPE)
    h_next = (1.0 - z_gate) * n_gate + z_gate * h_prev

    state_offsets = offs_b[:, None] * hidden_size + offs_h[None, :]
    tl.store(h_next_ptr + state_offsets, h_next, mask=bh_mask)


# Rewire the generic driver to the XPU3-safe kernels above.
_base._gru_input_gemm_kernel = _gru_input_gemm_kernel
_base._gru_step_kernel = _gru_step_kernel


@libentry()
@triton.jit
def _gru_output_kernel(
    hist_ptr,
    out_ptr,
    out_stride_s,
    out_stride_b,
    out_stride_f,
    out_feature_offset,
    seq_len,
    batch_size,
    hidden_size,
    REVERSE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    total = seq_len * batch_size * hidden_size
    mask = offs < total
    bh = batch_size * hidden_size
    s = offs // bh
    rem = offs % bh
    b_idx = rem // hidden_size
    h_idx = rem % hidden_size
    val = tl.load(hist_ptr + offs, mask=mask, other=0.0)
    if REVERSE:
        seq_idx = seq_len - 1 - s
    else:
        seq_idx = s
    out_off = (
        seq_idx * out_stride_s
        + b_idx * out_stride_b
        + (out_feature_offset + h_idx) * out_stride_f
    )
    tl.store(out_ptr + out_off, val, mask=mask)


@libentry()
@triton.jit
def _gru_freeze_kernel(
    h_next_ptr,
    h_prev_ptr,
    bs_t,
    batch_size,
    hidden_size,
    BLOCK: tl.constexpr,
):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < batch_size * hidden_size
    b_idx = offs // hidden_size
    frozen = b_idx >= bs_t
    hn = tl.load(h_next_ptr + offs, mask=mask, other=0.0)
    hp = tl.load(h_prev_ptr + offs, mask=mask, other=0.0)
    tl.store(h_next_ptr + offs, tl.where(frozen, hp, hn), mask=mask)


def _run_direction(
    layer_input,
    hx,
    layer_output,
    final_h,
    params,
    state_idx,
    param_idx,
    out_feature_offset,
    input_size,
    hidden_size,
    batch_size,
    seq_len,
    has_biases,
    reverse,
    batch_sizes=None,
):
    w_ih, w_hh, b_ih, b_hh = _base._param_group(params, param_idx, has_biases)
    _base._validate_weight(w_ih, 3 * hidden_size, input_size)
    _base._validate_weight(w_hh, 3 * hidden_size, hidden_size)
    if w_ih.dim() == 1:
        w_ih = w_ih.view(3 * hidden_size, input_size)
    if w_hh.dim() == 1:
        w_hh = w_hh.view(3 * hidden_size, hidden_size)
    w_ih = _base._transpose_weight(w_ih, 3 * hidden_size, input_size)
    w_hh = _base._transpose_weight(w_hh, 3 * hidden_size, hidden_size)
    w_ih_stride_r, w_ih_stride_c = w_ih.stride(0), w_ih.stride(1)
    w_hh_stride_r, w_hh_stride_c = w_hh.stride(0), w_hh.stride(1)
    b_ih_stride = _base._bias_stride(b_ih, 3 * hidden_size) if has_biases else 1
    b_hh_stride = _base._bias_stride(b_hh, 3 * hidden_size) if has_biases else 1

    if batch_size == 0:
        return

    block_h_step = _base._block_size(hidden_size, _base._STEP_BLOCK_H)
    block_k_step = _base._block_size(hidden_size, _base._STEP_BLOCK_K)
    grid = (
        triton.cdiv(batch_size, _base._BLOCK_B),
        triton.cdiv(hidden_size, block_h_step),
    )

    if layer_input.dtype == torch.float64:
        compute_dtype = tl.float64
        gate_dtype = torch.float64
    else:
        compute_dtype = tl.float32
        gate_dtype = torch.float32

    packed = batch_sizes is not None
    bs_host = batch_sizes.tolist() if packed else None

    input_gates = _base._empty(
        (seq_len, batch_size, 3 * hidden_size), gate_dtype, layer_input.device
    )
    input_gemm_grid = lambda META: (  # noqa: E731
        triton.cdiv(batch_size, META["BLOCK_B"]),
        seq_len,
        triton.cdiv(3 * hidden_size, META["BLOCK_N"]),
    )

    with runtime.torch_device_fn.device(layer_input.device):
        _gru_input_gemm_kernel[input_gemm_grid](
            layer_input,
            w_ih,
            b_ih,
            input_gates,
            batch_sizes if packed else input_gates,
            input_size,
            hidden_size,
            batch_size,
            layer_input.stride(0),
            layer_input.stride(1),
            layer_input.stride(2),
            w_ih_stride_r,
            w_ih_stride_c,
            b_ih_stride,
            input_gates.stride(0),
            input_gates.stride(1),
            input_gates.stride(2),
            PACKED=packed,
            HAS_BIAS=has_biases,
            COMPUTE_DTYPE=compute_dtype,
        )

        n_elem = batch_size * hidden_size
        freeze_grid = (triton.cdiv(n_elem, _FREEZE_BLOCK),)
        # Full state history with one unique slice per step (no aliasing).
        hist = _base._empty(
            (seq_len + 1, batch_size, hidden_size), hx.dtype, hx.device
        )
        _base._copy_hx_slice(hx, hist[0], state_idx, batch_size, hidden_size)
        for step in range(seq_len):
            seq_idx = seq_len - 1 - step if reverse else step
            h_prev = hist[step]
            h_next = hist[step + 1]
            _gru_step_kernel[grid](
                input_gates,
                h_prev,
                w_hh,
                b_hh,
                h_next,
                layer_output,
                batch_sizes if packed else h_prev,
                seq_idx,
                out_feature_offset,
                hidden_size,
                batch_size,
                input_gates.stride(0),
                input_gates.stride(1),
                input_gates.stride(2),
                w_hh_stride_r,
                w_hh_stride_c,
                b_hh_stride,
                layer_output.stride(0),
                layer_output.stride(1),
                layer_output.stride(2),
                HAS_BIAS=has_biases,
                BLOCK_B=_base._BLOCK_B,
                BLOCK_H=block_h_step,
                BLOCK_K=block_k_step,
                COMPUTE_DTYPE=compute_dtype,
                PACKED=False,
                num_warps=_base._STEP_NUM_WARPS,
                num_stages=_base._STEP_NUM_STAGES,
            )
            if packed:
                bs_t = bs_host[seq_idx]
                if bs_t < batch_size:
                    _gru_freeze_kernel[freeze_grid](
                        h_next,
                        h_prev,
                        bs_t,
                        batch_size,
                        hidden_size,
                        BLOCK=_FREEZE_BLOCK,
                    )
        # Scatter the full state history into layer_output in a single launch.
        out_elems = seq_len * batch_size * hidden_size
        out_grid = (triton.cdiv(out_elems, _FREEZE_BLOCK),)
        _gru_output_kernel[out_grid](
            hist[1:],
            layer_output,
            layer_output.stride(0),
            layer_output.stride(1),
            layer_output.stride(2),
            out_feature_offset,
            seq_len,
            batch_size,
            hidden_size,
            REVERSE=reverse,
            BLOCK=_FREEZE_BLOCK,
        )
        final_h_state = hist[seq_len]

    _base._store_hx_slice(final_h_state, final_h, state_idx, batch_size, hidden_size)


# Force the per-step path; the gemv/persistent barrier kernels fail to legalize on XPU3.
def _max_persistent_programs(device):
    return 0


_base._max_persistent_programs = _max_persistent_programs
_base._run_direction = _run_direction


def gru_data(*args, **kwargs):
    return _base.gru_data(*args, **kwargs)


