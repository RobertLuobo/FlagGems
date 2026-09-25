import logging
import sys

import torch
import triton
import triton.language as tl

# The generic implementation carries all the GRU triton kernels and host helpers.
# ``flag_gems.ops`` rebinds the ``gru`` attribute to the *function* (via
# ``from .gru import gru``), so ``import flag_gems.ops.gru`` would resolve to that
# function rather than the module. Grab the real module object from sys.modules.
import flag_gems.ops.gru  # noqa: F401  (ensure the module is imported)
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry, tl_extra_shim

_g = sys.modules["flag_gems.ops.gru"]

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# XPU / SDNN has no atomic scatter: the generic fast paths (``_gru_persistent_kernel``
# and ``_gru_gemv_kernel``) implement a grid-wide barrier with ``tl.atomic_add``
# (generic gru.py:576 / :735), which fails to legalize in ConvertTritonSDNNToLLVM
# ("failed to legalize operation 'tt.atomic_rmw'/'tt.addptr'"). This override runs
# a barrier-free per-timestep recurrence instead: stream ordering between launches
# provides the cross-program synchronization the atomic barrier would give.
#
# The generic ``_gru_step_kernel`` co-locates ``tl.sigmoid``/``tanh`` with ``tl.dot``
# and the K-reduction loop in one kernel body. On this backend that miscompiles the
# transcendentals to a reduced-precision (~5e-5) variant, which the recurrence
# amplifies past the fp32 atol (1e-4). We therefore split the step into two kernels:
#   * ``_gate_gemm``  -- dot + K loop only, emits the three gate pre-activations.
#   * ``_gate_act``   -- pure elementwise, applies sigmoid/tanh at full precision.
# All ``tl.dot`` calls use ``input_precision="ieee"`` (fp32-exact on P800; the
# generic ``allow_tf32=False`` path leaves ~1e-4 residual matmul error).
# ---------------------------------------------------------------------------


@libentry()
@triton.jit
def _in_gemm(
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
    # Batched input-side pre-activation: u = x @ W_ih^T + b_ih for every timestep.
    pid_b = tl.program_id(0)
    seq_idx = tl.program_id(1)
    pid_n = tl.program_id(2)
    offs_b = pid_b * BLOCK_B + tl.arange(0, BLOCK_B)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    b_mask = offs_b < batch_size
    n_mask = offs_n < 3 * hidden_size
    # NOTE: no PACKED early-return here. The SDNN TritonSDNNLoopGrid pass rejects an
    # early-return block that also contains a store ("cannot find unique early-return
    # block"). Padding rows (b >= batch_sizes[seq]) are harmless: their input x is
    # zero-filled, and the recurrence output is frozen to h_prev in ``_gate_act`` via
    # the row-active mask, so whatever these rows accumulate here is never observed.
    acc = tl.zeros((BLOCK_B, BLOCK_N), dtype=COMPUTE_DTYPE)
    for kb in range(0, tl.cdiv(input_size, BLOCK_K)):
        offs_k = kb * BLOCK_K + tl.arange(0, BLOCK_K)
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
        acc += tl.dot(x, w, out_dtype=COMPUTE_DTYPE, input_precision="ieee")
    if HAS_BIAS:
        b = tl.load(b_ih_ptr + offs_n * b_ih_stride, mask=n_mask, other=0.0)
        acc += b[None, :]
    oo = (
        seq_idx * u_stride_s
        + offs_b[:, None] * u_stride_b
        + offs_n[None, :] * u_stride_f
    )
    tl.store(u_ptr + oo, acc, mask=b_mask[:, None] & n_mask[None, :])


@libentry()
@triton.jit
def _gate_gemm(
    u_ptr,
    h_prev_ptr,
    w_hh_ptr,
    b_hh_ptr,
    gate_ptr,
    batch_sizes_ptr,
    seq_idx,
    hidden_size,
    batch_size,
    u_stride_s,
    u_stride_b,
    u_stride_f,
    w_hh_stride_r,
    w_hh_stride_c,
    b_hh_stride,
    gate_stride_b,
    gate_stride_f,
    HAS_BIAS: tl.constexpr,
    BLOCK_B: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_K: tl.constexpr,
    COMPUTE_DTYPE: tl.constexpr,
    PACKED: tl.constexpr,
):
    # Dot + loop kernel: compute the three gate pre-activations (r, z pre-sigmoid
    # accumulators and the hidden-side n term). Sigmoid/tanh are applied later in a
    # separate elementwise kernel because this backend compiles the transcendentals
    # to a reduced-precision (~1e-4) variant when they share a kernel body with tl.dot.
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    offs_b = pid_b * BLOCK_B + tl.arange(0, BLOCK_B)
    offs_h = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    bh_mask = (offs_b[:, None] < batch_size) & (offs_h[None, :] < hidden_size)
    gate_off_r = offs_b[:, None] * gate_stride_b + offs_h[None, :] * gate_stride_f
    gate_off_z = (
        offs_b[:, None] * gate_stride_b
        + (hidden_size + offs_h[None, :]) * gate_stride_f
    )
    gate_off_n = (
        offs_b[:, None] * gate_stride_b
        + (2 * hidden_size + offs_h[None, :]) * gate_stride_f
    )
    # No PACKED early-return (see _in_gemm). Padding rows compute throwaway gate
    # pre-activations here; ``_gate_act`` overwrites their output with h_prev.
    u_base = (
        seq_idx * u_stride_s
        + offs_b[:, None] * u_stride_b
        + offs_h[None, :] * u_stride_f
    )
    r_acc = tl.load(u_ptr + u_base, mask=bh_mask, other=0.0).to(COMPUTE_DTYPE)
    z_acc = tl.load(
        u_ptr + u_base + hidden_size * u_stride_f, mask=bh_mask, other=0.0
    ).to(COMPUTE_DTYPE)
    n_h_acc = tl.zeros((BLOCK_B, BLOCK_H), dtype=COMPUTE_DTYPE)
    for kb in range(0, tl.cdiv(hidden_size, BLOCK_K)):
        offs_k = kb * BLOCK_K + tl.arange(0, BLOCK_K)
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
        r_acc += tl.dot(h, w_r, out_dtype=COMPUTE_DTYPE, input_precision="ieee")
        z_acc += tl.dot(h, w_z, out_dtype=COMPUTE_DTYPE, input_precision="ieee")
        n_h_acc += tl.dot(h, w_n, out_dtype=COMPUTE_DTYPE, input_precision="ieee")
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
    tl.store(gate_ptr + gate_off_r, r_acc, mask=bh_mask)
    tl.store(gate_ptr + gate_off_z, z_acc, mask=bh_mask)
    tl.store(gate_ptr + gate_off_n, n_h_acc, mask=bh_mask)


@libentry()
@triton.jit
def _gate_act(
    u_ptr,
    gate_ptr,
    h_prev_ptr,
    h_next_ptr,
    out_ptr,
    seq_idx,
    out_feature_offset,
    hidden_size,
    batch_size,
    u_stride_s,
    u_stride_b,
    u_stride_f,
    gate_stride_b,
    gate_stride_f,
    out_stride_s,
    out_stride_b,
    out_stride_f,
    BLOCK_B: tl.constexpr,
    BLOCK_H: tl.constexpr,
    COMPUTE_DTYPE: tl.constexpr,
):
    # Pure elementwise: apply sigmoid/tanh and the GRU gating. No tl.dot / reduction
    # loop here, so the XPU backend keeps the full-precision transcendental path.
    #
    # This kernel is *identical* for dense and packed runs: it always writes the freshly
    # computed h_new to both the recurrence state and the output. Row freezing for packed
    # sequences is applied on the host (contiguous frozen tail copy in ``_run_direction``)
    # rather than with an in-kernel ``tl.where``: that ``tl.where`` poisons this backend's
    # strided ``out_ptr`` store codegen for small hidden sizes (garbage output at scattered
    # timesteps even when no row actually freezes), whereas the dense path -- byte-for-byte
    # the same shapes without the where -- is correct.
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    offs_b = pid_b * BLOCK_B + tl.arange(0, BLOCK_B)
    offs_h = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    bh_mask = (offs_b[:, None] < batch_size) & (offs_h[None, :] < hidden_size)
    state_off = offs_b[:, None] * hidden_size + offs_h[None, :]
    h_prev = tl.load(h_prev_ptr + state_off, mask=bh_mask, other=0.0).to(COMPUTE_DTYPE)
    out_off = (
        seq_idx * out_stride_s
        + offs_b[:, None] * out_stride_b
        + (out_feature_offset + offs_h[None, :]) * out_stride_f
    )
    gate_off_r = offs_b[:, None] * gate_stride_b + offs_h[None, :] * gate_stride_f
    gate_off_z = (
        offs_b[:, None] * gate_stride_b
        + (hidden_size + offs_h[None, :]) * gate_stride_f
    )
    gate_off_n = (
        offs_b[:, None] * gate_stride_b
        + (2 * hidden_size + offs_h[None, :]) * gate_stride_f
    )
    r_acc = tl.load(gate_ptr + gate_off_r, mask=bh_mask, other=0.0)
    z_acc = tl.load(gate_ptr + gate_off_z, mask=bh_mask, other=0.0)
    n_h_acc = tl.load(gate_ptr + gate_off_n, mask=bh_mask, other=0.0)
    u_base = (
        seq_idx * u_stride_s
        + offs_b[:, None] * u_stride_b
        + (2 * hidden_size + offs_h[None, :]) * u_stride_f
    )
    n_in = tl.load(u_ptr + u_base, mask=bh_mask, other=0.0).to(COMPUTE_DTYPE)
    r_gate = tl.sigmoid(r_acc)
    z_gate = tl.sigmoid(z_acc)
    n_gate = tl_extra_shim.tanh(n_in + r_gate * n_h_acc)
    h_new = (1.0 - z_gate) * n_gate + z_gate * h_prev
    tl.store(out_ptr + out_off, h_new, mask=bh_mask)
    tl.store(h_next_ptr + state_off, h_new, mask=bh_mask)


def _run_direction(
    layer_input,
    hx,
    layer_output,
    final_h,
    params,
    state_idx: int,
    param_idx: int,
    out_feature_offset: int,
    input_size: int,
    hidden_size: int,
    batch_size: int,
    seq_len: int,
    has_biases: bool,
    reverse: bool,
    batch_sizes=None,
):
    w_ih, w_hh, b_ih, b_hh = _g._param_group(params, param_idx, has_biases)
    _g._validate_weight(w_ih, 3 * hidden_size, input_size)
    _g._validate_weight(w_hh, 3 * hidden_size, hidden_size)
    if w_ih.dim() == 1:
        w_ih = w_ih.view(3 * hidden_size, input_size)
    if w_hh.dim() == 1:
        w_hh = w_hh.view(3 * hidden_size, hidden_size)
    w_ih = _g._transpose_weight(w_ih, 3 * hidden_size, input_size)
    w_hh = _g._transpose_weight(w_hh, 3 * hidden_size, hidden_size)
    w_ih_stride_r, w_ih_stride_c = w_ih.stride(0), w_ih.stride(1)
    w_hh_stride_r, w_hh_stride_c = w_hh.stride(0), w_hh.stride(1)
    b_ih_stride = _g._bias_stride(b_ih, 3 * hidden_size) if has_biases else 1
    b_hh_stride = _g._bias_stride(b_hh, 3 * hidden_size) if has_biases else 1

    if batch_size == 0:
        return

    block_h_step = _g._block_size(hidden_size, _g._STEP_BLOCK_H)
    block_k_step = _g._block_size(hidden_size, _g._STEP_BLOCK_K)
    grid = (
        triton.cdiv(batch_size, _g._BLOCK_B),
        triton.cdiv(hidden_size, block_h_step),
    )

    if layer_input.dtype == torch.float64:
        compute_dtype = tl.float64
        gate_dtype = torch.float64
    else:
        compute_dtype = tl.float32
        gate_dtype = torch.float32

    # Precompute input-side pre-activations for all timesteps in one batched GEMM.
    input_gates = _g._empty(
        (seq_len, batch_size, 3 * hidden_size), gate_dtype, layer_input.device
    )
    # Scratch for the per-step gate pre-activations produced by ``_gate_gemm`` and
    # consumed by ``_gate_act`` (r | z | n_h laid out along the feature axis).
    gate_buf = _g._empty((batch_size, 3 * hidden_size), gate_dtype, layer_input.device)

    BLOCK_B_IN, BLOCK_N_IN, BLOCK_K_IN = 16, 64, 32
    in_grid = (
        triton.cdiv(batch_size, BLOCK_B_IN),
        seq_len,
        triton.cdiv(3 * hidden_size, BLOCK_N_IN),
    )

    # Host-side per-timestep active-row counts. Passing batch_sizes[seq] as a plain
    # kernel scalar (rather than an in-kernel tl.load) keeps the ``_gate_act`` strided
    # output store codegen correct on this backend.
    if batch_sizes is not None:
        bs_host = batch_sizes.to("cpu", torch.int32).tolist()
    else:
        bs_host = None

    with torch_device_fn.device(layer_input.device):
        _in_gemm[in_grid](
            layer_input,
            w_ih,
            b_ih,
            input_gates,
            batch_sizes if batch_sizes is not None else input_gates,
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
            PACKED=batch_sizes is not None,
            HAS_BIAS=has_biases,
            BLOCK_B=BLOCK_B_IN,
            BLOCK_N=BLOCK_N_IN,
            BLOCK_K=BLOCK_K_IN,
            COMPUTE_DTYPE=compute_dtype,
        )

        # Barrier-free recurrence: one gate-GEMM + one activation launch per timestep.
        # Stream ordering between launches supplies the cross-program synchronization
        # that the generic persistent/gemv kernels get from the (unsupported) atomic
        # barrier. Sigmoid/tanh live in ``_gate_act`` (no tl.dot) to keep full precision.
        h_work = _g._empty((batch_size, hidden_size), hx.dtype, hx.device)
        _g._copy_hx_slice(hx, h_work, state_idx, batch_size, hidden_size)
        h_next = _g._empty((batch_size, hidden_size), hx.dtype, hx.device)
        for step in range(seq_len):
            seq_idx = seq_len - 1 - step if reverse else step
            _gate_gemm[grid](
                input_gates,
                h_work,
                w_hh,
                b_hh,
                gate_buf,
                batch_sizes if batch_sizes is not None else h_work,
                seq_idx,
                hidden_size,
                batch_size,
                input_gates.stride(0),
                input_gates.stride(1),
                input_gates.stride(2),
                w_hh_stride_r,
                w_hh_stride_c,
                b_hh_stride,
                gate_buf.stride(0),
                gate_buf.stride(1),
                HAS_BIAS=has_biases,
                BLOCK_B=_g._BLOCK_B,
                BLOCK_H=block_h_step,
                BLOCK_K=block_k_step,
                COMPUTE_DTYPE=compute_dtype,
                PACKED=batch_sizes is not None,
                num_warps=_g._STEP_NUM_WARPS,
                num_stages=_g._STEP_NUM_STAGES,
            )
            _gate_act[grid](
                input_gates,
                gate_buf,
                h_work,
                h_next,
                layer_output,
                seq_idx,
                out_feature_offset,
                hidden_size,
                batch_size,
                input_gates.stride(0),
                input_gates.stride(1),
                input_gates.stride(2),
                gate_buf.stride(0),
                gate_buf.stride(1),
                layer_output.stride(0),
                layer_output.stride(1),
                layer_output.stride(2),
                BLOCK_B=_g._BLOCK_B,
                BLOCK_H=block_h_step,
                COMPUTE_DTYPE=compute_dtype,
            )
            # Packed row freezing: sequences shorter than the current step must keep
            # their previous hidden state. enforce_sorted guarantees the still-active
            # rows are a contiguous prefix, so the frozen rows are the tail
            # [active_count:batch]. Restore that tail (a contiguous device slice copy)
            # rather than masking inside the kernel -- see the note in ``_gate_act``.
            if bs_host is not None:
                active_count = bs_host[seq_idx]
                if active_count < batch_size:
                    h_next[active_count:batch_size].copy_(
                        h_work[active_count:batch_size]
                    )
            h_work, h_next = h_next, h_work
        final_h_state = h_work

    _g._store_hx_slice(final_h_state, final_h, state_idx, batch_size, hidden_size)


def _gru_forward_impl(
    input_view,
    hx,
    params,
    output,
    final_h,
    num_layers,
    num_directions,
    hidden_size,
    input_size,
    batch_size,
    seq_len,
    has_biases,
    train,
    dropout,
    batch_sizes=None,
):
    # Layer/direction driver copied from the generic implementation so it calls the
    # barrier-free ``_run_direction`` above (rather than the generic one that would
    # select the atomic fast paths). No generic module state is mutated.
    layer_input = input_view
    for layer in range(num_layers):
        layer_input_size = input_size if layer == 0 else hidden_size * num_directions
        if layer == num_layers - 1:
            layer_output = output
        else:
            layer_output = _g._empty(
                (seq_len, batch_size, hidden_size * num_directions),
                input_view.dtype,
                input_view.device,
            )
        for direction in range(num_directions):
            state_idx = layer * num_directions + direction
            reverse = direction == 1
            _run_direction(
                layer_input,
                hx,
                layer_output,
                final_h,
                params,
                state_idx,
                state_idx,
                direction * hidden_size,
                layer_input_size,
                hidden_size,
                batch_size,
                seq_len,
                has_biases,
                reverse,
                batch_sizes,
            )

        layer_input = layer_output
        if train and dropout != 0.0 and layer + 1 < num_layers:
            layer_input, _ = _g._dropout(layer_input, dropout, True)

    return layer_input


def gru(
    input,
    hx,
    params,
    has_biases=True,
    num_layers=1,
    dropout=0.0,
    train=False,
    bidirectional=False,
    batch_first=False,
):
    logger.debug("GEMS_KUNLUNXIN GRU")
    logging.getLogger("flag_gems.ops.gru").debug("GEMS GRU")
    _g._validate_args(input, hx, params, has_biases, num_layers, dropout, bidirectional)

    if batch_first:
        batch_size, seq_len, input_size = input.shape
        input_view = input.transpose(0, 1)
    else:
        seq_len, batch_size, input_size = input.shape
        input_view = input
    if seq_len == 0:
        raise RuntimeError("Expected sequence length to be larger than 0 in RNN")

    hidden_size = hx.shape[2]
    num_directions = 2 if bidirectional else 1

    final_h = _g._empty(
        (num_layers * num_directions, batch_size, hidden_size),
        input.dtype,
        input.device,
    )
    output_tf = _g._empty(
        (seq_len, batch_size, hidden_size * num_directions),
        input.dtype,
        input.device,
    )
    _gru_forward_impl(
        input_view,
        hx,
        params,
        output_tf,
        final_h,
        num_layers,
        num_directions,
        hidden_size,
        input_size,
        batch_size,
        seq_len,
        has_biases,
        train,
        dropout,
    )

    output = output_tf.transpose(0, 1) if batch_first else output_tf
    return output, final_h


def gru_data(
    data,
    batch_sizes,
    hx,
    params,
    has_biases=True,
    num_layers=1,
    dropout=0.0,
    train=False,
    bidirectional=False,
):
    logger.debug("GEMS_KUNLUNXIN GRU_DATA")
    logging.getLogger("flag_gems.ops.gru").debug("GEMS GRU_DATA")
    if data.dim() != 2:
        raise RuntimeError("gru.data: packed data must have 2 dimensions")
    if batch_sizes.dim() != 1:
        raise RuntimeError("gru.data: batch_sizes must be 1-dimensional")
    if num_layers <= 0:
        raise RuntimeError("gru.data: num_layers must be greater than zero")
    if not 0.0 <= dropout <= 1.0:
        raise RuntimeError("gru.data: dropout probability must be between 0 and 1")
    if data.dtype not in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
        raise NotImplementedError(
            "FlagGems gru.data supports float16, bfloat16, float32, and float64"
        )
    num_directions = 2 if bidirectional else 1
    num_states = num_layers * num_directions
    if hx.dim() != 3:
        raise RuntimeError("gru.data: hidden state must have 3 dimensions")
    if hx.shape[0] != num_states:
        raise RuntimeError(
            f"gru.data: expected {num_states} hidden state rows, got {hx.shape[0]}"
        )
    expected_params = num_states * (4 if has_biases else 2)
    if len(params) != expected_params:
        raise RuntimeError(
            f"gru.data: expected {expected_params} parameter tensors, got {len(params)}"
        )
    if hx.device != data.device:
        raise RuntimeError("gru.data: data and hidden state must share a device")
    if hx.dtype != data.dtype:
        raise RuntimeError("gru.data: data and hidden state must share a dtype")

    num_steps = batch_sizes.numel()
    input_size = data.shape[1]
    batch = hx.shape[1]
    hidden_size = hx.shape[2]

    # Host-side batch_sizes / packed-row offsets. The generic ``_unpack_padded_kernel`` /
    # ``_pack_output_kernel`` gather-scatter kernels index ``offsets``/``batch_sizes`` with
    # per-program scalar ``tl.load``s; on this backend that pattern miscompiles into a
    # nondeterministic gather (scattered packed rows read/write the wrong padded row, e.g.
    # a single row off by a large delta that varies run-to-run). The generic gru.data path
    # never exercised these kernels on XPU (its input GEMM fails the LoopGrid pass first),
    # so they were unvalidated here. We replace both with contiguous per-timestep device
    # slice copies (ordered DMA, deterministic): for step t the active rows are the packed
    # slice [off:off+bs_t] <-> padded[t, :bs_t]. num_steps is the max sequence length (small).
    bs_list = batch_sizes.to("cpu", torch.int64).tolist()
    offsets_list = [0] * num_steps
    acc = 0
    for i in range(num_steps):
        offsets_list[i] = acc
        acc += bs_list[i]

    batch_sizes = batch_sizes.to(data.device)
    bs32 = batch_sizes.to(torch.int32)

    x_padded = _g._empty((num_steps, batch, input_size), data.dtype, data.device)
    x_padded.zero_()
    for t in range(num_steps):
        bs_t = bs_list[t]
        if bs_t > 0:
            off = offsets_list[t]
            x_padded[t, :bs_t].copy_(data[off : off + bs_t])

    hidden_total = hidden_size * num_directions
    final_h = _g._empty((num_states, batch, hidden_size), data.dtype, data.device)
    out_padded = _g._empty((num_steps, batch, hidden_total), data.dtype, data.device)
    _gru_forward_impl(
        x_padded,
        hx,
        params,
        out_padded,
        final_h,
        num_layers,
        num_directions,
        hidden_size,
        input_size,
        batch,
        num_steps,
        has_biases,
        train,
        dropout,
        batch_sizes=bs32,
    )

    out_packed = _g._empty((data.shape[0], hidden_total), data.dtype, data.device)
    for t in range(num_steps):
        bs_t = bs_list[t]
        if bs_t > 0:
            off = offsets_list[t]
            out_packed[off : off + bs_t].copy_(out_padded[t, :bs_t])

    return out_packed, final_h


__all__ = ["gru", "gru_data"]
