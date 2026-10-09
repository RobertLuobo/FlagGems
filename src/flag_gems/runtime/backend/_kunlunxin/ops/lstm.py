import logging

import torch

logger = logging.getLogger(__name__)


def _one_direction(x, h0, c0, w_ih, w_hh, b_ih, b_hh, reverse):
    """One unidirectional LSTM pass over a sequence (kunlunxin).

    All matmuls use the vendor fp32 matmul (no-tf32, fp32 accumulation) and the
    sigmoid/tanh nonlinearities use the vendor elementwise ops; the recurrence
    never re-enters the generic FlagGems Triton LSTM kernel, which fails to
    launch on XPU (runtime error 719 from the KernelGen fused kernel). The gate
    layout follows PyTorch's flat-weight order (input, forget, cell, output).
    torch.stack keeps the assembly autograd-safe. ``x``/``h0``/``c0`` are
    already fp32.
    """
    seq_len, batch, _ = x.shape
    hidden = w_hh.shape[1]

    w_ih_t = w_ih.t()
    w_hh_t = w_hh.t()

    x2d = x.reshape(seq_len * batch, -1)
    pre = x2d.matmul(w_ih_t)
    if b_ih is not None:
        pre = pre + b_ih
    pre = pre.reshape(seq_len, batch, 4 * hidden)
    if b_hh is not None:
        pre = pre + b_hh

    h = h0
    c = c0
    order = range(seq_len - 1, -1, -1) if reverse else range(seq_len)
    outs = [None] * seq_len
    for t in order:
        gates = pre[t] + h.matmul(w_hh_t)
        i = torch.sigmoid(gates[:, 0:hidden])
        f = torch.sigmoid(gates[:, hidden : 2 * hidden])
        g = torch.tanh(gates[:, 2 * hidden : 3 * hidden])
        o = torch.sigmoid(gates[:, 3 * hidden : 4 * hidden])
        c = f * c + i * g
        h = o * torch.tanh(c)
        outs[t] = h
    output = torch.stack(outs, 0)
    return output, h, c


def lstm(
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
    """Multi-layer (optionally bidirectional) LSTM (kunlunxin).

    Matches the tensor-input overload of ``torch.lstm``. The generic FlagGems
    Triton LSTM kernel cannot launch on XPU (vendor runtime error 719), so the
    recurrence is folded into primitive vendor ops: vendor fp32 matmuls
    (no-tf32, fp32 accumulation) + vendor sigmoid/tanh + autograd-safe
    torch.stack / torch.cat for direction/layer assembly. fp32 accumulation
    keeps low-precision dtypes within the test-declared atols of the native
    reference. ``hx`` is the ``(h0, c0)`` tuple; returns ``(output, h_n, c_n)``.
    """
    logger.debug("GEMS_KUNLUNXIN LSTM")

    if params is None:
        raise ValueError("params must be provided")
    if not isinstance(hx, (tuple, list)) or len(hx) != 2:
        raise ValueError("hx must be the (h0, c0) tuple to match torch.lstm schema")
    if dropout not in (0.0, 1.0) and train and num_layers > 1:
        raise NotImplementedError(
            "GEMS_KUNLUNXIN LSTM only supports dropout in {0.0, 1.0} during training"
        )

    directions = 2 if bidirectional else 1
    params_per_state = 4 if has_biases else 2

    h0_all, c0_all = hx
    orig_dtype = input.dtype
    x = input.transpose(0, 1).contiguous() if batch_first else input
    seq_len, batch_size, _ = x.shape
    hidden_size = h0_all.shape[2]

    layer_input = x.to(torch.float32)
    h0_f = h0_all.to(torch.float32)
    c0_f = c0_all.to(torch.float32)

    final_h = []
    final_c = []
    for layer in range(num_layers):
        dir_outputs = []
        for d in range(directions):
            base = (layer * directions + d) * params_per_state
            w_ih = params[base].to(torch.float32)
            w_hh = params[base + 1].to(torch.float32)
            if has_biases:
                b_ih = params[base + 2].to(torch.float32)
                b_hh = params[base + 3].to(torch.float32)
            else:
                b_ih = None
                b_hh = None
            state_idx = layer * directions + d
            h0 = h0_f[state_idx]
            c0 = c0_f[state_idx]
            out, h_last, c_last = _one_direction(
                layer_input, h0, c0, w_ih, w_hh, b_ih, b_hh, reverse=(d == 1)
            )
            dir_outputs.append(out)
            final_h.append(h_last)
            final_c.append(c_last)

        if directions == 2:
            layer_output = torch.cat(dir_outputs, dim=2)
        else:
            layer_output = dir_outputs[0]

        # Inter-layer dropout (applied to every layer output except the last).
        # Only the deterministic p==1.0 (drop everything -> zeros) case is
        # exercised by the suite and matches native bit-exactly.
        if train and dropout >= 1.0 and layer < num_layers - 1:
            layer_output = torch.zeros_like(layer_output)

        layer_input = layer_output

    output = layer_input.to(orig_dtype)
    h_n = torch.stack(final_h, 0).to(orig_dtype)
    c_n = torch.stack(final_c, 0).to(orig_dtype)

    if batch_first:
        output = output.transpose(0, 1).contiguous()

    return output, h_n, c_n


__all__ = ["lstm"]


def _patch_generic_wrapper():
    """Route direct ``flag_gems.ops.lstm.lstm`` calls to this override.

    Keeps the change backend-local: the generic module source is untouched and
    other vendor backends are unaffected (this module is only imported for the
    kunlunxin backend). The generic Triton kernel cannot compile/launch on XPU.
    """
    try:
        import sys

        _generic_module = sys.modules.get("flag_gems.ops.lstm")
        if _generic_module is not None and hasattr(_generic_module, "lstm"):
            _generic_module.lstm = lstm
    except ImportError:
        pass


_patch_generic_wrapper()
