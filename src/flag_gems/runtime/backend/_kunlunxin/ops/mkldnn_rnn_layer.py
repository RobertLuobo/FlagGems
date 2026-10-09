import logging

import torch

logger = logging.getLogger(__name__)


def _lstm_layer(input, w_ih, w_hh, b_ih, b_hh, hx, cx, reverse, hidden_size, has_biases):
    """One single-layer unidirectional LSTM pass (kunlunxin, mkldnn_rnn_layer mode=2).

    The generic FlagGems Triton kernel cannot compile/launch on XPU (the
    TritonXPUCoreTiling pass fails, same error family as rnn_tanh/lstm). The
    recurrence is folded into primitive vendor ops: vendor fp32 matmuls (fp32
    accumulation) + vendor sigmoid/tanh, with autograd-safe torch.stack for
    assembly. Gate layout follows PyTorch/oneDNN packing i, f, g, o. Autograd
    flows natively through these differentiable ops, so no custom Function is
    needed for the backward pass.
    """
    seq_len, batch_size, _ = input.shape

    x = input.to(torch.float32)
    w_ih_t = w_ih.to(torch.float32).t()
    w_hh_t = w_hh.to(torch.float32).t()
    h = hx.to(torch.float32)
    c = cx.to(torch.float32)

    x2d = x.reshape(seq_len * batch_size, -1)
    pre = x2d.matmul(w_ih_t)
    if has_biases:
        pre = pre + b_ih.to(torch.float32)
    pre = pre.reshape(seq_len, batch_size, 4 * hidden_size)
    if has_biases:
        pre = pre + b_hh.to(torch.float32)

    order = range(seq_len - 1, -1, -1) if reverse else range(seq_len)
    outs = [None] * seq_len
    for t in order:
        gates = pre[t] + h.matmul(w_hh_t)
        i_g = torch.sigmoid(gates[:, 0:hidden_size])
        f_g = torch.sigmoid(gates[:, hidden_size : 2 * hidden_size])
        g_g = torch.tanh(gates[:, 2 * hidden_size : 3 * hidden_size])
        o_g = torch.sigmoid(gates[:, 3 * hidden_size : 4 * hidden_size])
        c = f_g * c + i_g * g_g
        h = o_g * torch.tanh(c)
        outs[t] = h
    output = torch.stack(outs, 0)
    return output, h, c


def mkldnn_rnn_layer(
    input,
    weight0,
    weight1,
    weight2,
    weight3,
    hx_,
    cx_,
    reverse,
    batch_sizes,
    mode,
    hidden_size,
    num_layers,
    has_biases,
    bidirectional,
    batch_first,
    train,
):
    """Single-layer unidirectional LSTM layer (kunlunxin, mkldnn_rnn_layer mode=2).

    Mirrors ``torch.mkldnn_rnn_layer``: ``weight0/weight1`` are the input- and
    hidden-to-hidden weights ``(4H, input)`` / ``(4H, H)`` and ``weight2/weight3``
    the corresponding biases ``(4H,)``. Returns ``(output, hy, cy, workspace)``;
    the oneDNN ``workspace`` is opaque, so an empty placeholder is returned. The
    recurrence is folded into vendor fp32 matmul + vendor sigmoid/tanh because
    the generic Triton kernel fails to compile on XPU. Multi-layer,
    bidirectional, packed (``batch_sizes``), ``batch_first`` and non-LSTM
    ``mode`` all raise ``NotImplementedError``.
    """
    logger.debug("GEMS_KUNLUNXIN MKLDNN_RNN_LAYER")

    if mode != 2:
        raise NotImplementedError(
            "GEMS_KUNLUNXIN MKLDNN_RNN_LAYER only supports LSTM (mode=2)"
        )
    if num_layers != 1 or bidirectional:
        raise NotImplementedError(
            "GEMS_KUNLUNXIN MKLDNN_RNN_LAYER only supports single-layer unidirectional"
        )
    if batch_first:
        raise NotImplementedError(
            "GEMS_KUNLUNXIN MKLDNN_RNN_LAYER only supports batch_first=False (T, N, *) layout"
        )
    if batch_sizes is not None and len(batch_sizes) > 0:
        raise NotImplementedError(
            "GEMS_KUNLUNXIN MKLDNN_RNN_LAYER does not support packed sequences (batch_sizes)"
        )

    del train

    orig_dtype = input.dtype
    output, hy, cy = _lstm_layer(
        input,
        weight0,
        weight1,
        weight2,
        weight3,
        hx_,
        cx_,
        reverse,
        hidden_size,
        has_biases,
    )
    workspace = torch.empty(0, dtype=orig_dtype, device=input.device)
    return output.to(orig_dtype), hy.to(orig_dtype), cy.to(orig_dtype), workspace


__all__ = ["mkldnn_rnn_layer"]


def _patch_generic_wrapper():
    """Route direct ``flag_gems.ops.mkldnn_rnn_layer.mkldnn_rnn_layer`` calls here.

    Keeps the change backend-local: the generic module source is untouched and
    other vendor backends are unaffected (this module is only imported for the
    kunlunxin backend). The generic Triton kernel cannot compile/launch on XPU.
    """
    try:
        import sys

        _generic_module = sys.modules.get("flag_gems.ops.mkldnn_rnn_layer")
        if _generic_module is not None and hasattr(
            _generic_module, "mkldnn_rnn_layer"
        ):
            _generic_module.mkldnn_rnn_layer = mkldnn_rnn_layer
        # Also rebind the name re-exported into the flag_gems.ops package
        # namespace (what ``flag_gems.ops.mkldnn_rnn_layer`` resolves to), used
        # by the benchmark harness and direct package-level references.
        _ops_pkg = sys.modules.get("flag_gems.ops")
        if _ops_pkg is not None and hasattr(_ops_pkg, "mkldnn_rnn_layer"):
            _ops_pkg.mkldnn_rnn_layer = mkldnn_rnn_layer
    except ImportError:
        pass


_patch_generic_wrapper()
