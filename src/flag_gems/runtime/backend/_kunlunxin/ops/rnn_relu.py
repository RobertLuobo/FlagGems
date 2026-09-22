import logging
import sys

import torch
import triton
import triton.language as tl

from .mm import mm as _gems_mm

logger = logging.getLogger(__name__)


@triton.jit
def _rnd_dt(
    v,
    SHIFT: tl.constexpr,
    BIAS: tl.constexpr,
    AND1: tl.constexpr,
    MASK: tl.constexpr,
): 
    u = v.to(tl.int32, bitcast=True)
    r = u + BIAS + ((u >> SHIFT) & AND1)
    return (r & MASK).to(tl.float32, bitcast=True)


@triton.jit
def _rnn_relu_step_kernel(
    h_ptr,
    w_hh_t_ptr,
    a_ptr,
    b_hh_ptr,
    h_out_ptr,
    out_step_ptr,
    H_PAD: tl.constexpr,
    BLOCK_H: tl.constexpr,
    SHIFT: tl.constexpr,
    BIAS: tl.constexpr,
    AND1: tl.constexpr,
    MASK: tl.constexpr,
):
    b = tl.program_id(0)
    dt = out_step_ptr.dtype.element_ty
    for hb in range(H_PAD // BLOCK_H):
        o_offs = hb * BLOCK_H + tl.arange(0, BLOCK_H)
        acc = tl.zeros([BLOCK_H], dtype=tl.float32)
        for kb in range(H_PAD // BLOCK_H):
            k_offs = kb * BLOCK_H + tl.arange(0, BLOCK_H)
            h_vec = tl.load(h_ptr + b * H_PAD + k_offs)
            w_tile = tl.load(w_hh_t_ptr + o_offs[:, None] * H_PAD + k_offs[None, :])
            acc += tl.sum(w_tile * h_vec.to(tl.float32)[None, :], axis=1)
        bhh = tl.load(b_hh_ptr + o_offs).to(tl.float32)
        av = tl.load(a_ptr + b * H_PAD + o_offs).to(tl.float32)
        c = _rnd_dt(acc, SHIFT, BIAS, AND1, MASK)
        cb = _rnd_dt(c + bhh, SHIFT, BIAS, AND1, MASK)
        pre = _rnd_dt(av + cb, SHIFT, BIAS, AND1, MASK)
        h_new = tl.where(pre > 0, pre, 0.0)
        tl.store(h_out_ptr + b * H_PAD + o_offs, h_new.to(dt))
        tl.store(out_step_ptr + b * H_PAD + o_offs, h_new.to(dt))


def rnn_relu(
    input,
    hx=None,
    params=None,
    has_biases=True,
    num_layers=1,
    dropout=0.0,
    train=False,
    bidirectional=False,
    batch_first=False,
):
    logger.debug("GEMS_KUNLUNXIN RNN_RELU")

    if params is None:
        raise ValueError("params must be provided")
    if hx is None:
        raise ValueError("hx must be provided to match torch.rnn_relu schema")
    if not (num_layers == 1 and not bidirectional and dropout == 0):
        raise NotImplementedError(
            "GEMS RNN_RELU only supports single-layer unidirectional without dropout"
        )

    w_ih = params[0]
    w_hh = params[1]
    if has_biases:
        b_ih = params[2]
        b_hh = params[3]
    else:
        b_ih = None
        b_hh = None

    x = input.transpose(0, 1).contiguous() if batch_first else input
    seq_len, batch_size, input_size = x.shape
    hidden_size = w_hh.shape[0]
    hx2d = hx.reshape(batch_size, hidden_size)

    x2d = x.reshape(seq_len * batch_size, input_size)

    # Parameters are nn.Parameter objects (requires_grad=True by default) even
    # in inference; only input/hx gradients actually require the autograd-safe
    # native chain. train=True also routes to the native chain (gradients are
    # requested for the weight parameters).
    need_autograd = (
        train or input.requires_grad or (hx is not None and hx.requires_grad)
    )

    # Fused path is gated to pow2 hidden_size <= 128: taller tiles (256/512)
    # exhaust uni_sram during XPU kernel compilation (OOM at compile time).
    if (
        not need_autograd
        and hidden_size <= 128
        and ((hidden_size & (hidden_size - 1)) == 0)
    ):
        w_ih32 = w_ih.to(torch.float32)
        # Input projection through the Gems Triton GEMM (not torch.mm, which
        # would dispatch to the native XDNN fallback). mm() accepts the strided
        # ``.t()`` view directly (see its own note) and accumulates in fp32.
        eih = _gems_mm(x2d.to(torch.float32), w_ih32.t()).to(x.dtype)
        a_all = (eih + b_ih) if b_ih is not None else eih
        a_all = a_all.reshape(seq_len, batch_size, hidden_size)
        if b_hh is None:
            b_hh = torch.zeros((hidden_size,), dtype=x.dtype, device=x.device)
        # The kernel does the two adds in the element dtype (see its docstring:
        # an explicit ``.to(dt).to(fp32)`` round-trip is folded away and does
        # not round), so ``h``, ``a`` and ``b_hh`` are handed over in ``x.dtype``
        # exactly as native stores them, and only the weight is upcast to fp32.
        w_hh32 = w_hh.to(torch.float32)
        hp = hidden_size
        blk = hp
        # (SHIFT, BIAS, AND1, MASK) for _rnd_dt: fp32 is the identity, bf16 drops
        # 16 mantissa bits, fp16 drops 13.  See _rnd_dt for why an explicit
        # `.to(dt).to(fp32)` round-trip is not usable here.
        rnd = {
            torch.float32: (16, 0, 0, -1),
            torch.bfloat16: (16, 0x7FFF, 1, -65536),
            torch.float16: (13, 0x0FFF, 1, -8192),
        }[x.dtype]
        h_buf = torch.zeros((batch_size, hp), dtype=x.dtype, device=x.device)
        h_in = torch.zeros((batch_size, hp), dtype=x.dtype, device=x.device)
        h_in[:, :hidden_size] = hx2d
        out_buf = torch.zeros((seq_len, batch_size, hp), dtype=x.dtype, device=x.device)
        for t in range(seq_len):
            _rnn_relu_step_kernel[(batch_size,)](
                h_in,
                w_hh32,
                a_all[t],
                b_hh,
                h_buf,
                out_buf[t],
                H_PAD=hp,
                BLOCK_H=blk,
                SHIFT=rnd[0],
                BIAS=rnd[1],
                AND1=rnd[2],
                MASK=rnd[3],
            )
            h_in, h_buf = h_buf, h_in
        output = out_buf[..., :hidden_size]
        h = h_in[..., :hidden_size]
    else:
        w_hh_t = w_hh.t().contiguous()
        try:
            pre = (
                (x2d.matmul(w_ih.t()) + b_ih)
                if b_ih is not None
                else x2d.matmul(w_ih.t())
            )
            pre = pre.reshape(seq_len, batch_size, hidden_size)
            if b_hh is not None:
                pre = pre + b_hh
            h = hx2d
            outputs = []
            for t in range(seq_len):
                h = torch.relu(torch.mm(h, w_hh_t) + pre[t])
                outputs.append(h)
            output = torch.stack(outputs, 0)
        except ZeroDivisionError:
            # per-step small-shape recurrence, same math, crash-free
            h = hx2d
            outputs = []
            for t in range(seq_len):
                ih_t = (
                    x[t].matmul(w_ih.t()) + b_ih
                    if b_ih is not None
                    else x[t].matmul(w_ih.t())
                )
                hh_t = (
                    h.matmul(w_hh.t()) + b_hh
                    if b_hh is not None
                    else h.matmul(w_hh.t())
                )
                h = torch.relu(ih_t.to(torch.float32) + hh_t.to(torch.float32)).to(
                    x.dtype
                )
                outputs.append(h)
            output = torch.stack(outputs, 0)

    if batch_first:
        output = output.transpose(0, 1).contiguous()

    return output, h.unsqueeze(0)


__all__ = ["rnn_relu"]


_REPATCH_MARKER = "_flag_gems_rnn_relu_repatch"
_SYS_FINDER_ATTR = "_flag_gems_rnn_relu_repatch_finder"


class _RnnReluRepatchLoader:
    """Delegating loader that re-applies the backend override once the generic
    ``flag_gems.ops.rnn_relu`` module has finished executing."""

    _flag_gems_rnn_relu_repatch = _REPATCH_MARKER

    def __init__(self, inner):
        self._inner = inner

    def __getattr__(self, name):
        return getattr(self._inner, name)

    def create_module(self, spec):
        return self._inner.create_module(spec)

    def exec_module(self, module):
        self._inner.exec_module(module)
        module.rnn_relu = rnn_relu


class _RnnReluRepatchFinder:
    _flag_gems_rnn_relu_repatch = _REPATCH_MARKER
    TARGET = "flag_gems.ops.rnn_relu"

    def find_spec(self, fullname, path=None, target=None):
        if fullname != self.TARGET:
            return None
        for finder in sys.meta_path:
            if getattr(finder, _REPATCH_MARKER, None) is not None:
                continue
            try:
                spec = finder.find_spec(fullname, path, target)
            except Exception:
                continue
            if spec is not None and spec.loader is not None:
                if getattr(spec.loader, _REPATCH_MARKER, None) is None:
                    spec.loader = _RnnReluRepatchLoader(spec.loader)
                return spec
        return None


def _patch_generic_wrapper():
    try:
        import importlib

        _generic_module = sys.modules.get("flag_gems.ops.rnn_relu")
        if _generic_module is None:
            _generic_module = importlib.import_module("flag_gems.ops.rnn_relu")
        if hasattr(_generic_module, "rnn_relu"):
            _generic_module.rnn_relu = rnn_relu
    except ImportError:
        pass

    if getattr(sys, _SYS_FINDER_ATTR, None) is None:
        try:
            setattr(sys, _SYS_FINDER_ATTR, _RnnReluRepatchFinder())
            sys.meta_path.insert(0, getattr(sys, _SYS_FINDER_ATTR))
        except Exception:
            pass


_patch_generic_wrapper()
