import logging

import torch

from .bmm import bmm
from .mm import mm

logger = logging.getLogger(__name__)


class _GemsMM(torch.autograd.Function):
    """Differentiable 2-D matmul backed by the Kunlunxin (XPU) ``mm`` kernel.

    The backward products are themselves expressed through ``_GemsMM.apply`` so
    higher-order gradients stay on the XPU ``mm`` kernel (no native
    ``torch.matmul`` / ``torch.mm`` fallback).  Recursion is bounded by the
    differentiation order, not unbounded, because the gradients are computed
    directly rather than by re-differentiating a recomputed forward.
    """

    @staticmethod
    def forward(ctx, a, b):
        ctx.save_for_backward(a, b)
        return mm(a, b)

    @staticmethod
    def backward(ctx, grad_out):
        a, b = ctx.saved_tensors
        grad_a = _GemsMM.apply(grad_out, b.transpose(-2, -1))
        grad_b = _GemsMM.apply(a.transpose(-2, -1), grad_out)
        return grad_a, grad_b


class _GemsBMM(torch.autograd.Function):
    """Differentiable batched matmul backed by the Kunlunxin ``bmm`` kernel."""

    @staticmethod
    def forward(ctx, a, b):
        ctx.save_for_backward(a, b)
        return bmm(a, b)

    @staticmethod
    def backward(ctx, grad_out):
        a, b = ctx.saved_tensors
        grad_a = _GemsBMM.apply(grad_out, b.transpose(-2, -1))
        grad_b = _GemsBMM.apply(a.transpose(-2, -1), grad_out)
        return grad_a, grad_b


def _matmul(input, other):
    logger.debug("GEMS_KUNLUNXIN LINALG_MATMUL")

    # Fold 1D inputs into 2D per torch.matmul semantics:
    # (K,) @ (K, N) -> (N,), (M, K) @ (K,) -> (M,), (K,) @ (K,) -> scalar
    fold_first = input.dim() == 1
    fold_last = other.dim() == 1
    if fold_first:
        input = input.unsqueeze(0)
    if fold_last:
        other = other.unsqueeze(1)

    # Handle 2D case: (M, K) @ (K, N) -> (M, N)
    if input.dim() == 2 and other.dim() == 2:
        M, K = input.shape
        _, N = other.shape
        if M == 0 or N == 0 or K == 0:
            # Zero-sized matmul is well defined (empty output, or all-zero when
            # only the contracted dim is empty); allocate the result directly
            # instead of launching a kernel over an empty grid.
            out = torch.zeros((M, N), dtype=input.dtype, device=input.device)
        else:
            out = _GemsMM.apply(input, other)
        if fold_first and fold_last:
            return out.reshape(())
        if fold_first:
            return out[0]
        if fold_last:
            return out[:, 0]
        return out

    # Handle batched cases: broadcast the batch dims, then run a single bmm
    # over the flattened batch.  expand produces stride-0 views; the autograd
    # graph reduces the broadcast gradients automatically.
    assert input.dim() >= 2 and other.dim() >= 2, "incompatible dimensions"
    bshape = torch.broadcast_shapes(input.shape[:-2], other.shape[:-2])
    M, K = input.shape[-2], input.shape[-1]
    N = other.shape[-1]
    assert K == other.shape[-2], "incompatible dimensions"
    a = input.expand(bshape + (M, K))
    b = other.expand(bshape + (K, N))
    nbatch = 1
    for s in bshape:
        nbatch *= s
    if nbatch == 0:
        # empty batch: no kernel launch (grid of 0 is invalid), just an empty
        # output with the folded 1-D dims restored
        out = torch.empty(bshape + (M, N), dtype=input.dtype, device=input.device)
        if fold_first:
            out = out.squeeze(-2)
        if fold_last:
            out = out.squeeze(-1)
        return out
    out = _GemsBMM.apply(a.reshape(nbatch, M, K), b.reshape(nbatch, K, N))
    out = out.reshape(bshape + (M, N))
    if fold_first:
        out = out.squeeze(-2)
    if fold_last:
        out = out.squeeze(-1)
    return out


def linalg_matmul(input, other):
    """Matrix product of two tensors (alias for torch.linalg.matmul).

    Routes through the Kunlunxin (XPU) ``mm`` / ``bmm`` Triton kernels instead
    of the generic ``flag_gems.ops.mm`` path, whose stream-k and general-mm
    autotune configs do not lower on the XPU backend.
    """
    # matmul is only defined for floating point dtypes; reject integral and
    # complex inputs with errors matching the ATen CUDA backend instead of
    # letting them fail inside the Triton kernel.
    if not input.is_floating_point():
        raise NotImplementedError(
            f"linalg_matmul not implemented for '{input.dtype}' on this device"
        )
    if not other.is_floating_point():
        raise NotImplementedError(
            f"linalg_matmul not implemented for '{other.dtype}' on this device"
        )
    if input.dtype != other.dtype:
        raise RuntimeError(
            f"expected mat1 and mat2 to have the same dtype, but got: "
            f"{input.dtype} != {other.dtype}"
        )
    return _matmul(input, other)
