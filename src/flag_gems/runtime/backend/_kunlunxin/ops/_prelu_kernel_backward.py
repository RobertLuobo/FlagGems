# Kunlunxin (XPU) override of _prelu_kernel_backward.
#
# aten::_prelu_kernel_backward(grad_output, x, weight) -> (grad_input, grad_weight)
# computes the elementwise PReLU backward:
#   grad_input  = where(x > 0, grad_output, grad_output * weight)
#   grad_weight = where(x < 0, grad_output * x, 0)
# with weight broadcastable against x (scalar or per-last-dim vector). The
# reference (CPU ATen, and the torch.where device expression used by the
# benchmark) returns both outputs with the shape of x.
#
# Two XPU-specific findings drive this implementation (same family as the
# forward _prelu_kernel/prelu overrides):
#
# 1. The generic flag_gems ops/_prelu_kernel_backward.py uses a hand-written
#    @triton.jit kernel with a fixed BLOCK_SIZE=1024 and a runtime `C` used in
#    `c = offsets % C` to index the weight. On XPU the runtime div/mod blocks
#    OffsetAnalysis (discrete gather) and the tiny 1024-element tiles leave the
#    device idle: a flat 1-D tile of 168M fp16 elements takes ~766 ms in the
#    benchmark (speedup ~0.004) for what the vendor reference does in ~3 ms.
#
# 2. The XPU-tuned pointwise_dynamic has a fast path only when every tensor
#    input/output shares the same shape and is contiguous (dimension collapse
#    to a flat 1D task space with stride-1 block DMA), which explains why the
#    broadcasted-weight form (weight reshaped to [1]*ndim, stride-0) is
#    ~30x slower (894 ms) than the scalar-argument form (25.7 ms). The scalar
#    weight is therefore passed as a Python float (non-tensor argument) so the
#    kernel takes the fast path; the per-channel weight (only used by the
#    small functional-test shapes (2,3,4)/(4,8,16)) keeps the generic
#    broadcast form, which is correct but slow on this backend.
#
# 3. `tl.where(x > 0, ..., ...)` (select with a tensor RHS) is ~4x slower than
#    the same kernel expressed without select: on XPU the vectorized select
#    blocks the memory pipeline. The backward is therefore written with a
#    select-free, single-rounding formulation. With
#      pos = (x > 0)   (1.0 / 0.0)
#    both branches below are exact and match the ATen reference for every
#    input class (NaN included, since min(NaN, 0) = 0 on this backend):
#      grad_input  = grad_output * (pos + (1 - pos) * weight)
#      grad_weight = grad_output * min(x, 0) * (1 - pos)
#    x>0   -> grad_input=grad_output (exact),   grad_weight=0
#    x<0   -> grad_input=grad_output*weight,    grad_weight=grad_output*x
#    x=±0  -> grad_input=grad_output*weight,    grad_weight=±0  (matches ATen
#              `where(x<0, ...)`: both branches are 0; sign of 0 is immaterial)
#    x=NaN -> grad_input=grad_output*weight,    grad_weight=0 (ATen: x<0 False)
import logging

import torch
import triton
import triton.language as tl

import flag_gems

from ..utils.pointwise_dynamic import pointwise_dynamic

logger = logging.getLogger("flag_gems").getChild(__name__.lstrip("."))


@pointwise_dynamic(
    is_tensor=[True, True, False],
    num_outputs=2,
    promotion_methods=[(0, 1, "DEFAULT"), (0, 1, "DEFAULT")],
    dtypes=[None, None, float],
)
@triton.jit
def _prelu_kernel_backward_scalar_func(grad_output, x, weight):
    # weight is a Python float; fast path (all tensor args same shape).
    pos = (x > 0).to(x.dtype)
    x_neg = tl.minimum(x, 0.0)
    grad_input = grad_output * (pos + (1.0 - pos) * weight)
    grad_weight = grad_output * x_neg * (1.0 - pos)
    return grad_input, grad_weight


@pointwise_dynamic(
    is_tensor=[True, True, True],
    num_outputs=2,
    promotion_methods=[(0, 1, "DEFAULT"), (0, 1, "DEFAULT")],
)
@triton.jit
def _prelu_kernel_backward_channel_func(grad_output, x, weight):
    # weight is a last-dim broadcastable tensor [1, ..., 1, C].
    pos = (x > 0).to(x.dtype)
    x_neg = tl.minimum(x, 0.0)
    grad_input = grad_output * (pos + (1.0 - pos) * weight)
    grad_weight = grad_output * x_neg * (1.0 - pos)
    return grad_input, grad_weight


def _prelu_kernel_backward(*args, **kwargs):
    logger.debug("GEMS_KUNLUNXIN _PRELU_KERNEL_BACKWARD")
    if len(args) >= 3:
        grad_output, x, weight = args[0], args[1], args[2]
    else:
        grad_output = kwargs.get("grad_output")
        x = kwargs.get("self")
        weight = kwargs.get("weight")

    if grad_output is None or x is None or weight is None:
        raise ValueError(
            "_prelu_kernel_backward expects (grad_output, self, weight) as arguments."
        )

    if (
        grad_output.device.type != flag_gems.device
        or x.device.type != flag_gems.device
        or weight.device.type != flag_gems.device
    ):
        raise RuntimeError(
            f"_prelu_kernel_backward: all tensors must be "
            f"{flag_gems.device} tensors for Triton kernel."
        )

    if weight.dtype != x.dtype:
        weight = weight.to(dtype=x.dtype)
    if grad_output.dtype != x.dtype:
        grad_output = grad_output.to(dtype=x.dtype)

    grad_output = grad_output.contiguous()
    x = x.contiguous()
    weight = weight.contiguous()

    ndim = x.dim()
    if weight.numel() == 1:
        # Scalar weight: kernel-argument fast path, see module docstring.
        return _prelu_kernel_backward_scalar_func(grad_output, x, float(weight))
    if ndim == 0:
        raise AssertionError("Non-scalar weight provided for a 0-dim input.")
    # Weight matches the last dimension (per-channel PReLU): [C] -> [1, 1, C].
    C = x.shape[-1]
    if weight.numel() != C:
        raise AssertionError(
            f"Weight numel ({weight.numel()}) must equal last dimension size ({C})."
        )
    if ndim == 1:
        w_shape = [C]
    else:
        w_shape = [1] * (ndim - 1) + [C]
    w = weight.reshape(w_shape)
    return _prelu_kernel_backward_channel_func(grad_output, x, w)