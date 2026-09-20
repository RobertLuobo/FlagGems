import logging

import torch
import triton
import triton.language as tl
from _kunlunxin.utils.codegen_config_utils import CodeGenConfig

from ..utils.pointwise_dynamic import pointwise_dynamic

logger = logging.getLogger(__name__)

config_ = CodeGenConfig(
    512,
    (65536, 65536, 65536),
    32,
    True,
    prefer_1d_tile=True,
    buffer_size_limit=4096,
    isCloseVectorization=False,
    # kunlunAutoGrid=True lets the 1d-tile wrapper pick num_ctas=1 when
    # num_tasks <= 2048*64 instead of the fixed 12-CTA launch.  At (64,64)
    # (4096 elems) the fixed grid spreads the work over 12 CTAs and the
    # per-CTA launch/cluster overhead dominates the end-to-end latency:
    # 4096-elem leaky_relu latency 7.6 -> 5.6us (bf16) / 8.0 -> 5.2us (fp16)
    # / 6.6 -> 5.3us (fp32) when combined with the pre-allocated out0 in
    # leaky_relu() below (harness/solution/rolloutB/ab_arms.txt); the 5-round
    # robust dtype-balanced speedup goes 0.7565 -> 0.9216.  Shapes above the
    # 2048*64 threshold keep the 12-CTA grid, so the large shapes are
    # unaffected.  Same knob and rationale as clamp.py / neg.py / rsub.py.
    kunlunAutoGrid=True,
    unroll_num=8,
)


@pointwise_dynamic(
    is_tensor=[True, False], promotion_methods=[(0, "DEFAULT")], config=config_
)
@triton.jit
def leaky_relu_kernel(x, negative_slope):
    x_fp32 = x.to(tl.float32)
    return tl.maximum(x_fp32, 0.0) + negative_slope * tl.minimum(x_fp32, 0.0)


def leaky_relu(A, negative_slope=0.01):
    logger.debug("GEMS_KUNLUNXIN LEAKY_RELU")
    # Allocate the result here and pass it as `out0`.  Without it
    # pointwise_dynamic must re-derive the result dtype on every call
    # (`type_promotion(A, negative_slope, DEFAULT)` + a fresh
    # `torch.empty_like`), and at launch-bound shapes that host-side work is a
    # large share of the end-to-end latency.  The pre-fix 5-round harness
    # median for (64,64) was 9.60/8.74/10.67us (bf16/fp16/fp32); with out0 and
    # the kunlunAutoGrid knob above it reaches torch parity (0.9838/1.0024/
    # 1.0826 robust speedup, ab_arms.txt + agg_after_leaky_relu.txt).  The
    # (4096,4096)/(64,512,512) cases are device bound and unaffected.  Same
    # pattern as clamp_max()/rsub_scalar().
    #
    # Pre-allocating is only valid while the promoted result dtype is provably
    # A.dtype: `promotion_methods=[(0, "DEFAULT")]` promotes a floating-point
    # tensor with a Python numeric scalar, and a Python scalar is "weak" so it
    # never widens the tensor (and leaky_relu is documented to preserve the
    # input dtype anyway).  Integer/bool tensors or a non-numeric scalar keep
    # the generic promotion path.
    if A.is_floating_point() and type(negative_slope) in (int, float):
        return leaky_relu_kernel(A, negative_slope, out0=torch.empty_like(A))
    return leaky_relu_kernel(A, negative_slope)


def leaky_relu_(A, negative_slope=0.01):
    logger.debug("GEMS_KUNLUNXIN LEAKY_RELU_")
    return leaky_relu_kernel(A, negative_slope, out0=A)


def leaky_relu_out(A, negative_slope=0.01, *, out=None):
    logger.debug("GEMS_KUNLUNXIN LEAKY_RELU_OUT")
    if out is None:
        return leaky_relu_kernel(A, negative_slope)
    return leaky_relu_kernel(A, negative_slope, out0=out)


_LEAKY_BACKWARD_DTYPES = (torch.float16, torch.float32, torch.bfloat16)


@pointwise_dynamic(
    is_tensor=[True, True, False], promotion_methods=[(0, "DEFAULT")], config=config_
)
@triton.jit
def leaky_relu_backward_kernel(g, x, negative_slope):
    step = tl.minimum(tl.maximum(x * 1.0e30, 0.0), 1.0)
    return g * (negative_slope + (1.0 - negative_slope) * step)


@pointwise_dynamic(
    is_tensor=[True, True, False], promotion_methods=[(0, "DEFAULT")], config=config_
)
@triton.jit
def leaky_relu_backward_general_kernel(g, x, negative_slope):
    x_fp32 = x.to(tl.float32)
    g_fp32 = g.to(tl.float32)
    return tl.where(x_fp32 > 0.0, g_fp32, g_fp32 * negative_slope)


def leaky_relu_backward(grad_output, self, negative_slope=0.01, self_is_result=False):
    logger.debug("GEMS_KUNLUNXIN LEAKY_RELU_BACKWARD")
    if grad_output.numel() == 0:
        return torch.empty_like(self)
    if grad_output.dtype in _LEAKY_BACKWARD_DTYPES and (
        grad_output.is_contiguous() and self.is_contiguous()
    ):
        return leaky_relu_backward_kernel(grad_output, self, negative_slope)
    return leaky_relu_backward_general_kernel(grad_output, self, negative_slope)
