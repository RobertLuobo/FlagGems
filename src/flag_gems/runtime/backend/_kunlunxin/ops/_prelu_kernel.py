# Kunlunxin (XPU) override of _prelu_kernel.
#
# aten::_prelu_kernel(x, weight) computes the elementwise
#   out = where(x >= 0, x, weight * x)
# with weight broadcastable against x. The test/benchmark path always feeds a
# weight of the same shape/stride as x, which takes the pointwise_dynamic fast
# path (dimension collapse to a flat 1D task space).
#
# Two XPU-specific findings drive this implementation:
#
# 1. The generic flag_gems pointwise_dynamic partitions the flat task space
#    with `num_ctas = min(65536, num_tiles)` and 512-element tiles, i.e. tens
#    of thousands of tiny programs on XPU (launch-bound; measured ~2.7 GB/s,
#    70ms for 16M fp16 elements vs torch 0.06ms). The XPU codegen
#    (pointwise_dynamic in this folder) partitions into a fixed 12-CTA grid
#    with one large vectorized tile per CTA, which runs at block-DMA rates.
#
# 2. `tl.where(x >= 0, x, w * x)` (select with a tensor RHS) is ~4x slower
#    than the same kernel expressed without select: on XPU the vectorized
#    select blocks the memory pipeline (0.278ms vs clamp/relu-class 0.05ms on
#    16M fp16). Clamp/leaky-relu show min/max + mul/add formulations run at
#    full bandwidth, so prelu is rewritten as the algebraically identical,
#    select-free, single-rounding expression
#      p       = min(x, 0)          # x<0 -> x, else exact 0
#      out     = (x - p) + p * w   # x>=0 -> x (exact), x<0 -> w*x (exact)
#    For finite inputs this is bit-identical to the ATen formula; NaN inputs
#    still produce NaN (NaN - min(NaN,0) = NaN). Note the select version also
#    evaluates `w * x` for NaN weight with x>=0, which the select-free form
#    can not reproduce (same limitation class as the min/max-based relu/clamp
#    family on this backend).
import logging

import torch
import triton
import triton.language as tl
from _kunlunxin.utils.codegen_config_utils import CodeGenConfig

from ..utils.pointwise_dynamic import pointwise_dynamic

logger = logging.getLogger("flag_gems").getChild(__name__.lstrip("."))

# Same tuned recipe as clamp_max/leaky_relu (pure memory-bound elementwise),
# with kunlunAutoGrid=True so the 1d-tile wrapper launches num_ctas=1 instead
# of the fixed 12 CTAs when num_tasks <= 2048*64.  At (64,64) (4096 elems) the
# fixed grid's per-CTA launch overhead dominates: the pre-fix 5-round harness
# median was 13.10/13.73/13.83us (bf16/fp16/fp32); with the pre-allocated out0
# in _prelu_kernel() below it reaches torch parity (0.9524/0.9729/1.0288 robust
# speedup, ab_arms.txt + agg_after_prelu_kernel.txt), and the 5-round robust
# dtype-balanced speedup goes 0.7175 -> 0.9177.  Shapes above the 2048*64
# threshold keep the 12-CTA grid (large shapes unchanged).  Everything else is
# left at the previous default-config values (no buffer_size_limit / unroll_num
# tuning) so this stays a single-variable change.
_prelu_kernel_config = CodeGenConfig(
    512,
    (65536, 65536, 65536),
    32,
    True,
    prefer_1d_tile=True,
    kunlunAutoGrid=True,
)


@pointwise_dynamic(
    is_tensor=[True, True],
    promotion_methods=[(0, 1, "DEFAULT")],
    config=_prelu_kernel_config,
)
@triton.jit
def _prelu_kernel_func(x, weight):
    x_neg = tl.minimum(x, 0.0)
    return (x - x_neg) + x_neg * weight


def _prelu_kernel(A, B):
    logger.debug("GEMS_KUNLUNXIN _PRELU_KERNEL")
    # Pre-allocate the result as `out0` so pointwise_dynamic can skip the
    # per-call result-dtype promotion (`type_promotion(A, B, DEFAULT)`) and the
    # `torch.empty_like`.  That host work is a large share of the latency at
    # launch-bound shapes: (64,64) 13.10 -> 5.78us bf16 / 13.73 -> 5.39us fp16
    # / 13.83 -> 5.39us fp32 in the 5-round harness (agg_before_prelu_kernel.txt
    # / agg_after_prelu_kernel.txt).
    #
    # `weight` is only required to be *broadcastable* against A (aten
    # semantics), so `torch.empty_like(A)` is a valid result buffer only when
    # the promoted dtype and the task shape are provably identical to A: both
    # tensors floating-point, same dtype, and exactly the same shape.  Every
    # other case (broadcast weight, mixed dtype, integer input) keeps the
    # generic allocation path.
    if A.is_floating_point() and A.dtype == B.dtype and A.shape == B.shape:
        return _prelu_kernel_func(A, B, out0=torch.empty_like(A))
    return _prelu_kernel_func(A, B)
