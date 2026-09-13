# Kunlunxin (XPU) override of log2 / log2_.
#
# log2 was NOT overridden by kunlunxin, so it fell to the generic bare
# `@pointwise_dynamic` (no CodeGenConfig) kernel `tl.log2(x.to(fp32))`.
# On XPU this miscompiles for EVERY dtype (baseline: 18/18 fail, e.g.
# res=-0.4897 vs ref=-0.7065 on fp16) — the same class of vectorized-log
# miscompile already documented in the sibling log1p.py, where the open
# vectorization path produces log values off by a constant.
#
# Fix: mirror the proven log1p.py recipe:
#   1. tuned CodeGenConfig with isCloseVectorization=True (vectorization CLOSED)
#      — this is what fixes the log miscompile on XPU;
#   2. compute log2 via the natural-log intrinsic that is known-good under
#      closed vectorization: log2(x) = ln(x) * (1/ln2), instead of the raw
#      tl.log2 builtin. 1/ln2 = log2(e) = 1.4426950408889634.
# log2_ shares the kernel via out0=A.
import logging

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
    isCloseVectorization=True,
    kunlunAutoGrid=True,
    unroll_num=8,
)


@pointwise_dynamic(promotion_methods=[(0, "INT_TO_FLOAT")], config=config_)
@triton.jit
def log2_func(x):
    return (tl.log(x.to(tl.float32)) * 1.4426950408889634).to(x.dtype)


def log2(A):
    logger.debug("GEMS_KUNLUNXIN LOG2")
    return log2_func(A)


def log2_(A):
    logger.debug("GEMS_KUNLUNXIN LOG2_")
    # ATen in-place log2_ is only valid for floating-point tensors; native
    # torch (CPU: "result type Float can't be cast to the desired output type
    # Long"; XPU/xdnn: [NOT IMPLEMENTED]) raises for integral inputs, while
    # writing back the float log2 into an int tensor would silently truncate.
    if not A.is_floating_point():
        raise TypeError(f"log2_ does not support dtype {A.dtype}")
    log2_func(A, out0=A)
    return A
