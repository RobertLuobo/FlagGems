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
def log10_func(x):
    return tl.log(x.to(tl.float32)) * 0.4342944819032518


def log10(A):
    return log10_func(A)


def log10_(A):
    # ATen in-place log10_ is only valid for floating-point tensors; native
    # torch (CPU: "result type Float can't be cast to the desired output type
    # Long"; XPU/xdnn: [NOT IMPLEMENTED]) raises for integral inputs, while
    # writing back the float log10 into an int tensor would silently truncate.
    if not A.is_floating_point():
        raise TypeError(f"log10_ does not support dtype {A.dtype}")
    log10_func(A, out0=A)
    return A


def log10_out(A, out):
    return log10_func(A, out0=out)
