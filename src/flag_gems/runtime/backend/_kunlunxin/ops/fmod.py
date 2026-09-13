import logging

import triton
import triton.language as tl
import triton.language.extra.xpu.libdevice as xpu
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


@triton.jit
def _fmod(x, y):
    # x, y are fp32. Use the XPU native fmodf extern (xpu::fmodf, exact C
    # fmod semantics: x - trunc(x/y)*y, correctly rounded). The previous
    # implementation went through fp64 division, which is emulated on XPU
    # (measured ~10x slower on (4096,4096) fp32: 18.1ms vs 1.75ms). The
    # single-instruction extern also removes the fp32-quotient trunc
    # boundary hazard of an int-cast trunc implementation (off-by-one
    # quotient when x/y is within 1 ulp of an integer).
    return xpu.fmod(x, y)


@pointwise_dynamic(
    is_tensor=[True, True], promotion_methods=[(0, 1, "DEFAULT")], config=config_
)
@triton.jit
def fmod_func(x, y):
    dtype = x.dtype
    return _fmod(x.to(tl.float32), y.to(tl.float32)).to(dtype)


@pointwise_dynamic(
    is_tensor=[True, False], promotion_methods=[(0, 1, "DEFAULT")], config=config_
)
@triton.jit
def fmod_func_tensor_scalar(x, y):
    dtype = x.dtype
    return _fmod(x.to(tl.float32), y.to(tl.float32)).to(dtype)


def fmod_tensor(A, B):
    return fmod_func(A, B)


def fmod_scalar(A, B):
    return fmod_func_tensor_scalar(A, B)


def fmod_tensor_(A, B):
    return fmod_func(A, B, out0=A)


def fmod_scalar_(A, B):
    return fmod_func_tensor_scalar(A, B, out0=A)
