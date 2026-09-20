import logging
import struct

import torch
import triton
import triton.language as tl
from _kunlunxin.utils.codegen_config_utils import CodeGenConfig

from flag_gems.utils import tl_extra_shim

from ..utils.pointwise_dynamic import pointwise_dynamic

logger = logging.getLogger(__name__)
div_rn = tl_extra_shim.div_rn

config_ = CodeGenConfig(
    512,
    (65536, 65536, 65536),
    32,
    True,
    prefer_1d_tile=True,
    buffer_size_limit=4096,
    isCloseVectorization=True,
    unroll_num=8,
)


@pointwise_dynamic(promotion_methods=[(0, 1, "INT_TO_FLOAT")], config=config_)
@triton.jit
def true_div_func(x, y):
    return x / y


@pointwise_dynamic(is_tensor=[True, False], promotion_methods=[(0, 1, "INT_TO_FLOAT")])
@triton.jit
def true_div_func_tensor_scalar(x, y):
    return x / y


@pointwise_dynamic(is_tensor=[False, True], promotion_methods=[(0, 1, "INT_TO_FLOAT")])
@triton.jit
def true_div_func_scalar_tensor(x, y):
    return x / y


DIV_SCALAR_CFG_THRESHOLD = 1 << 20


@pointwise_dynamic(
    is_tensor=[True, False], promotion_methods=[(0, 1, "INT_TO_FLOAT")], config=config_
)
@triton.jit
def true_div_func_tensor_scalar_cfg(x, y):
    return x / y


@pointwise_dynamic(
    is_tensor=[False, True], promotion_methods=[(0, 1, "INT_TO_FLOAT")], config=config_
)
@triton.jit
def true_div_func_scalar_tensor_cfg(x, y):
    return x / y


CFG_UNROLL16 = CodeGenConfig(
    512,
    (65536, 65536, 65536),
    32,
    True,
    prefer_1d_tile=True,
    buffer_size_limit=4096,
    isCloseVectorization=True,
    unroll_num=16,
)


@pointwise_dynamic(
    is_tensor=[True, False],
    promotion_methods=[(0, 1, "INT_TO_FLOAT")],
    config=CFG_UNROLL16,
)
@triton.jit
def true_div_func_tensor_scalar_cfg16(x, y):
    return x / y


DIV_TENSOR_U16_MIN_NUMEL = 1 << 22


@pointwise_dynamic(promotion_methods=[(0, 1, "INT_TO_FLOAT")], config=CFG_UNROLL16)
@triton.jit
def true_div_func_u16(x, y):
    return x / y


def true_divide(A, B):
    logger.debug("GEMS_KUNLUNXIN TRUE_DIVIDE")
    if isinstance(A, torch.Tensor) and isinstance(B, torch.Tensor):
        if (
            A.dtype in (torch.float16, torch.float32)
            and A.numel() >= DIV_TENSOR_U16_MIN_NUMEL
        ):
            return true_div_func_u16(A, B)
        # Pre-allocate the output and pass it as `out0`. Without it,
        # pointwise_dynamic has to derive the promoted result dtype from scratch
        # on every call (`elementwise_dtypes(A, B, INT_TO_FLOAT)`), which costs
        # ~6.5us of pure host time on this stack - more than the whole device
        # time of a launch-bound shape like (64,64) (measured: the (64,64)
        # tensor-tensor case drops by ~8.5us, i.e. down to the bare triton
        # launch floor; the 16.7M-element cases are device bound and are
        # unaffected). Same pattern as rsub_scalar() in ../rsub.py and
        # fill_scalar() in ../fill.py.
        #
        # Passing out0 is only equivalent to the generic allocation when the
        # promoted result dtype is provably A.dtype and the task shape is
        # provably A.shape:
        #   * two operands of the SAME floating-point dtype promote to that
        #     dtype (INT_TO_FLOAT only widens integer/bool inputs, and a float
        #     operand always dominates them), and
        #   * with equal shapes the task space is that shape, so an
        #     `empty_like(A)` is exactly the buffer pointwise_dynamic would
        #     have allocated itself (it uses `torch.empty_like` on the first
        #     operand as well).
        # Every other combination (mixed dtypes, integer/bool/complex
        # operands, broadcasting) keeps the generic promotion path.
        if A.dtype == B.dtype and A.dtype.is_floating_point and A.shape == B.shape:
            return true_div_func(A, B, out0=torch.empty_like(A))
        return true_div_func(A, B)
    elif isinstance(A, torch.Tensor):
        if A.is_complex():
            return torch.view_as_complex(
                true_div_func_tensor_scalar(torch.view_as_real(A), B)
            )
        if A.numel() >= DIV_SCALAR_CFG_THRESHOLD:
            return true_div_func_tensor_scalar_cfg(A, B)
        return true_div_func_tensor_scalar(A, B)
    elif isinstance(B, torch.Tensor):
        if B.numel() >= DIV_SCALAR_CFG_THRESHOLD:
            return true_div_func_scalar_tensor_cfg(A, B)
        return true_div_func_scalar_tensor(A, B)
    else:
        return torch.tensor(A / B)


def true_divide_tensor(A, B):
    """Canonical Tensor overload of true_divide (explicit aten true_divide.Tensor).

    The generic flag_gems.ops.true_divide.true_divide_tensor routes through the
    generic flag_gems.ops.div.true_divide, whose pointwise kernel lacks the
    Kunlunxin tuned CodeGenConfig (measured ~330x slower on XPU for
    (4096,4096) fp32: 69ms vs 205us). Exporting this vendor implementation lets
    SpecOpRegistrar swap it in so torch.true_divide(tensor, tensor) and
    aten::true_divide.Tensor use the same fast tuned kernel as div/div_.
    """
    logger.debug("GEMS_KUNLUNXIN TRUE_DIVIDE_TENSOR")
    logging.getLogger("flag_gems.ops.true_divide").debug("GEMS TRUE_DIVIDE")
    return true_divide(A, B)


def true_divide_out(A, B, out):
    logger.debug("GEMS_KUNLUNXIN TRUE_DIVIDE_OUT")
    if isinstance(A, torch.Tensor) and isinstance(B, torch.Tensor):
        if (
            A.dtype in (torch.float16, torch.float32)
            and A.numel() >= DIV_TENSOR_U16_MIN_NUMEL
        ):
            return true_div_func_u16(A, B, out0=out)
        return true_div_func(A, B, out0=out)
    elif isinstance(A, torch.Tensor):
        if A.numel() >= DIV_SCALAR_CFG_THRESHOLD:
            return true_div_func_tensor_scalar_cfg(A, B, out0=out)
        return true_div_func_tensor_scalar(A, B, out0=out)
    elif isinstance(B, torch.Tensor):
        if B.numel() >= DIV_SCALAR_CFG_THRESHOLD:
            return true_div_func_scalar_tensor_cfg(A, B, out0=out)
        return true_div_func_scalar_tensor(A, B, out0=out)
    else:
        return torch.tensor(A / B) if out is None else out.fill_(A / B)


def true_divide_(A, B):
    logger.debug("GEMS_KUNLUNXIN TRUE_DIVIDE_")
    if isinstance(B, torch.Tensor):
        if (
            A.dtype in (torch.float16, torch.float32)
            and A.numel() >= DIV_TENSOR_U16_MIN_NUMEL
        ):
            return true_div_func_u16(A, B, out0=A)
        return true_div_func(A, B, out0=A)
    else:
        if A.numel() >= DIV_SCALAR_CFG_THRESHOLD:
            if A.dtype == torch.float32:
                return true_div_func_tensor_scalar_cfg16(A, B, out0=A)
            return true_div_func_tensor_scalar_cfg(A, B, out0=A)
        return true_div_func_tensor_scalar(A, B, out0=A)


def divide(A, B):
    """Out-of-place division (aten::divide): alias of true_divide."""
    logger.debug("GEMS_KUNLUNXIN DIVIDE")
    return true_divide(A, B)


def true_divide_tensor_(A, B):
    """Canonical Tensor overload for in-place true division (aten::true_divide.Tensor_)."""
    logger.debug("GEMS_KUNLUNXIN TRUE_DIVIDE_TENSOR_")
    logging.getLogger("flag_gems.ops.true_divide_").debug("GEMS TRUE_DIVIDE_")
    return true_divide_(A, B)


@triton.jit
def _trunc_q(q):
    return tl.where(tl.abs(q) < 8388608.0, tl.cast(q, tl.int32).to(tl.float32), q)


@triton.jit
def _floor_q(q):
    t = tl.where(tl.abs(q) < 8388608.0, tl.cast(q, tl.int32).to(tl.float32), q)
    return tl.where((q < 0) & (q != t), t - 1.0, t)


@triton.jit
def _floor_div_fp32(x, y):
    q = div_rn(x, y)
    t = tl.where(tl.abs(q) < 8388608.0, tl.cast(q, tl.int32).to(tl.float32), q)
    mod0 = tl.fma(t, -y, x)
    adj = (mod0 != 0.0) & ((y < 0.0) != (mod0 < 0.0))
    div = div_rn(x - mod0, y)
    div = tl.where(adj, div - 1.0, div)
    fd = _floor_q(div)
    fd = tl.where(div - fd > 0.5, fd + 1.0, fd)
    fd = tl.where(
        div == 0.0,
        tl.where((x < 0.0) != (y < 0.0), -0.0, 0.0),
        fd,
    )
    return tl.where(y == 0.0, q, fd)


@pointwise_dynamic(promotion_methods=[(0, 1, "DEFAULT")], config=config_)
@triton.jit
def trunc_div_func(x, y):
    return _trunc_q(div_rn(x, y))


@pointwise_dynamic(
    is_tensor=[True, False], promotion_methods=[(0, 1, "DEFAULT")], config=config_
)
@triton.jit
def trunc_div_func_tensor_scalar(x, y):
    return _trunc_q(div_rn(x, tl.cast(y, x.dtype)))


@pointwise_dynamic(
    is_tensor=[False, True], promotion_methods=[(0, 1, "DEFAULT")], config=config_
)
@triton.jit
def trunc_div_func_scalar_tensor(x, y):
    return _trunc_q(div_rn(tl.cast(x, y.dtype), y))


@pointwise_dynamic(promotion_methods=[(0, 1, "DEFAULT")])
@triton.jit
def trunc_div_int_func(x, y):
    return x // y


@pointwise_dynamic(is_tensor=[True, False], promotion_methods=[(0, 1, "DEFAULT")])
@triton.jit
def trunc_div_int_func_tensor_scalar(x, y):
    return x // y


@pointwise_dynamic(is_tensor=[False, True], promotion_methods=[(0, 1, "DEFAULT")])
@triton.jit
def trunc_div_int_func_scalar_tensor(x, y):
    return x // y


def trunc_divide(A, B):
    logger.debug("GEMS_KUNLUNXIN TRUNC_DIVIDE")
    if isinstance(A, torch.Tensor) and not A.is_floating_point():
        if isinstance(B, torch.Tensor):
            return trunc_div_int_func(A, B)
        else:
            return trunc_div_int_func_tensor_scalar(A, B)
    if isinstance(B, torch.Tensor) and not B.is_floating_point():
        return trunc_div_int_func_scalar_tensor(A, B)
    if isinstance(A, torch.Tensor) and isinstance(B, torch.Tensor):
        return trunc_div_func(A, B)
    elif isinstance(A, torch.Tensor):
        return trunc_div_func_tensor_scalar(A, B)
    elif isinstance(B, torch.Tensor):
        return trunc_div_func_scalar_tensor(A, B)
    else:
        return torch.tensor(A / B)


def trunc_divide_(A, B):
    logger.debug("GEMS_KUNLUNXIN TRUNC_DIVIDE_")
    if not A.is_floating_point():
        if isinstance(B, torch.Tensor):
            return trunc_div_int_func(A, B, out0=A)
        else:
            return trunc_div_int_func_tensor_scalar(A, B, out0=A)
    if isinstance(B, torch.Tensor):
        return trunc_div_func(A, B, out0=A)
    else:
        return trunc_div_func_tensor_scalar(A, B, out0=A)


@triton.jit
def _int_floordiv(x, y):
    # Triton `//` and `%` on integers are truncating (C semantics), while PyTorch
    # floor-division additionally needs a one-off correction when the signs
    # differ and the remainder is non-zero.
    # The remainder is derived from the quotient instead of using `x % y`:
    # for truncating division the two are exactly equivalent, and it keeps a
    # single integer division per element. The previous form emitted both
    # `llvm.srem` and `llvm.sdiv` for every value (IR verified on XPU), which
    # doubled the cost of the dominant operation of this kernel.
    q = x // y
    r = x - q * y
    c1 = r != 0
    c2 = (x < 0) ^ (y < 0)
    return tl.where(c1 & c2, q - 1, q)


@triton.jit
def _float_floordiv_corrected(x, y):
    return _floor_div_fp32(x, y)


@pointwise_dynamic(promotion_methods=[(0, 1, "DEFAULT")], config=config_)
@triton.jit
def floor_div_func_corrected(x, y):
    if x.type.scalar.is_int() & y.type.scalar.is_int():
        return _int_floordiv(x, y)
    else:
        return _float_floordiv_corrected(x.to(tl.float32), y.to(tl.float32))


@pointwise_dynamic(
    is_tensor=[True, False], promotion_methods=[(0, "DEFAULT")], config=config_
)
@triton.jit
def floor_div_func_corrected_tensor_scalar(x, y):
    if x.type.scalar.is_int() & y.type.scalar.is_int():
        return _int_floordiv(x, y)
    else:
        return _float_floordiv_corrected(x.to(tl.float32), y.to(tl.float32))


@pointwise_dynamic(is_tensor=[True, False], promotion_methods=[(0, "DEFAULT")])
@triton.jit
def floor_div_lowp_tensor_scalar_func(x, y):
    y = tl.full(x.shape, y, x.dtype)
    return _floor_div_fp32(x.to(tl.float32), y.to(tl.float32))


@pointwise_dynamic(
    is_tensor=[False, True], promotion_methods=[(0, 1, "DEFAULT")], config=config_
)
@triton.jit
def floor_div_func_corrected_scalar_tensor(x, y):
    if x.type.scalar.is_int() & y.type.scalar.is_int():
        return _int_floordiv(x, y)
    else:
        return _float_floordiv_corrected(x.to(tl.float32), y.to(tl.float32))


def _as_bfloat16_scalar(value):
    bits = struct.unpack(">I", struct.pack(">f", float(value)))[0]
    exponent = bits & 0x7F800000
    mantissa = bits & 0x007FFFFF
    if exponent != 0x7F800000:
        bits += 0x7FFF + ((bits >> 16) & 1)
    elif mantissa:
        bits |= 0x00400000
    bits &= 0xFFFF0000
    return struct.unpack(">f", struct.pack(">I", bits))[0]


def floor_divide(A, B):
    logger.debug("GEMS_KUNLUNXIN FLOOR_DIVIDE")
    if isinstance(A, torch.Tensor) and isinstance(B, torch.Tensor):
        return floor_div_func_corrected(A, B)
    elif isinstance(A, torch.Tensor):
        if A.dtype in (torch.float16, torch.bfloat16):
            if A.dtype == torch.bfloat16:
                B = _as_bfloat16_scalar(B)
            return floor_div_lowp_tensor_scalar_func(A, B)
        # Allocate the result tensor here and pass it as `out0`. Without it,
        # pointwise_dynamic re-derives the promoted result dtype from scratch on
        # every call (`type_promotion(A, DEFAULT)` inside prepare_args), which is
        # pure host time paid before the launch and dominates at a launch-bound
        # shape such as (64,64). Same pattern as true_divide() above in this
        # file, rsub_scalar() in ../rsub.py and clamp_max() in ../clamp.py; the
        # large shapes are device bound and are unaffected.
        #
        # Passing out0 is only equivalent to the generic allocation when the
        # result is provably A.dtype with A.shape AND A.is_contiguous():
        #   * the tensor-scalar schema uses promotion method `[(0, "DEFAULT")]`,
        #     i.e. only operand 0 (the tensor) participates, so DEFAULT promotion
        #     of A is A.dtype for every dtype A can have here;
        #   * with a single tensor input the task shape is A.shape;
        #   * `A.is_contiguous()` keeps the *route* identical as well: without
        #     out0 the fast path is decided on the single-element list [A], which
        #     a non-overlapping-dense-but-non-contiguous A also passes, whereas
        #     with out0 the list becomes [out0, A] and that fast path would not
        #     be taken. Non-contiguous inputs therefore keep the pre-existing
        #     generic route untouched.
        if A.is_contiguous():
            return floor_div_func_corrected_tensor_scalar(
                A, B, out0=torch.empty_like(A)
            )
        return floor_div_func_corrected_tensor_scalar(A, B)
    elif isinstance(B, torch.Tensor):
        return floor_div_func_corrected_scalar_tensor(A, B)
    else:
        return torch.tensor(A // B)


def floor_divide_(A, B):
    logger.debug("GEMS_KUNLUNXIN FLOOR_DIVIDE_")
    if isinstance(B, torch.Tensor):
        return floor_div_func_corrected(A, B, out0=A)
    else:
        if A.dtype in (torch.float16, torch.bfloat16):
            if A.dtype == torch.bfloat16:
                B = _as_bfloat16_scalar(B)
            return floor_div_lowp_tensor_scalar_func(A, B, out0=A)
        return floor_div_func_corrected_tensor_scalar(A, B, out0=A)


def div_mode(A, B, rounding_mode=None):
    if rounding_mode is None:
        return true_divide(A, B)
    elif rounding_mode == "trunc":
        return trunc_divide(A, B)
    elif rounding_mode == "floor":
        return floor_divide(A, B)
    else:
        msg = f"div expected rounding_mode to be one of None, 'trunc', or 'floor' but found {rounding_mode}."
        raise ValueError(msg)


def div_mode_(A, B, rounding_mode=None):
    if rounding_mode is None:
        return true_divide_(A, B)
    elif rounding_mode == "trunc":
        return trunc_divide_(A, B)
    elif rounding_mode == "floor":
        return floor_divide_(A, B)
    else:
        msg = f"div expected rounding_mode to be one of None, 'trunc', or 'floor' but found {rounding_mode}."
        raise ValueError(msg)


@triton.jit
def _remainder(x, y):
    r = x % y
    c1 = r != 0
    c2 = (x < 0) ^ (y < 0)
    return tl.where(c1 & c2, r + y, r)


@pointwise_dynamic(promotion_methods=[(0, 1, "DEFAULT")])
@triton.jit
def rem_tt(x, y):
    return _remainder(x, y)


@pointwise_dynamic(is_tensor=[True, False], promotion_methods=[(0, 1, "DEFAULT")])
@triton.jit
def rem_ts(x, y):
    return _remainder(x, y)


@pointwise_dynamic(is_tensor=[False, True], promotion_methods=[(0, 1, "DEFAULT")])
@triton.jit
def rem_st(x, y):
    return _remainder(x, y)


REMAINDER_CFG_THRESHOLD = 1 << 20


@pointwise_dynamic(promotion_methods=[(0, 1, "DEFAULT")], config=config_)
@triton.jit
def rem_tt_cfg(x, y):
    return _remainder(x, y)


@pointwise_dynamic(
    is_tensor=[True, False], promotion_methods=[(0, 1, "DEFAULT")], config=config_
)
@triton.jit
def rem_ts_cfg(x, y):
    return _remainder(x, y)


@pointwise_dynamic(
    is_tensor=[False, True], promotion_methods=[(0, 1, "DEFAULT")], config=config_
)
@triton.jit
def rem_st_cfg(x, y):
    return _remainder(x, y)


def _fold_scalar_into_tensor_dtype(value, dtype):
    """Reproduce ATen's "wrapped number" fold for the scalar operand.

    ``aten::remainder.Scalar_Tensor`` converts the Python scalar with
    ``Scalar::to<T>()`` (a C-style truncating conversion) *before* the kernel
    runs, so ``300 % <int8 tensor>`` really computes ``44 % y``. The shared
    pointwise generator instead keeps the scalar in a wide integer type and
    truncates only when storing to the (narrower) output dtype, which silently
    disagrees with ATen for any scalar that does not fit the tensor dtype.
    Verified on XPU (aten CPU oracle): int8 300/130, int16 40000/32768,
    int32 +/-2**40/2**31 all returned the wide-scalar result.

    Python ``bool`` is folded to ``int`` as well: the generated scalar kernel
    carries ``do_not_specialize=["val0"]``, so a ``bool`` first argument binds
    ``val0`` to ``i1`` and aborts with ``CompilationError`` (ATen returns the
    ``True % y`` result).
    """
    if isinstance(value, bool):
        value = int(value)
    if (
        isinstance(value, int)
        and not dtype.is_floating_point
        and not dtype.is_complex
        and dtype != torch.bool
    ):
        info = torch.iinfo(dtype)
        mod = 1 << info.bits
        value &= mod - 1
        if info.min < 0 and value > info.max:
            value -= mod
    return value


def remainder(A, B):
    logger.debug("GEMS_KUNLUNXIN FLOOR_DIVIDE")
    if isinstance(A, torch.Tensor) and isinstance(B, torch.Tensor):
        if max(A.numel(), B.numel()) >= REMAINDER_CFG_THRESHOLD:
            return rem_tt_cfg(A, B)
        return rem_tt(A, B)
    elif isinstance(A, torch.Tensor):
        if A.numel() >= REMAINDER_CFG_THRESHOLD:
            return rem_ts_cfg(A, B)
        return rem_ts(A, B)
    elif isinstance(B, torch.Tensor):
        A = _fold_scalar_into_tensor_dtype(A, B.dtype)
        if B.numel() >= REMAINDER_CFG_THRESHOLD:
            return rem_st_cfg(A, B)
        return rem_st(A, B)
    else:
        return torch.tensor(A % B)


def remainder_(A, B):
    logger.debug("GEMS_KUNLUNXIN REMAINDER_")
    if isinstance(B, torch.Tensor):
        if max(A.numel(), B.numel()) >= REMAINDER_CFG_THRESHOLD:
            return rem_tt_cfg(A, B, out0=A)
        return rem_tt(A, B, out0=A)
    else:
        if A.numel() >= REMAINDER_CFG_THRESHOLD:
            return rem_ts_cfg(A, B, out0=A)
        return rem_ts(A, B, out0=A)
