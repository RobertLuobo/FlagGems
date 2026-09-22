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


@pointwise_dynamic(
    is_tensor=[True, True, True, True],
    num_outputs=2,
    promotion_methods=[(0, 1, 2, 3, "INT_TO_FLOAT"), (0, 1, 2, 3, "INT_TO_FLOAT")],
    config=config_,
)
@triton.jit
def true_div_complex_kernel(ar, ai, br, bi):
    # Smith's method complex division: divide by the larger-magnitude component
    # so the ratio is bounded by 1 (avoids overflow and keeps the error at a
    # couple of ulp), matching torch's own complex division algorithm.
    abs_br = tl.abs(br)
    abs_bi = tl.abs(bi)
    use_br = abs_br >= abs_bi

    # When |br| >= |bi|: ratio = bi/br, denom = br + bi*ratio
    ratio1 = tl.where(br == 0, 0.0, bi / br)
    denom1 = br + bi * ratio1
    real1 = (ar + ai * ratio1) / denom1
    imag1 = (ai - ar * ratio1) / denom1

    # When |bi| > |br|: ratio = br/bi, denom = bi + br*ratio
    ratio2 = tl.where(bi == 0, 0.0, br / bi)
    denom2 = bi + br * ratio2
    real2 = (ar * ratio2 + ai) / denom2
    imag2 = (ai * ratio2 - ar) / denom2

    real = tl.where(use_br, real1, real2)
    imag = tl.where(use_br, imag1, imag2)
    return real, imag


def _complex_real_parts(z, upcast):
    zr = torch.view_as_real(z)
    if upcast:
        zr = zr.to(torch.float32)
    return zr[..., 0].contiguous(), zr[..., 1].contiguous()


def _true_divide_complex_tensors(A, B): 
    A_is_complex = A.is_complex()
    B_is_complex = B.is_complex()
    if A_is_complex and B_is_complex:
        upcast = A.dtype == torch.complex32
        ar, ai = _complex_real_parts(A, upcast)
        br, bi = _complex_real_parts(B, upcast)
        real, imag = true_div_complex_kernel(ar, ai, br, bi)
        if upcast:
            real, imag = real.to(torch.float16), imag.to(torch.float16)
        return torch.view_as_complex(torch.stack((real, imag), dim=-1))
    elif A_is_complex:
        # (a+bi) / c: divide both lanes by the real tensor (broadcast)
        upcast = A.dtype == torch.complex32
        Ar = torch.view_as_real(A)
        if upcast:
            Ar = Ar.to(torch.float32)
            Br = B.unsqueeze(-1).to(torch.float32)
        else:
            Br = B.unsqueeze(-1)
        out = true_div_func(Ar, Br)
        if upcast:
            out = out.to(torch.float16)
        return torch.view_as_complex(out.contiguous())
    else:
        # a / (c+di) == (a+0i) / (c+di)
        #
        # NOTE: c5694666 wrote `ar = A.unsqueeze(-1)` + `ai = zeros_like(br)`,
        # which broadcasts to rank(A)+1 (e.g. (32,32)/(32,32)c -> (32,32,32)c);
        # that came from the `A_is_complex` branch where the lane dim really
        # exists. Here both lanes must have A's own shape.
        upcast = B.dtype == torch.complex32
        br, bi = _complex_real_parts(B, upcast)
        ar = A.to(br.dtype)
        ai = torch.zeros_like(ar)
        real, imag = true_div_complex_kernel(ar, ai, br, bi)
        if upcast:
            real, imag = real.to(torch.float16), imag.to(torch.float16)
        return torch.view_as_complex(torch.stack((real, imag), dim=-1))


def _same_layout_out0(A, B): 
    if A.is_floating_point() and B.dtype == A.dtype and B.shape == A.shape:
        return torch.empty_like(A)
    return None


def true_divide(A, B):
    logger.debug("GEMS_KUNLUNXIN TRUE_DIVIDE")
    if isinstance(A, torch.Tensor) and isinstance(B, torch.Tensor):
        if A.is_complex() or B.is_complex():
            return _true_divide_complex_tensors(A, B)
        kernel = true_div_func
        if (
            A.dtype in (torch.float16, torch.float32)
            and A.numel() >= DIV_TENSOR_U16_MIN_NUMEL
        ):
            kernel = true_div_func_u16
        out0 = _same_layout_out0(A, B)
        if out0 is not None:
            return kernel(A, B, out0=out0)
        return kernel(A, B)
    elif isinstance(A, torch.Tensor):
        if A.is_complex():
            # The pointwise code generator has no complex scalar dtype mapping.
            # Divide interleaved real/imag lanes with the existing Triton kernel.
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
