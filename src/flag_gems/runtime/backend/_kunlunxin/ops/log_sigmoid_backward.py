import logging

import torch
import triton
import triton.language as tl
import triton.language.extra.xpu.libdevice as xpu
from _kunlunxin.utils.codegen_config_utils import CodeGenConfig

from flag_gems.utils import triton_lang_extension as ext

from ..utils.pointwise_dynamic import pointwise_dynamic as xpu_pointwise_dynamic

logger = logging.getLogger(__name__)


BUFFER_SIZE_LIMIT = 8192
IS_CLOSE_MEMORY_ASYNC = False
FLAT_NUM_WARPS = 8

# Flat-kernel tuning, measured on P800 (xpu3) with 16.7M contiguous elements:
# dtype -> (BLOCK_SIZE, unroll_num, buffer_size_limit).  See
# harness/solution/log_sigmoid_backward/README.md for the raw numbers.
FLAT_TUNE = {
    torch.float16: (131072, 2, 4096),
    torch.bfloat16: (65536, 8, BUFFER_SIZE_LIMIT),
    torch.float32: (32768, 4, BUFFER_SIZE_LIMIT),
}
FLAT_TUNE_DEFAULT = (65536, 4, BUFFER_SIZE_LIMIT)

# The tuned tiles launch very few programs below this element count, so small
# inputs use a smaller tile with more programs (measured faster for all dtypes).
SMALL_MAX_NUMEL = 1 << 21
SMALL_TUNE = {
    torch.float16: (16384, 2, BUFFER_SIZE_LIMIT),
    torch.bfloat16: (16384, 8, BUFFER_SIZE_LIMIT),
    torch.float32: (16384, 4, BUFFER_SIZE_LIMIT),
}
SMALL_TUNE_DEFAULT = (16384, 4, BUFFER_SIZE_LIMIT)

config_ = CodeGenConfig(
    512,
    (65536, 65536, 65536),
    32,
    True,
    prefer_1d_tile=True,
    buffer_size_limit=4096,
    kunlunAutoGrid=True,
    unroll_num=8,
)


@xpu_pointwise_dynamic(promotion_methods=[(0, 1, "DEFAULT")], config=config_)
@triton.jit
def log_sigmoid_backward_func(grad_output, self):
    go = grad_output.to(tl.float32)
    x = self.to(tl.float32)
    return (go * tl.sigmoid(0.0 - x)).to(grad_output.dtype)


def _pick_block(n_elements, dtype):
    if n_elements <= SMALL_MAX_NUMEL:
        block_size, unroll_num, buffer_size = SMALL_TUNE.get(dtype, SMALL_TUNE_DEFAULT)
    else:
        block_size, unroll_num, buffer_size = FLAT_TUNE.get(dtype, FLAT_TUNE_DEFAULT)
    if n_elements % block_size == 0:
        return block_size, FLAT_NUM_WARPS, unroll_num, buffer_size, False
    if n_elements <= 65536:
        return 2048, 4, 2, BUFFER_SIZE_LIMIT, True
    if n_elements <= (1 << 20):
        return 16384, FLAT_NUM_WARPS, 2, BUFFER_SIZE_LIMIT, True
    return 65536, FLAT_NUM_WARPS, 2, BUFFER_SIZE_LIMIT, True


@triton.jit
def _log_sigmoid_backward_derivative(x):
    # d/dx log_sigmoid(x) = sigmoid(-x) = 0.5 * (1 - tanh(x / 2)).
    # A single `tanhf` call measured ~1.6x faster than `1 / (1 + exp(x))` here:
    # `tl.exp` lowers to the fast `llvm.intr.exp2` intrinsic, but the fp32
    # divide that follows it dominates on this backend.
    return 0.5 - 0.5 * xpu.tanh(0.5 * x)


@triton.jit
def log_sigmoid_backward_flat_kernel(
    grad_output_ptr,
    self_ptr,
    grad_input_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = ext.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    g = tl.load(grad_output_ptr + offsets, mask=mask)
    x = tl.load(self_ptr + offsets, mask=mask)
    derivative = _log_sigmoid_backward_derivative(x.to(tl.float32))
    res = g.to(tl.float32) * derivative
    tl.store(
        grad_input_ptr + offsets,
        res.to(grad_input_ptr.dtype.element_ty),
        mask=mask,
    )


@triton.jit
def log_sigmoid_backward_flat_kernel_unmasked(
    grad_output_ptr,
    self_ptr,
    grad_input_ptr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = ext.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    g = tl.load(grad_output_ptr + offsets)
    x = tl.load(self_ptr + offsets)
    derivative = _log_sigmoid_backward_derivative(x.to(tl.float32))
    res = g.to(tl.float32) * derivative
    tl.store(grad_input_ptr + offsets, res.to(grad_input_ptr.dtype.element_ty))


def _can_use_flat_kernel(grad_output, self, grad_input=None):
    return (
        grad_output.shape == self.shape
        and grad_output.dtype == self.dtype
        and grad_output.is_contiguous()
        and self.is_contiguous()
        and (
            grad_input is None
            or (
                grad_input.shape == self.shape
                and grad_input.dtype == self.dtype
                and grad_input.is_contiguous()
            )
        )
    )


def _launch_flat_kernel(grad_output, self, grad_input):
    n_elements = self.numel()
    if n_elements == 0:
        return grad_input
    block_size, num_warps, unroll_num, buffer_size, masked = _pick_block(
        n_elements, self.dtype
    )
    if masked:
        grid = (triton.cdiv(n_elements, block_size),)
        log_sigmoid_backward_flat_kernel[grid](
            grad_output,
            self,
            grad_input,
            n_elements,
            BLOCK_SIZE=block_size,
            num_warps=num_warps,
            unroll_num=unroll_num,
            buffer_size_limit=buffer_size,
            isCloseMemoryAsync=IS_CLOSE_MEMORY_ASYNC,
        )
    else:
        grid = (n_elements // block_size,)
        log_sigmoid_backward_flat_kernel_unmasked[grid](
            grad_output,
            self,
            grad_input,
            BLOCK_SIZE=block_size,
            num_warps=num_warps,
            unroll_num=unroll_num,
            buffer_size_limit=buffer_size,
            isCloseMemoryAsync=IS_CLOSE_MEMORY_ASYNC,
        )
    return grad_input


def log_sigmoid_backward(grad_output, self, buffer):
    logger.debug("GEMS_KUNLUNXIN LOG_SIGMOID_BACKWARD")
    if _can_use_flat_kernel(grad_output, self):
        return _launch_flat_kernel(grad_output, self, torch.empty_like(self))
    return log_sigmoid_backward_func(grad_output, self)


def log_sigmoid_backward_out(grad_output, self, buffer, *, grad_input):
    logger.debug("GEMS_KUNLUNXIN LOG_SIGMOID_BACKWARD OUT")
    if _can_use_flat_kernel(grad_output, self, grad_input):
        return _launch_flat_kernel(grad_output, self, grad_input)
    return log_sigmoid_backward_func(grad_output, self, out0=grad_input)
