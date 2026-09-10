import logging

import torch
import triton
import triton.language as tl

from flag_gems.ops.index_select_backward import (
    index_select_backward as generic_index_select_backward,
)
from flag_gems.utils import dim_compress, libentry

from .mm import mm

logger = logging.getLogger(__name__)

_MAX_ONE_HOT_ELEMENTS = 20_000_000

# One-hot build config (2026-09-09 XPU3): the previous 2-D row-block kernel
# (R=32 rows x C=dim_size_out cols) is the measured hot spot of the whole op:
# for the (4096, 4104) benchmark shape it takes ~26 ms because dim_size_out is
# not a power of two, so the inner tl.arange tile is padded to 8192 lanes and
# every 2-D masked store is scalarised/spilled (measuring ~1.3 GB/s).  Every
# 2-D masked store variant measured is also numerically wrong (the backend
# drops/loses lanes when the mask kills a large share of a store tile), so the
# builder is rewritten as one program per row with only 1-D chunked masked
# stores (1-D masked stores are safe, measured 25x faster: ~26 ms -> ~1.05 ms
# at 4096x4104 fp16).  The inner chunk CB is a compile-time power of two and
# the last chunk carries the (light) remainder mask.
_ONE_HOT_CB = 2048
_MAX_ONE_HOT_INNER = 8192

# The one-hot gemm is the hot path for the large benchmark shapes. The
# general-mm wrapper picks BLOCK_K=256 for M,N > 512; the direct launch with
# BLOCK_K=128 measures ~1.3-1.4x faster on 4096^3-class tiles (2026-08-16
# XPU6 tile sweep), so launch the mm kernel directly for those shapes.
_DIRECT_MM_MIN = 1024
_DIRECT_MM_BM = 256
_DIRECT_MM_BN = 256
_DIRECT_MM_BK = 128
_DIRECT_MM_WARPS = 8


def _mm_large(a, b):
    """mm with the tuned tile for large K-multiples-of-128 shapes."""
    M, K = a.shape
    _, N = b.shape
    if (
        M < _DIRECT_MM_MIN
        or N < _DIRECT_MM_MIN
        or K % _DIRECT_MM_BK != 0
        or M % _DIRECT_MM_BM != 0
        or N % _DIRECT_MM_BN != 0
    ):
        return mm(a, b)
    c = torch.empty((M, N), dtype=a.dtype, device=a.device)
    grid = (
        (M // _DIRECT_MM_BM) * (N // _DIRECT_MM_BN),
        1,
    )
    from .mm import mm_kernel

    mm_kernel[grid](
        a,
        b,
        c,
        M,
        N,
        K,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        c.stride(0),
        c.stride(1),
        dot_out_dtype=tl.float32,
        BLOCK_M=_DIRECT_MM_BM,
        BLOCK_N=_DIRECT_MM_BN,
        BLOCK_K=_DIRECT_MM_BK,
        GROUP_M=1,
        SPLIT_K=1,
        EVEN_K=True,
        num_warps=_DIRECT_MM_WARPS,
        num_stages=2,
    )
    return c


@libentry()
@triton.jit
def _make_one_hot_rows_kernel(
    out,
    index,
    index_len,
    dim_size_out,
    CB: tl.constexpr,
    NCHUNK: tl.constexpr,
    IS_FP16: tl.constexpr,
    IS_BF16: tl.constexpr,
):
    # one-hot (index_len, dim_size_out); one program per index position, all
    # stores 1-D chunked (2-D masked stores miscompile on this backend).
    i = tl.program_id(0)
    if i < index_len:
        idx = tl.load(index + i)
        for j in tl.static_range(NCHUNK):
            cols = j * CB + tl.arange(0, CB)
            cmask = cols < dim_size_out
            val = cols == idx
            if IS_FP16:
                tl.store(out + i * dim_size_out + cols, val.to(tl.float16), mask=cmask)
            elif IS_BF16:
                tl.store(out + i * dim_size_out + cols, val.to(tl.bfloat16), mask=cmask)
            else:
                tl.store(out + i * dim_size_out + cols, val.to(tl.float32), mask=cmask)


@libentry()
@triton.jit
def _make_one_hot_cols_kernel(
    out,
    index,
    dim_size_out,
    index_len,
    CB: tl.constexpr,
    NCHUNK: tl.constexpr,
    IS_FP16: tl.constexpr,
    IS_BF16: tl.constexpr,
):
    # Transposed one-hot (dim_size_out, index_len): out[n, i] = (index[i] == n).
    # One program per output bucket row, 1-D chunked stores only.
    n = tl.program_id(0)
    if n < dim_size_out:
        for j in tl.static_range(NCHUNK):
            cols = j * CB + tl.arange(0, CB)
            cmask = cols < index_len
            idx = tl.load(index + cols, mask=cmask, other=-1)
            val = idx == n
            if IS_FP16:
                tl.store(out + n * index_len + cols, val.to(tl.float16), mask=cmask)
            elif IS_BF16:
                tl.store(out + n * index_len + cols, val.to(tl.bfloat16), mask=cmask)
            else:
                tl.store(out + n * index_len + cols, val.to(tl.float32), mask=cmask)


def index_select_backward(grad, self_sizes, dim, index):
    logger.debug("GEMS_KUNLUNXIN INDEX_SELECT_BACKWARD")

    dim = dim % grad.ndim
    index_len = index.numel()
    dim_size_out = self_sizes[dim]
    one_hot_elements = index_len * dim_size_out

    if (
        index_len == 0
        or one_hot_elements > _MAX_ONE_HOT_ELEMENTS
        or index_len > _MAX_ONE_HOT_INNER
        or dim_size_out > _MAX_ONE_HOT_INNER
        or grad.dtype not in (torch.float16, torch.bfloat16, torch.float32)
    ):
        return generic_index_select_backward(grad, self_sizes, dim, index)

    index = index.to(torch.int64)
    orig_dtype = grad.dtype
    is_fp16 = orig_dtype == torch.float16
    is_bf16 = orig_dtype == torch.bfloat16

    if dim == grad.ndim - 1:
        # out[..., k] = sum_i grad[..., i] * (index[i] == k)
        M = grad.numel() // index_len
        grad_flat = grad.reshape(M, index_len)
        one_hot = torch.empty(
            (index_len, dim_size_out), dtype=orig_dtype, device=grad.device
        )
        _make_one_hot_rows_kernel[(index_len,)](
            one_hot,
            index,
            index_len,
            dim_size_out,
            CB=_ONE_HOT_CB,
            NCHUNK=triton.cdiv(dim_size_out, _ONE_HOT_CB),
            IS_FP16=is_fp16,
            IS_BF16=is_bf16,
        )
        out = _mm_large(grad_flat, one_hot)
        return out.reshape(self_sizes)

    if dim == 0:
        # out[k, ...] = sum_i grad[i, ...] * (index[i] == k)
        M = grad.numel() // index_len
        grad_flat = grad.reshape(index_len, M)
        one_hot_t = torch.empty(
            (dim_size_out, index_len), dtype=orig_dtype, device=grad.device
        )
        _make_one_hot_cols_kernel[(dim_size_out,)](
            one_hot_t,
            index,
            dim_size_out,
            index_len,
            CB=_ONE_HOT_CB,
            NCHUNK=triton.cdiv(index_len, _ONE_HOT_CB),
            IS_FP16=is_fp16,
            IS_BF16=is_bf16,
        )
        out = _mm_large(one_hot_t, grad_flat)
        return out.reshape(self_sizes)

    # mid-dim: compressed (permute + one-hot + mm) path, exact but copies.
    grad_compressed = dim_compress(grad, dim)
    grad_flat = grad_compressed.reshape(-1, index_len)

    one_hot = torch.empty(
        (index_len, dim_size_out),
        dtype=orig_dtype,
        device=grad.device,
    )
    _make_one_hot_rows_kernel[(index_len,)](
        one_hot,
        index,
        index_len,
        dim_size_out,
        CB=_ONE_HOT_CB,
        NCHUNK=triton.cdiv(dim_size_out, _ONE_HOT_CB),
        IS_FP16=is_fp16,
        IS_BF16=is_bf16,
    )
    out_flat = _mm_large(grad_flat, one_hot)

    compressed_shape = list(grad_compressed.shape)
    compressed_shape[-1] = dim_size_out
    out_flat = out_flat.reshape(compressed_shape)
    order = [i for i in range(out_flat.ndim - 1)]
    order.insert(dim, out_flat.ndim - 1)
    out = out_flat.permute(order).contiguous()
    return out
