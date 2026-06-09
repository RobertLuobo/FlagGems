import logging

import torch
import triton
import triton.language as tl

from flag_gems import runtime
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import dim_compress, libentry
from flag_gems.utils import triton_lang_extension as ext

logger = logging.getLogger(__name__)


@triton.jit
def welford_func(mean_x, count_x, M_x, mean_y, count_y, M_y):
    count = count_x + count_y
    delta = mean_y - mean_x
    ratio = tl.where(count > count - count, count_y / count, count - count)
    mean = mean_x + delta * ratio
    M = M_x + M_y + delta * delta * count_x * ratio
    return mean, count, M


@libentry()
@triton.autotune(configs=runtime.get_tuned_config("var_mean"), key=["M", "N"])
@triton.jit(do_not_specialize=["correction"])
def var_welford_kernel(
    X,
    Var,
    M,
    N,
    correction,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    # Map the program id to the row of X it should compute.
    pid = ext.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]
    X = X + pid * N
    Var = Var + pid
    row_mask = pid < M

   # Pass 1: compute mean
    _sum = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
    for off in range(0, N, BLOCK_N):
        cols = off + tl.arange(0, BLOCK_N)[None, :]
        col_mask = cols < N
        mask = row_mask and col_mask
        x = tl.load(X + cols, mask, other=0.0).to(tl.float32)
        _sum += x
    mean = tl.sum(_sum, axis=1)[:, None] / N

    # Pass 2: compute sum of squared deviations
    _acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
    for off in range(0, N, BLOCK_N):
        cols = off + tl.arange(0, BLOCK_N)[None, :]
        col_mask = cols < N
        mask = row_mask and col_mask
        x = tl.load(X + cols, mask, other=0.0).to(tl.float32)
        diff = x - mean
        _acc += diff * diff * mask
    var = tl.sum(_acc, axis=1)[:, None] / (N - correction)
    tl.store(Var, var, row_mask)



@libentry()
@triton.jit
def var_kernel_1(
    X,
    Acc,
    Average,
    Count,
    N,
    BLOCK_N: tl.constexpr,
):
    # Map the program id to the row of X it should compute.
    pid = ext.program_id(0)
    offset = pid * BLOCK_N + tl.arange(0, BLOCK_N)

    X = X + offset
    Acc = Acc + pid
    Average = Average + pid
    Count = Count + pid
    mask = offset < N

    x = tl.load(X, mask, other=0.0).to(tl.float32)

    count = tl.sum(mask.to(tl.float32))
    average = tl.sum(x) / count
    acc = tl.sum(x * x) - count * average * average

    tl.store(Average, average)
    tl.store(Acc, acc)
    tl.store(Count, count)


@libentry()
@triton.heuristics(runtime.get_heuristic_config("var_mean"))
@triton.jit(do_not_specialize=["correction"])
def var_kernel_2(
    Acc,
    Average,
    Count,
    Var,
    N,
    correction,
    BLOCK_NUM,
    BLOCK_N: tl.constexpr,
):
    offset = tl.arange(0, BLOCK_N)
    mask = offset < BLOCK_NUM
    Acc = Acc + offset
    Average = Average + offset
    Count = Count + offset
    acc = tl.load(Acc, mask, other=0.0).to(tl.float32)
    average = tl.load(Average, mask, other=0.0).to(tl.float32)
    count = tl.load(Count, mask, other=0.0).to(tl.float32)

    total_count = tl.sum(count)
    total_sum = tl.sum(average * count)
    global_mean = total_sum / total_count
    delta = (average - global_mean) * mask
    nvar = tl.sum(acc + count * delta * delta)

    var = nvar / (N - correction)
    tl.store(Var, var)


def var(x, dim=None, *, correction=None, keepdim=False):
    logger.debug("GEMS VAR")
    if correction is None:
        correction = 1.0

    if dim is None or len(dim) == x.ndim:
        dim = list(range(x.ndim))
        shape = [1] * x.ndim
        N = x.numel()
        var = torch.empty(shape, dtype=x.dtype, device=x.device)
        BLOCK_N = 1024
        BLOCK_NUM = triton.cdiv(N, BLOCK_N)
        acc = torch.empty([BLOCK_NUM], dtype=x.dtype, device=x.device)
        average = torch.empty([BLOCK_NUM], dtype=x.dtype, device=x.device)
        count = torch.empty([BLOCK_NUM], dtype=x.dtype, device=x.device)

        with torch_device_fn.device(x.device):
            var_kernel_1[(BLOCK_NUM,)](x, acc, average, count, N, BLOCK_N=BLOCK_N)
            var_kernel_2[(1,)](acc, average, count, var, N, correction, BLOCK_NUM)
    else:
        shape = list(x.shape)
        dim = [d % x.ndim for d in dim]
        x = dim_compress(x, dim)
        N = 1
        for i in dim:
            N *= shape[i]
            shape[i] = 1
        M = x.numel() // N
        var = torch.empty(shape, dtype=x.dtype, device=x.device)

        grid = lambda META: (triton.cdiv(M, META["BLOCK_M"]),)
        with torch_device_fn.device(x.device):
            var_welford_kernel[grid](x, var, M, N, correction)

    if not keepdim:
        var = var.squeeze(dim=dim)
    return var


def var_dim(x, dim=None, *, correction=None, keepdim=False):
    logger.debug("GEMS VAR_DIM")
    return var(x, dim=dim, correction=correction, keepdim=keepdim)


def var_correction(x, dim=None, *, correction=None, keepdim=False):
    logger.debug("GEMS VAR_CORRECTION")
    return var(x, dim=dim, correction=correction, keepdim=keepdim)
