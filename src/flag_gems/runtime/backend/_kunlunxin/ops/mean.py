# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import builtins
import logging

import torch
import triton
import triton.language as tl

# from flag_gems import runtime
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import dim_compress, libentry
from flag_gems.utils import triton_lang_extension as ext

from ..utils.block_size_utils import get_block_size_1d

logger = logging.getLogger(__name__)


@libentry()
@triton.jit
def mean_scalar_kernel(inp, out, M, BLOCK_SIZE: tl.constexpr):
    """Scalar mean over all M elements.
    On XPU (USE_XHPC): intercepted by baidu::xpu::api::mean binding.
    Triton fallback (single CTA): sequential accumulation for correctness.
    Params for binding:
      kernelParams[0] = inp, kernelParams[1] = out
      kernelConsts[2] = M,   kernelConsts[3] = BLOCK_SIZE
    """
    acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for off in range(0, M, BLOCK_SIZE):
        offset = off + tl.arange(0, BLOCK_SIZE)
        mask = offset < M
        v = tl.load(inp + offset, mask=mask, other=0.0).to(tl.float32)
        acc += v
    result = tl.sum(acc) / M
    tl.store(out, result)


def mean(inp, *, dtype=None):
    logger.debug("GEMS_KUNLUNXIN MEAN")
    M = inp.numel()
    if dtype is None:
        dtype = inp.dtype
    if M == 0:
        # torch.mean of an empty tensor is 0/0 = NaN; the flat kernel below
        # cannot handle a zero trip count.
        return torch.full([], float("nan"), dtype=dtype, device=inp.device)
    BLOCK_SIZE = get_block_size_1d(M, inp.element_size())
    out = torch.empty([], dtype=dtype, device=inp.device)

    with torch_device_fn.device(inp.device):
        mean_scalar_kernel[(1, 1, 1)](inp, out, M, BLOCK_SIZE, buffer_size_limit=2048)
    return out

_TILE_BUDGET = 32768
_N_WIDE = 8192


def _block_n(N):
    if N > _N_WIDE:
        return builtins.min(triton.next_power_of_2(N), 2048)  # wide for large N
    return builtins.min(triton.next_power_of_2(N), 512)  # tall-friendly otherwise


def heur_n_block_size(args):
    return _block_n(args["N"])


def heur_m_block_size(args):
    block_n = _block_n(args["N"])
    block_m = triton.next_power_of_2(triton.cdiv(args["M"], 12))  # cluster_num
    return builtins.min(block_m, builtins.max(_TILE_BUDGET // block_n, 1))


@libentry()
# @triton.autotune(
#     configs=runtime.get_tuned_config("mean"),
#     key=["M", "N"],
# )
@triton.heuristics(
    values={
        "BLOCK_M": heur_m_block_size,
        "BLOCK_N": heur_n_block_size,
    },
)
@triton.jit
def mean_dim_kernel(X, Mean, M, N, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    """2-D reduction: reduce N-dim for each of M rows.
    On XPU (USE_XHPC): intercepted by baidu::xpu::api::mean_dim binding.
    Params for binding:
      kernelParams[0] = X,    kernelParams[1] = Mean
      kernelParams[2] = M,    kernelParams[3] = N  (runtime scalars)
      kernelConsts[4] = BLOCK_M (constexpr), kernelConsts[5] = BLOCK_N (constexpr)
    """
    # Map the program id to the row of X it should compute.
    pid = ext.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]
    X = X + pid * N
    Mean = Mean + pid
    row_mask = pid < M

    _mean = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
    for off in range(0, N, BLOCK_N):
        cols = off + tl.arange(0, BLOCK_N)[None, :]
        col_mask = cols < N
        mask = row_mask and col_mask

        a = tl.load(X + cols, mask, other=0.0).to(tl.float32)
        _mean += a
    mean = tl.sum(_mean, axis=1)[:, None] / N
    tl.store(Mean, mean, row_mask)


@libentry()
@triton.jit
def mean_dim_mid_kernel(X, Out, M, N, K, BLOCK_K: tl.constexpr):
    """Mid-dim (K>1) row-reduce WITHOUT the dim_compress transpose copy.

    X is the original [M, N, K] (M = outer product, N = reduction length,
    K = inner product) contiguous layout; each program handles one m-row and
    one BLOCK_K-slice of K. Per XPU constraints (HARNESS_SUMMARY 2.5/3.6):
    fully UNMASKED loads with the K index clamped in-bounds (no OOB read, no
    garbage), fp32 accumulation, NO tl.sum / where-in-reduce inside the loop
    (pure elementwise add), the clamped tail lanes are zeroed by an arithmetic
    multiply (not tl.where inside a reduce), and stores are masked.
    """
    if tl.constexpr(X.dtype.element_ty == tl.float16) or tl.constexpr(
        X.dtype.element_ty == tl.bfloat16
    ):
        cdtype = tl.float32
    else:
        cdtype = X.dtype.element_ty

    pid_m = ext.program_id(0)
    pid_k = ext.program_id(1)

    k_off = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    k_clamped = tl.minimum(k_off, K - 1)
    k_mask = k_off < K

    p = X + pid_m * N * K + k_clamped
    acc = tl.zeros([BLOCK_K], dtype=cdtype)
    for _ in range(0, N):
        v = tl.load(p).to(cdtype)
        acc += v
        p += K
    mean = (acc / N) * k_mask.to(cdtype)
    tl.store(Out + pid_m * K + k_off, mean, mask=k_mask)


def mean_dim(x, dim, keepdim=False, *, dtype=None):
    logger.debug("GEMS_KUNLUNXIN MEAN_DIM")

    if dtype is None:
        dtype = x.dtype
    if dim is None or dim == () or dim == []:
        out = mean(x, dtype=dtype)
        if keepdim:
            out = out.reshape([1] * x.ndim)
        return out

    shape = list(x.shape)
    dim = [d % x.ndim for d in dim]

    if len(dim) == 1:
        dim0 = dim[0]
        N = shape[dim0]
        M = 1
        for i in shape[:dim0]:
            M *= i
        K = (x.numel() // (M * N)) if (M * N) else 0
        if K > 1 and M > 0 and N > 0:
            x = x.contiguous()
            out_shape = shape[:dim0] + [1] + shape[dim0 + 1 :]
            if N == 1:
                # N=1: mean over a size-1 dim is the identity (same as the
                # historic N==1 fast path; no dim_compress here, so this is a
                # zero-copy view / dtype cast only).
                out = x.to(dtype=dtype).reshape(out_shape)
                if not keepdim:
                    out = out.squeeze(dim=dim0)
                return out
            if x.dtype in (torch.float16, torch.bfloat16):
                try:
                    xv = x.view(M, N, K)
                    ones = torch.full(
                        (M, 1, N), 1.0 / N, dtype=x.dtype, device=x.device
                    )
                    bmm_out = torch.bmm(ones, xv)
                    out = bmm_out.reshape(out_shape)
                    if not keepdim:
                        out = out.squeeze(dim=dim0)
                    return out
                except Exception:
                    pass
            out = torch.empty(out_shape, dtype=dtype, device=x.device)
            BLOCK_K = 128 if K >= 128 else triton.next_power_of_2(K)
            grid = (M, triton.cdiv(K, BLOCK_K))
            with torch_device_fn.device(x.device):
                mean_dim_mid_kernel[grid](
                    x, out, M, N, K, BLOCK_K=BLOCK_K, buffer_size_limit=2048
                )
            if not keepdim:
                out = out.squeeze(dim=dim0)
            return out
    # ------------------------------------------------------------------------

    x = dim_compress(x, dim)
    N = 1
    for i in dim:
        N *= shape[i]
        shape[i] = 1
    M = x.numel() // N if N > 0 else 0

    # Reducing over an empty (size-0) dimension means 0/0 = NaN for every
    # output element, matching torch's reference behavior.
    if N == 0:
        out = torch.full(shape, float("nan"), dtype=dtype, device=x.device)
        if not keepdim:
            out = out.squeeze(dim)
        return out

    # No output rows at all: the result is empty, no computation needed.
    if M == 0:
        out = torch.empty(shape, dtype=dtype, device=x.device)
        if not keepdim:
            out = out.squeeze(dim)
        return out

    # Edge case: M=1 means all dims are reduced → global mean over N elements.
    # mean_dim XPU API does not support M=1.
    if M == 1:
        scalar_out = mean(x, dtype=dtype)  # 0-d tensor
        out = scalar_out.reshape(shape)
        if not keepdim:
            out = out.squeeze(dim)
        return out

    # Edge case: N=1 means reducing a trivial (size-1) dimension.
    # mean of 1 element = that element; just copy with dtype conversion.
    # mean_dim XPU API does not support N=1.
    if N == 1:
        out = x.to(dtype=dtype).reshape(shape)
        if not keepdim:
            out = out.squeeze(dim)
        return out

    out = torch.empty(shape, dtype=dtype, device=x.device)
    grid = lambda META: (triton.cdiv(M, META["BLOCK_M"]),)

    with torch_device_fn.device(x.device):
        mean_dim_kernel[grid](x, out, M, N, buffer_size_limit=2048)
    if not keepdim:
        out = out.squeeze(dim)
    return out
