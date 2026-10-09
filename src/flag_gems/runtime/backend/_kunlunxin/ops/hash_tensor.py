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

import logging

import torch
import triton
import triton.language as tl

from flag_gems.utils import libentry

logger = logging.getLogger(__name__)


@triton.jit
def _xor_combine(a, b):
    return a ^ b


@triton.jit
def _to_u64(val, dtype_code: tl.constexpr):
    if dtype_code == 0:  # float32 -> float64 bits
        u = val.to(tl.float64).to(tl.uint64, bitcast=True)
    elif dtype_code == 1:  # float16 -> float64 bits
        u = val.to(tl.float64).to(tl.uint64, bitcast=True)
    elif dtype_code == 2:  # bfloat16 -> float64 bits
        u = val.to(tl.float64).to(tl.uint64, bitcast=True)
    elif dtype_code == 3:  # float64 bits
        u = val.to(tl.uint64, bitcast=True)
    elif dtype_code == 4:  # int64 bits
        u = val.to(tl.uint64, bitcast=True)
    elif dtype_code == 5:  # int32 -> sign-extend int64 -> bits
        u = val.to(tl.int64).to(tl.uint64, bitcast=True)
    elif dtype_code == 6:  # int16 -> sign-extend int64 -> bits
        u = val.to(tl.int64).to(tl.uint64, bitcast=True)
    elif dtype_code == 7:  # int8 -> sign-extend int64 -> bits
        u = val.to(tl.int64).to(tl.uint64, bitcast=True)
    elif dtype_code == 8:  # uint8 -> zero-extend
        u = val.to(tl.uint64)
    elif dtype_code == 9:  # bool -> zero-extend
        u = val.to(tl.uint64)
    else:  # 10: uint64 partials passthrough
        u = val
    return u


@libentry()
@triton.jit
def hash_row_kernel(
    x_ptr,
    out_ptr,
    N,
    row_stride,
    col_stride,
    n_split_size,
    out_row_stride,
    dtype_code: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_s = tl.program_id(1)

    n_start = pid_s * n_split_size
    n_stop = n_start + n_split_size

    row_base = pid_m * row_stride
    acc = tl.full((), 0, tl.uint64)
    for n0 in range(n_start, n_stop, BLOCK_N):
        n_idx = n0 + tl.arange(0, BLOCK_N)
        n_mask = (n_idx < N) & (n_idx < n_stop)
        offs = row_base + n_idx * col_stride
        v = tl.load(x_ptr + offs, mask=n_mask, other=0)
        u = _to_u64(v, dtype_code)
        u = tl.where(n_mask, u, 0)
        acc ^= tl.reduce(u, 0, _xor_combine)

    out_off = pid_m * out_row_stride + pid_s
    store_lane = tl.arange(0, 8)
    store_mask = store_lane < 1
    acc_vec = acc + tl.zeros([8], tl.uint64)
    tl.store(out_ptr + out_off + store_lane, acc_vec, mask=store_mask)


def hash_tensor(x, dim, keepdim=False, mode=0):
    """XOR reduction of 64-bit bit patterns along specified dimensions.

    Kunlunxin (XPU3) override: the generic implementation loads a 2-D
    [BLOCK_M, BLOCK_N] tile and tree-reduces along axis 1, which trips the
    TritonXPUCoreTiling pass. Here each program owns exactly one output row and
    performs a 1-D (trailing-axis) reduction, avoiding the 2-D tile entirely.
    """
    logger.debug("GEMS_KUNLUNXIN HASH TENSOR")
    logger.debug("GEMS HASH TENSOR")

    dtype = x.dtype
    device = x.device

    dtype_map = {
        torch.float32: 0,
        torch.float16: 1,
        torch.bfloat16: 2,
        torch.float64: 3,
        torch.int64: 4,
        torch.int32: 5,
        torch.int16: 6,
        torch.int8: 7,
        torch.uint8: 8,
        torch.bool: 9,
    }
    dtype_code = dtype_map[dtype]

    if dim is None or (isinstance(dim, (list, tuple)) and len(dim) == 0):
        dim = list(range(x.ndim))
    elif isinstance(dim, int):
        dim = [dim]
    else:
        dim = list(dim)

    if x.ndim == 0:
        dim = []

    dim = [(d % x.ndim) if x.ndim > 0 else 0 for d in dim]
    dim = sorted(set(dim))

    output_shape = []
    for i in range(x.ndim):
        if i not in dim:
            output_shape.append(x.shape[i])
        elif keepdim:
            output_shape.append(1)

    reduce_size = 1
    for d in dim:
        reduce_size *= x.shape[d]

    output_numel = 1
    for s in output_shape:
        output_numel *= s

    M = output_numel
    N = reduce_size

    perm = [i for i in range(x.ndim) if i not in dim] + dim
    xperm = x if perm == list(range(x.ndim)) else x.permute(perm)
    xr = xperm.reshape(M, N)
    row_stride = xr.stride(0)
    col_stride = xr.stride(1)

    output = torch.empty(M, dtype=torch.uint64, device=device)

    if M > 0:
        BLOCK_N = min(triton.next_power_of_2(N), 1024) if N > 0 else 1
        BLOCK_N = max(BLOCK_N, 1)

        max_splits = (N + BLOCK_N - 1) // BLOCK_N
        if N >= 32768 and max_splits >= 16 and M < 64:
            num_n_splits = 16
        else:
            num_n_splits = 1

        if num_n_splits > 1:
            n_split_size = (N + num_n_splits - 1) // num_n_splits
            partials = torch.empty(M * num_n_splits, dtype=torch.uint64, device=device)
            grid = (M, num_n_splits)
            hash_row_kernel[grid](
                xr,
                partials,
                N,
                row_stride,
                col_stride,
                n_split_size,
                num_n_splits,
                dtype_code,
                BLOCK_N,
            )

            N2 = num_n_splits
            BLOCK_N2 = triton.next_power_of_2(N2)
            grid2 = (M, 1)
            hash_row_kernel[grid2](
                partials,
                output,
                N2,
                num_n_splits,
                1,
                N2,
                1,
                10,
                BLOCK_N2,
            )
        else:
            grid = (M, 1)
            hash_row_kernel[grid](
                xr,
                output,
                N,
                row_stride,
                col_stride,
                N,
                1,
                dtype_code,
                BLOCK_N,
            )

    if len(output_shape) == 0:
        output = output.reshape([])
    else:
        output = output.reshape(output_shape)

    return output
