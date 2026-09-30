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

from flag_gems.ops.hash_tensor import _to_u64, _xor_combine
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)


@libentry()
@triton.jit
def hash_flat_kernel(
    x_ptr,
    out_ptr,
    M,
    N,
    row_stride,
    col_stride,
    n_split_size,
    out_row_stride,
    dtype_code: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """XOR-reduce a logical (M, N) matrix along N with a flat 1D reduction.

    Each program handles one output row (pid_m) and the
    [pid_s*n_split_size, +n_split_size) slice of the reduction dim. XOR is
    associative/commutative, so partials are accumulated element-wise into a
    BLOCK_N-wide vector and reduced once at the end. All tiles are 1D, which
    avoids the 2D-tile + tt.reduce path that fails XPU3 core tiling.
    """
    pid_m = tl.program_id(0)
    pid_s = tl.program_id(1)

    if pid_m < M:
        n_start = pid_s * n_split_size
        n_stop = n_start + n_split_size
        row_base = pid_m * row_stride
        acc_vec = tl.zeros([BLOCK_N], dtype=tl.uint64)
        for n0 in range(n_start, n_stop, BLOCK_N):
            n_idx = n0 + tl.arange(0, BLOCK_N)
            n_mask = (n_idx < N) & (n_idx < n_stop)
            offs = row_base + n_idx * col_stride
            v = tl.load(x_ptr + offs, mask=n_mask, other=0)
            u = _to_u64(v, dtype_code)
            u = tl.where(n_mask, u, 0)
            acc_vec = acc_vec ^ u
        acc = tl.reduce(acc_vec, 0, _xor_combine)
        tl.store(out_ptr + (pid_m * out_row_stride + pid_s), acc)


def _tile_shape(N, split_size):
    span = min(N, split_size)
    block_n = max(1, min(triton.next_power_of_2(span), 1024))
    return block_n


def hash_tensor(x, dim, keepdim=False, mode=0):
    logger.debug("GEMS_KUNLUNXIN HASH TENSOR")

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

    # Logical (M, N) matrix: kept axes -> rows, reduced axes -> cols.
    perm = [i for i in range(x.ndim) if i not in dim] + dim
    xperm = x if perm == list(range(x.ndim)) else x.permute(perm)
    xr = xperm.reshape(M, N)
    row_stride = xr.stride(0)
    col_stride = xr.stride(1)

    output = torch.empty(M, dtype=torch.uint64, device=device)

    if M > 0:
        max_splits = (N + 1023) // 1024
        if N >= 32768 and max_splits >= 16:
            num_n_splits = 16
        else:
            num_n_splits = 1

        with torch_device_fn.device(device):
            if num_n_splits > 1:
                n_split_size = (N + num_n_splits - 1) // num_n_splits
                partials = torch.empty(
                    M * num_n_splits, dtype=torch.uint64, device=device
                )
                block_n = _tile_shape(N, n_split_size)
                grid = (M, num_n_splits)
                hash_flat_kernel[grid](
                    xr,
                    partials,
                    M,
                    N,
                    row_stride,
                    col_stride,
                    n_split_size,
                    num_n_splits,
                    dtype_code,
                    block_n,
                )

                # Stage 2: XOR-combine the num_n_splits partials per row.
                block_n2 = _tile_shape(num_n_splits, num_n_splits)
                grid2 = (M, 1)
                hash_flat_kernel[grid2](
                    partials,
                    output,
                    M,
                    num_n_splits,
                    num_n_splits,
                    1,
                    num_n_splits,
                    1,
                    10,
                    block_n2,
                )
            else:
                block_n = _tile_shape(N, N)
                grid = (M, 1)
                hash_flat_kernel[grid](
                    xr,
                    output,
                    M,
                    N,
                    row_stride,
                    col_stride,
                    N,
                    1,
                    dtype_code,
                    block_n,
                )

    if len(output_shape) == 0:
        output = output.reshape([])
    else:
        output = output.reshape(output_shape)

    return output
