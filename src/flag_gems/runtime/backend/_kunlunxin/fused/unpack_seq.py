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
import sys

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)


def _select_unpack_seq_config(
    B: int,
    Lmax: int,
    D: int,
    element_size: int,
) -> tuple[int, int, int, int]:
    if element_size <= 4 and B >= 512 and Lmax <= 16 and D >= 512:
        return 128, 256, 4, 2
    return 64, 64, 4, 2


@triton.jit
def _unpack_seq_kernel(
    packed_ptr,  # [B, Lmax, D]
    out_ptr,  # [N, D]
    lengths_ptr,  # *i32, [B]
    starts_ptr,  # *i32, [B] exclusive prefix sum of lengths
    N: tl.constexpr,
    Lmax: tl.constexpr,
    D: tl.constexpr,
    BLOCK_T: tl.constexpr,  # timesteps per program
    BLOCK_D: tl.constexpr,  # features per program
):
    pid_b = tl.program_id(0)  # batch id
    pid_t = tl.program_id(1)  # block over time dimension
    pid_d = tl.program_id(2)  # block over feature dimension
    off_t = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)  # [BLOCK_T]
    off_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)  # [BLOCK_D]

    # Output start index is a precomputed exclusive prefix sum loaded as a
    # scalar. This avoids the in-kernel tl.sum(axis=0) reduction whose
    # scalar/vector mix fails TritonXPULegalize on XPU3 ('arith.addi' op
    # requires the same type for all operands and results).
    out_start = tl.load(starts_ptr + pid_b)
    seq_len = tl.load(lengths_ptr + pid_b)

    t_mask = off_t < Lmax
    valid_row = (off_t < seq_len) & t_mask

    out_row = out_start + off_t

    # Clamp the gather row into [0, Lmax-1] so the packed load can run WITHOUT
    # a row mask. On XPU3 a row-masked load couples its active-lane footprint
    # to a subsequent store; keeping the load unmasked (only the feature mask
    # applies) decouples it from the valid_row store below. Padded timesteps
    # read in-bounds garbage that the store simply never commits.
    off_t_safe = tl.maximum(tl.minimum(off_t, Lmax - 1), 0)
    # Clamp the destination row into [0, N-1] for pointer safety; invalid rows
    # are discarded by the store mask regardless.
    out_row_safe = tl.maximum(tl.minimum(out_row, N - 1), 0)

    packed_row_ptr = (
        packed_ptr + (pid_b * Lmax + off_t_safe)[:, None] * D + off_d[None, :]
    )
    out_row_ptr = out_ptr + out_row_safe[:, None] * D + off_d[None, :]

    d_mask = off_d[None, :] < D
    packed_vals = tl.load(packed_row_ptr, mask=d_mask, other=0)
    tl.store(out_row_ptr, packed_vals, mask=valid_row[:, None] & d_mask)


def unpack_seq_triton_xpu(
    packed_tensor: torch.Tensor,
    lengths: torch.Tensor,
    block_t: int = 64,
    block_d: int = 64,
) -> torch.Tensor:
    logger.debug("GEMS_KUNLUNXIN UNPACK_SEQ_TRITON")
    original_shape = packed_tensor.shape
    if len(original_shape) > 3:
        B, Lmax = original_shape[:2]
        packed_reshaped = packed_tensor.reshape(B, Lmax, -1)
        D = packed_reshaped.shape[2]
    else:
        B, Lmax, D = packed_tensor.shape
        packed_reshaped = packed_tensor

    lengths_int = lengths.int()
    lengths_list = lengths_int.tolist()
    N = int(sum(lengths_list))

    # Exclusive prefix sum of lengths computed on host as index bookkeeping
    # (the gathered values themselves stay entirely in the kernel).
    starts_list = [0] * B
    acc = 0
    for i in range(B):
        starts_list[i] = acc
        acc += lengths_list[i]
    starts = torch.tensor(starts_list, dtype=torch.int32, device=packed_tensor.device)

    out = torch.empty((N, D), device=packed_tensor.device, dtype=packed_tensor.dtype)
    num_warps = 4
    num_stages = 2
    if block_t == 64 and block_d == 64:
        block_t, block_d, num_warps, num_stages = _select_unpack_seq_config(
            B, Lmax, D, packed_reshaped.element_size()
        )

    grid = (B, triton.cdiv(Lmax, block_t), triton.cdiv(D, block_d))
    _unpack_seq_kernel[grid](
        packed_reshaped,
        out,
        lengths_int,
        starts,
        N,
        Lmax,
        D,
        BLOCK_T=block_t,
        BLOCK_D=block_d,
        num_warps=num_warps,
        num_stages=num_stages,
    )

    if len(original_shape) > 3:
        output_shape = (N,) + original_shape[2:]
        out = out.reshape(output_shape)

    return out


def _install():
    """Route ``flag_gems.fused.unpack_seq_triton`` to the XPU3 implementation.

    ``unpack_seq_triton`` is consumed by direct import
    (``from flag_gems.fused import unpack_seq_triton`` in
    tests/test_unpack_seq.py), so the SpecOpRegistrar namespace swap cannot
    reach it. Both the submodule attribute
    (``flag_gems.fused.unpack_seq.unpack_seq_triton``) and the package
    re-export (``flag_gems.fused.unpack_seq_triton``, bound at import by
    ``from flag_gems.fused.unpack_seq import unpack_seq_triton``) are patched
    here so that a later ``from flag_gems.fused import unpack_seq_triton`` binds
    the XPU version."""
    sub = sys.modules.get("flag_gems.fused.unpack_seq")
    if sub is not None:
        sub.unpack_seq_triton = unpack_seq_triton_xpu
    pkg = sys.modules.get("flag_gems.fused")
    if pkg is not None:
        pkg.unpack_seq_triton = unpack_seq_triton_xpu


_install()
