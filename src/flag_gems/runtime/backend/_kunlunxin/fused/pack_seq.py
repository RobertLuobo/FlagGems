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


def _select_pack_seq_config(
    B: int,
    Lmax: int,
    D: int,
    element_size: int,
) -> tuple[int, int, int, int]:
    if element_size <= 2 and B >= 512 and Lmax <= 16 and D >= 1024:
        return 128, 256, 4, 2
    return 64, 64, 4, 2


@triton.jit
def _pack_seq_kernel(
    x_ptr,  # [N, D]
    out_ptr,  # [B, Lmax, D]
    lengths_ptr,  # *i32, [B]
    starts_ptr,  # *i32, [B] exclusive prefix sum of lengths
    N: tl.constexpr,
    D: tl.constexpr,
    Lmax: tl.constexpr,
    PAD_VALUE: tl.constexpr,
    PAD_IS_UINT8: tl.constexpr,
    BLOCK_T: tl.constexpr,  # timesteps per program
    BLOCK_D: tl.constexpr,  # features per program
):
    pid_b = tl.program_id(0)  # batch id
    pid_t = tl.program_id(1)  # block over time dimension
    pid_d = tl.program_id(2)  # block over feature dimension
    off_t = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)  # [BLOCK_T]
    off_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)  # [BLOCK_D]

    # Start index is a precomputed exclusive prefix sum loaded as a scalar.
    # This avoids the in-kernel tl.sum(axis=0) reduction whose scalar/vector
    # mix fails TritonXPULegalize on XPU3 ('arith.addi' same-type error).
    in_start = tl.load(starts_ptr + pid_b)
    seq_len = tl.load(lengths_ptr + pid_b)

    t_mask = off_t < Lmax

    in_row = in_start + off_t
    valid_row = (off_t < seq_len) & t_mask

    # Clamp the gather row into [0, N-1] so the load can run WITHOUT a row mask.
    # On XPU3 a row-masked load couples its active-lane footprint to the
    # subsequent store, so a store whose mask (t_mask) is wider than the load
    # mask (valid_row) silently drops the extra rows -> padded rows stay
    # uninitialized. Reading in-bounds garbage for padded rows is harmless
    # because tl.where discards it below.
    in_row_safe = tl.maximum(tl.minimum(in_row, N - 1), 0)

    x_row_ptr = x_ptr + in_row_safe[:, None] * D + off_d[None, :]
    out_row_ptr = out_ptr + (pid_b * Lmax + off_t)[:, None] * D + off_d[None, :]

    d_mask = off_d[None, :] < D
    if PAD_IS_UINT8:
        pad_vals = tl.full([BLOCK_T, BLOCK_D], PAD_VALUE, tl.uint8)
    else:
        pad_vals = tl.full([BLOCK_T, BLOCK_D], PAD_VALUE, tl.float32)

    # Single store per output location. Merge pad and gathered values in
    # registers via tl.where instead of two stores to the same base pointer:
    # on XPU3 the compiler eliminates the first (pad) masked store when a later
    # store aliases the same pointer, leaving padded rows uninitialized.
    x_vals = tl.load(x_row_ptr, mask=d_mask, other=0)
    merged = tl.where(valid_row[:, None], x_vals, pad_vals.to(x_vals.dtype))
    tl.store(out_row_ptr, merged, mask=t_mask[:, None] & d_mask)


def pack_seq_triton_xpu(
    x: torch.Tensor,
    lengths: torch.Tensor,
    pad_value: float | int = -float("inf"),
    block_t: int = 64,
    block_d: int = 64,
) -> torch.Tensor:
    logger.debug("GEMS PACK_SEQ_TRITON")
    is_uint8 = x.dtype == torch.uint8
    if is_uint8:
        assert (
            isinstance(pad_value, int) and 0 <= pad_value <= 255
        ), f"uint8 pack requires an integer pad in [0, 255], got {pad_value!r}"
        pad_constexpr: int | float = int(pad_value)
    else:
        pad_constexpr = float(pad_value)

    original_shape = x.shape
    if len(original_shape) > 2:
        N = original_shape[0]
        x_reshaped = x.reshape(N, -1)
        D = x_reshaped.shape[1]
    else:
        N, D = x.shape
        x_reshaped = x

    lengths_int = lengths.int()
    lengths_list = lengths_int.tolist()
    B = len(lengths_list)
    Lmax = int(max(lengths_list))

    # Exclusive prefix sum of lengths computed on host as index bookkeeping
    # (the packed gather/pad values themselves stay entirely in the kernel).
    starts_list = [0] * B
    acc = 0
    for i in range(B):
        starts_list[i] = acc
        acc += lengths_list[i]
    starts = torch.tensor(starts_list, dtype=torch.int32, device=x.device)

    out = torch.empty((B, Lmax, D), device=x.device, dtype=x.dtype)
    num_warps = 4
    num_stages = 2
    if block_t == 64 and block_d == 64:
        block_t, block_d, num_warps, num_stages = _select_pack_seq_config(
            B, Lmax, D, x_reshaped.element_size()
        )

    grid = (B, triton.cdiv(Lmax, block_t), triton.cdiv(D, block_d))
    _pack_seq_kernel[grid](
        x_reshaped,
        out,
        lengths_int,
        starts,
        N,
        D,
        Lmax,
        PAD_VALUE=pad_constexpr,
        PAD_IS_UINT8=is_uint8,
        BLOCK_T=block_t,
        BLOCK_D=block_d,
        num_warps=num_warps,
        num_stages=num_stages,
    )

    if len(original_shape) > 2:
        out = out.reshape((B, Lmax) + original_shape[1:])

    return out


def _install():
    """Route ``flag_gems.fused.pack_seq_triton`` to the XPU3 implementation.

    ``pack_seq_triton`` is consumed by direct import
    (``from flag_gems.fused import pack_seq_triton`` in tests/test_pack_seq.py),
    so the SpecOpRegistrar namespace swap cannot reach it. Both the submodule
    attribute (``flag_gems.fused.pack_seq.pack_seq_triton``) and the package
    re-export (``flag_gems.fused.pack_seq_triton``, bound at import by
    ``from flag_gems.fused.pack_seq import pack_seq_triton``) are patched here so
    that a later ``from flag_gems.fused import pack_seq_triton`` binds the XPU
    version."""
    sub = sys.modules.get("flag_gems.fused.pack_seq")
    if sub is not None:
        sub.pack_seq_triton = pack_seq_triton_xpu
    pkg = sys.modules.get("flag_gems.fused")
    if pkg is not None:
        pkg.pack_seq_triton = pack_seq_triton_xpu


_install()
