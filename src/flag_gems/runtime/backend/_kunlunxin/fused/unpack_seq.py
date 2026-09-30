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

import importlib
import logging
import sys

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)

_gen = importlib.import_module("flag_gems.fused.unpack_seq")


@triton.jit
def _unpack_seq_triton_kernel(
    packed_ptr,  # [B, Lmax, D]
    out_ptr,  # [N, D]
    starts_ptr,  # *i32, [B]
    lengths_ptr,  # *i32, [B]
    B: tl.constexpr,
    Lmax: tl.constexpr,
    D: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_t = tl.program_id(1)
    pid_d = tl.program_id(2)
    off_t = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)
    off_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)

    in_start = tl.load(starts_ptr + pid_b)
    seq_len = tl.load(lengths_ptr + pid_b)

    t_mask = off_t < Lmax
    valid_row = (off_t < seq_len) & t_mask
    out_row = in_start + off_t

    packed_row_ptr = packed_ptr + (pid_b * Lmax + off_t)[:, None] * D + off_d[None, :]
    out_row_ptr = out_ptr + out_row[:, None] * D + off_d[None, :]

    d_mask = off_d[None, :] < D
    packed_vals = tl.load(packed_row_ptr, mask=valid_row[:, None] & d_mask)
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

    lengths_i32 = lengths.int()
    lengths_host = lengths_i32.tolist()
    starts_host = []
    acc = 0
    for length in lengths_host:
        starts_host.append(acc)
        acc += length
    N = acc

    out = torch.empty((N, D), device=packed_tensor.device, dtype=packed_tensor.dtype)
    starts = torch.tensor(
        starts_host, dtype=torch.int32, device=packed_tensor.device
    )

    num_warps = 4
    num_stages = 2
    if block_t == 64 and block_d == 64:
        block_t, block_d, num_warps, num_stages = _gen._select_unpack_seq_config(
            B, Lmax, D, packed_reshaped.element_size()
        )

    grid = (B, triton.cdiv(Lmax, block_t), triton.cdiv(D, block_d))
    _unpack_seq_triton_kernel[grid](
        packed_reshaped,
        out,
        starts,
        lengths_i32,
        B,
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
    (``from flag_gems.fused import unpack_seq_triton``), so the SpecOpRegistrar
    namespace swap cannot reach it. Both the submodule attribute
    (``flag_gems.fused.unpack_seq.unpack_seq_triton``) and the package re-export
    (``flag_gems.fused.unpack_seq_triton``) are patched so a later
    ``from flag_gems.fused import unpack_seq_triton`` binds the XPU version."""
    sub = sys.modules.get("flag_gems.fused.unpack_seq")
    if sub is not None:
        sub.unpack_seq_triton = unpack_seq_triton_xpu
    pkg = sys.modules.get("flag_gems.fused")
    if pkg is not None:
        pkg.unpack_seq_triton = unpack_seq_triton_xpu


_install()
