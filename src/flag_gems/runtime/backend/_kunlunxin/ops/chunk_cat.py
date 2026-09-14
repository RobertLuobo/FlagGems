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
#
# Kunlunxin (TritonXPU) specialization of _chunk_cat.
#
# Why this override exists (2026-09-05, XPU 7)
# --------------------------------------------
# flag_gems.use_gems() registers a Python "slice.Tensor" implementation on the
# ATen dispatcher, and on this torch build that Python impl is called with
# 4 positional arguments (self, dim, start, stop) instead of the declared
# (self, dim, start, end, step). As a result any NON-FULL slice executed
# inside a use_gems() context raises
#     TypeError: slice() missing 1 required positional argument: 'step'
# The generic implementation in src/flag_gems/ops/_chunk_cat.py hits that
# defect in its multi-tensor path:
#   - dst_flat[dst_start : dst_start + len(src_flat)] = src_flat   (dim == 0)
#   - output[tuple(output_idx)] = chunk_data  (dim > 0, contains partial
#     slices too)
# so every multi-tensor call died with TypeError (functional baseline:
# 3 failed / 9 passed, all of test_chunk_cat_multiple_tensors).
#
# This file re-implements the multi-tensor path with slice-free construction
# only (chunk / cat / stack / reshape / zeros -- all of which dispatch to
# native or known-good implementations) and reuses the proven generic
# single-tensor Triton kernels unchanged, so the benchmarked single-tensor
# path does not move.
#
# Reference layout (verified empirically against the CPU ATen reference):
#   chunk_size      = ceil(dim_size / num_chunks)
#   interleaved[i]  = cat([tensor.chunk(i) for tensor in tensors], dim=dim)
#   out             = stack([x.reshape(shape[:dim] + [-1])
#                            for x in interleaved], dim=dim)

import logging
from typing import List

import torch

from flag_gems.ops._chunk_cat import _chunk_cat_triton

logger = logging.getLogger(__name__)


def _chunk_cat_multi_tensor(
    tensors: List[torch.Tensor], dim: int, num_chunks: int, ndim: int
) -> torch.Tensor:
    dim = dim % ndim
    dim_size = tensors[0].shape[dim]
    chunk_size = (dim_size + num_chunks - 1) // num_chunks
    dtype = tensors[0].dtype
    device = tensors[0].device
    num_tensors = len(tensors)

    # 1. Split each tensor into `num_chunks` pieces and zero-pad tail pieces
    #    up to `chunk_size` (matches torch.chunk + zero padding).
    padded = []
    for tensor in tensors:
        padded_chunks = []
        for chunk in torch.chunk(tensor, num_chunks, dim=dim):
            if chunk.shape[dim] < chunk_size:
                pad_shape = list(chunk.shape)
                pad_shape[dim] = chunk_size - chunk.shape[dim]
                chunk = torch.cat(
                    [chunk, torch.zeros(pad_shape, dtype=dtype, device=device)],
                    dim=dim,
                )
            padded_chunks.append(chunk)
        # torch.chunk may return fewer than `num_chunks` pieces (it splits
        # into ceil(dim_size / chunk_size) pieces); the reference pads the
        # missing trailing chunks with zeros.
        if len(padded_chunks) < num_chunks:
            zero_shape = list(tensor.shape)
            zero_shape[dim] = chunk_size
            padded_chunks.extend(
                [
                    torch.zeros(zero_shape, dtype=dtype, device=device)
                    for _ in range(num_chunks - len(padded_chunks))
                ]
            )
        padded.append(padded_chunks)

    # 2. Interleave: for each chunk index, concatenate every tensor's chunk at
    #    that index, then flatten all dimensions after `dim`.
    interleaved = [
        torch.cat([padded[t][i] for t in range(num_tensors)], dim=dim)
        for i in range(num_chunks)
    ]
    flattened = [x.reshape(list(x.shape[:dim]) + [-1]) for x in interleaved]
    return torch.stack(flattened, dim=dim)


def chunk_cat(tensors: List[torch.Tensor], dim: int, num_chunks: int) -> torch.Tensor:
    """_chunk_cat on Kunlunxin.

    Single-tensor inputs reuse the generic Triton kernels
    (``_chunk_cat_triton``); multi-tensor inputs are assembled with
    slice-free operations only (see module docstring).
    """
    if len(tensors) == 0:
        raise ValueError("_chunk_cat(): expected a non-empty list of Tensors")

    if num_chunks <= 0:
        raise ValueError(f"_chunk_cat(): num_chunks must be positive, got {num_chunks}")

    ndim = tensors[0].ndim
    if ndim == 0:
        raise ValueError("_chunk_cat(): expected tensors with at least 1 dimension")
    if dim < -ndim or dim >= ndim:
        raise IndexError(
            f"_chunk_cat(): dim {dim} out of range for tensor with {ndim} dimensions"
        )

    if len(tensors) == 1:
        # The generic Triton kernels are only equivalent to the ATen reference
        # (torch.chunk boundary distribution) when `dim_size` is a multiple of
        # `num_chunks`; uneven division and ndim > 2 are routed to the
        # slice-free assembly instead.
        dim = dim % ndim
        if tensors[0].ndim <= 2 and tensors[0].shape[dim] % num_chunks == 0:
            return _chunk_cat_triton(tensors[0], dim, num_chunks)
        return _chunk_cat_multi_tensor(tensors, dim, num_chunks, ndim)

    return _chunk_cat_multi_tensor(tensors, dim, num_chunks, ndim)
