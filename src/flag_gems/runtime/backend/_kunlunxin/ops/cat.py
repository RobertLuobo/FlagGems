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
from typing import List, Tuple, Union

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)
# Historical note: the general (any tensor count) path below used to copy
# through a StridedBuffer + pointwise_dynamic `copy_func` (config_ with
# is_cat=True / buffer_size_limit=512).  That is gone: for inner-dim cat the
# path ran a gems `contiguous()` on the permuted source whose strided
# copy_slice kernel faults with an illegal memory access for fp16 shapes in
# the 196608 < numel <= 393216 band (test_concatenate [dtype0-shape15-1]).
# The general path now uses the exact same native `_copy_from` + narrow-view
# recipe as `cat_out` below (and the 3-equal-shape dim-0 S>2**18 branch),
# which cannot overrun the output for any dtype.


@triton.jit
def cat_three_contiguous_kernel(
    inp0,
    inp1,
    inp2,
    out,
    n_elements: tl.constexpr,
    cat_extent: tl.constexpr,
    inner_numel: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    cat_index = (offsets // inner_numel) % (3 * cat_extent)
    source_index = cat_index // cat_extent
    input_offsets = (
        (offsets // (3 * cat_extent * inner_numel)) * (cat_extent * inner_numel)
        + (cat_index % cat_extent) * inner_numel
        + offsets % inner_numel
    )
    values = tl.where(
        source_index == 0,
        tl.load(inp0 + input_offsets, mask=mask),
        tl.where(
            source_index == 1,
            tl.load(inp1 + input_offsets, mask=mask),
            tl.load(inp2 + input_offsets, mask=mask),
        ),
    )
    tl.store(out + offsets, values, mask=mask)


@triton.jit
def cat_out_slab_kernel(
    inp0,
    inp1,
    inp2,
    out,
    S: tl.constexpr,
    row_stride: tl.constexpr,
    PRE: tl.constexpr,
    BLOCK: tl.constexpr,
    HMASK: tl.constexpr,
):
    """Single-launch copy of 3 equal-shape contiguous inputs into `out`.

    The cat dim is decomposed into `PRE` rows of `S` contiguous elements per
    input. Program (pid_x, pid_y) copies the block `pid_x*BLOCK + arange` of
    input `y = pid_y % 3` at row `r = pid_y // 3` into `out` at the matching
    contiguous slab. Everything is a contiguous block DMA (stride-1) on both
    sides; no per-lane div/mod/select is needed (only per-program branching),
    which is what keeps the kernel at bandwidth on XPU.
    """
    pid_x = tl.program_id(0)
    y = tl.program_id(1) % 3
    r = tl.program_id(1) // 3
    offs = pid_x * BLOCK + tl.arange(0, BLOCK)
    if HMASK:
        mask = offs < S
    else:
        mask = None
    if y == 0:
        in_ptr = inp0
    elif y == 1:
        in_ptr = inp1
    else:
        in_ptr = inp2
    values = tl.load(in_ptr + r * S + offs, mask=mask)
    tl.store(out + r * row_stride + y * S + offs, values, mask=mask)


def cat(
    A: Union[Tuple[torch.Tensor, ...], List[torch.Tensor]], dim: int = 0
) -> torch.Tensor:
    logger.debug("GEMS_KUNLUNXIN CAT")

    if len(A) == 0:
        raise RuntimeError("torch.cat(): expected a non-empty list of Tensors")
    if len(A) == 1:
        return A[0]

    # remove torch.Size([0]) tensors
    empty_template = A[0]
    A = list(A)
    for i in range(len(A) - 1, -1, -1):
        if A[i].shape == torch.Size([0]):
            A.pop(i)
    if len(A) == 0:
        return torch.empty_like(empty_template)
    elif len(A) == 1:
        return A[0]

    assert dim >= -A[0].ndim and dim < A[0].ndim, f"Invalid dim: {dim}"
    # Convert negative dim to positive
    dim = dim % A[0].ndim

    # Same rank check
    inp_shapes = [list(_.shape) for _ in A]
    inp0_shape = inp_shapes[0]
    for s in inp_shapes[1:]:
        if len(s) != len(inp0_shape):
            raise RuntimeError(
                f"Tensors must have same number of dimensions: got {len(inp0_shape)} and {len(s)}"
            )
    # Same size check
    for tensor_idx, inp_shape in enumerate(inp_shapes):
        for idx, (common_length, length) in enumerate(zip(inp0_shape, inp_shape)):
            if idx == dim:
                continue
            elif length != common_length:
                raise RuntimeError(
                    f"Sizes of tensors must match except in dimension {dim}. "
                    f"Expected size {common_length} but got size {length} for tensor number "
                    f"{tensor_idx} in the list"
                )

    out_shape = list(inp0_shape)
    out_shape[dim] = sum(s[dim] for s in inp_shapes)

    if len(A) == 3 and all(
        a.is_contiguous() and list(a.shape) == inp0_shape for a in A
    ):
        if dim == 0 and all(a.dtype == A[0].dtype for a in A):
            # dim-0 cat of 3 equal-shape contiguous inputs: everything happens in
            # the front dimension, so each input maps to one CONTIGUOUS slab of
            # the output. Two probed fast paths (XPU 2026-08-16, probe matrix
            # identical to the concat benchmark: 10 shapes x fp16/fp32/bf16/
            # int16/int32):
            #   * S <= 2**18: single-launch constexpr slab kernel. The three
            #     inputs are copied with pure block DMA (one static branch per
            #     program id, no per-lane div/mod) -> 1.2-1.4x at small S and
            #     still >= the 3-copy engine up to S = 2**18.
            #   * S > 2**18: 3x native at::_copy_from into narrow views (the
            #     vendor strided-copy engine saturates bandwidth; the triton
            #     slab kernel drops to 0.56-0.7x there).
            # The generic three-contiguous kernel below is 10-20x slower in this
            # range (0.03-0.4x; per-element div/mod path), so it is no longer
            # used for the dim-0 case.
            S = 1
            for s in inp0_shape:
                S *= s
            out0 = torch.empty(out_shape, dtype=A[0].dtype, device=A[0].device)
            if 0 < S <= 2**18:
                if S <= 4096:
                    BLOCK = 2048
                    num_warps = 2
                else:
                    BLOCK = 8192
                    num_warps = 8
                cat_out_slab_kernel[(triton.cdiv(S, BLOCK), 3)](
                    A[0],
                    A[1],
                    A[2],
                    out0,
                    S=S,
                    row_stride=3 * S,
                    PRE=1,
                    BLOCK=BLOCK,
                    HMASK=(S % BLOCK) != 0,
                    num_warps=num_warps,
                )
            else:
                start = 0
                for a in A:
                    w = a.shape[0]
                    torch.ops.aten._copy_from(a, out0.narrow(0, start, w), False)
                    start += w
            return out0

        if all(a.dtype == A[0].dtype for a in A):
            # dim>0: 3 same-shape contiguous inputs. The old
            # cat_three_contiguous_kernel computed every source index with
            # per-element div/mod: measured on XPU 2026-09-02 it is ~0.03x
            # for big grids ((4096,4096)/(64,512,512) dim=-1: 7.0-8.5s) but
            # sits at the ~7ms measurement floor for tiny grids (<=16K
            # elements), where the 3-call native copy is worse. So:
            #   * n_elements <= 16384: keep the old kernel (at the floor,
            #     matches the dim-0 slab kernel's floor cost).
            #   * n_elements > 16384: write each source directly into the
            #     output through a narrow view using the native ATen
            #     strided-copy engine (`_copy_from` is never overridden by
            #     gems) — the exact same recipe as the `cat_out` inner-dim
            #     branch and the dim-0 S>2**18 branch above.
            out0 = torch.empty(out_shape, dtype=A[0].dtype, device=A[0].device)
            n_elements = out0.numel()
            if n_elements > 16384:
                start = 0
                for a in A:
                    w = a.shape[dim]
                    torch.ops.aten._copy_from(a, out0.narrow(dim, start, w), False)
                    start += w
                return out0
            inner_numel = 1
            for size in inp0_shape[dim + 1 :]:
                inner_numel *= size
            cat_three_contiguous_kernel[(triton.cdiv(n_elements, 1024),)](
                A[0],
                A[1],
                A[2],
                out0,
                n_elements=n_elements,
                cat_extent=inp0_shape[dim],
                inner_numel=inner_numel,
                BLOCK_SIZE=1024,
                num_warps=4,
            )
            return out0

        inner_numel = 1
        for size in inp0_shape[dim + 1 :]:
            inner_numel *= size
        out0 = torch.empty(out_shape, dtype=A[0].dtype, device=A[0].device)
        n_elements = out0.numel()
        cat_three_contiguous_kernel[(triton.cdiv(n_elements, 1024),)](
            A[0],
            A[1],
            A[2],
            out0,
            n_elements=n_elements,
            cat_extent=inp0_shape[dim],
            inner_numel=inner_numel,
            BLOCK_SIZE=1024,
            num_warps=4,
        )
        return out0

    # General path (any tensor count, dim-0 or inner-dim): write each source
    # directly into the fresh output through a narrow view using the native
    # ATen strided-copy engine (`_copy_from` is never overridden by gems) --
    # the exact same recipe as the `cat_out` branches and the 3-equal-shape
    # dim-0 S>2**18 branch.  The previous StridedBuffer + `copy_func` triton
    # path ran a gems `contiguous()` on the permuted source in the inner-dim
    # case, whose strided `copy_slice` kernel (dtype=float16, 196608 < numel
    # <= 393216 -> 19-CTA 16384-lane tile after the 32768-lane guard) faults
    # with an illegal memory access on XPU (repro: test_concatenate
    # [dtype0-shape15-1]).
    out0 = torch.empty(out_shape, dtype=A[0].dtype, device=A[0].device)
    start = 0
    for a in A:
        w = a.shape[dim]
        torch.ops.aten._copy_from(a, out0.narrow(dim, start, w), False)
        start += w
    return out0


def cat_out(
    A: Union[Tuple[torch.Tensor, ...], List[torch.Tensor]],
    dim: int = 0,
    *,
    out: torch.Tensor,
) -> torch.Tensor:
    # cat.out was NOT overridden by kunlunxin before, so it fell back to the
    # generic ops/cat.py::cat_out, whose hand-written raw @triton.jit
    # `cat_copy_func_kernel_4` (fixed BLOCK=1024, no @libentry caching)
    # recompiles per shape/launch -> the IR dump shows ~2750
    # cat_copy_func_kernel_4 modules (43MB / 546K lines). Route cat.out through
    # the exact same tuned StridedBuffer + pointwise_dynamic `copy_func` path
    # that the kunlunxin `cat` override already uses (bounded tiles + autoGrid +
    # libentry cache), just writing into the caller-provided `out`.
    logger.debug("GEMS_KUNLUNXIN CAT_OUT")

    if len(A) == 0:
        raise RuntimeError("torch.cat(): expected a non-empty list of Tensors")

    A = list(A)
    # remove torch.Size([0]) tensors
    for i in range(len(A) - 1, -1, -1):
        if A[i].shape == torch.Size([0]):
            A.pop(i)

    if len(A) == 0:
        out.resize_(0)
        return out

    if len(A) == 1:
        t = A[0]
        out.resize_(t.shape)
        out.copy_(t)
        return out

    assert dim >= -A[0].ndim and dim < A[0].ndim, f"Invalid dim: {dim}"
    # Convert negative dim to positive
    dim = dim % A[0].ndim

    # Same rank check
    inp_shapes = [list(_.shape) for _ in A]
    inp0_shape = inp_shapes[0]
    for s in inp_shapes[1:]:
        if len(s) != len(inp0_shape):
            raise RuntimeError(
                f"Tensors must have same number of dimensions: got {len(inp0_shape)} and {len(s)}"
            )
    # Same size check
    for tensor_idx, inp_shape in enumerate(inp_shapes):
        for idx, (common_length, length) in enumerate(zip(inp0_shape, inp_shape)):
            if idx == dim:
                continue
            elif length != common_length:
                raise RuntimeError(
                    f"Sizes of tensors must match except in dimension {dim}. "
                    f"Expected size {common_length} but got size {length} for tensor number "
                    f"{tensor_idx} in the list"
                )

    out_shape = list(inp0_shape)
    out_shape[dim] = sum(s[dim] for s in inp_shapes)
    if list(out.shape) != out_shape:
        out.resize_(out_shape)

    if (
        len(A) == 3
        and dim == 0
        and out.is_contiguous()
        and all(a.is_contiguous() and a.dtype == A[0].dtype for a in A)
        and all(list(a.shape) == inp0_shape for a in A)
    ):
        # Single-launch fast path for the benchmark-shaped case (3 equal-shape
        # contiguous inputs, dim-0 cat into a contiguous out): each input maps
        # to one contiguous dim-0 slab, so one contiguous-DMA kernel copies all
        # three inputs in a single launch. The launch-does-all allocation makes
        # this up to 2x faster than the 3 native strided copies of the fallback
        # branch for small slabs, and it only loses to the native copies once
        # the slab is large enough that the 3-copy engine reaches saturation,
        # so it is gated to S <= 2**17 (probed on XPU 2026-08-15).
        S = inp0_shape[0]
        for s in inp0_shape[1:]:
            S *= s
        if 0 < S <= 2**17:
            if S <= 4096:
                BLOCK = 2048
                num_warps = 2
            else:
                BLOCK = 8192
                num_warps = 8
            cat_out_slab_kernel[(triton.cdiv(S, BLOCK), 3)](
                A[0],
                A[1],
                A[2],
                out,
                S=S,
                row_stride=3 * S,
                PRE=1,
                BLOCK=BLOCK,
                HMASK=(S % BLOCK) != 0,
                num_warps=num_warps,
            )
            return out

    if dim == 0 and out.is_contiguous():
        # The output consists of contiguous dim-0 slabs, so write each source
        # directly into `out`. A contiguous dim-0 slab is exactly a narrow view
        # of `out`, so use the native ATen strided-copy engine (same as the
        # inner-dim branch below); `_copy_from` is not overridden by gems.
        start = 0
        for a in A:
            w = a.shape[0]
            torch.ops.aten._copy_from(a, out.narrow(0, start, w), False)
            start += w
        return out

    # Inner-dimension concatenation (contiguous output): write each source
    # directly into `out` through a narrow view + the native ATen strided-copy
    # engine (`_copy_from` is not overridden by gems). This avoids the old
    # permute->contiguous->copy->permute round trip in `cat()` — which runs a
    # gems `copy_` on the result that costs ~40ms fixed on XPU — and the
    # intermediate materialization of a full `result` tensor.
    if out.is_contiguous() and all(a.is_contiguous() for a in A):
        start = 0
        for a in A:
            w = a.shape[dim]
            torch.ops.aten._copy_from(a, out.narrow(dim, start, w), False)
            start += w
        return out

    # Non-contiguous output or non-contiguous sources: build the result first,
    # then write it into `out` with the native strided-copy engine (same choice
    # of engine as the branch above; avoids the gems `copy_` override).
    result = cat(A, dim)
    torch.ops.aten._copy_from(result, out, False)
    return out
