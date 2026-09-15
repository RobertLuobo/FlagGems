# Kunlunxin (XPU) override of `_conj`.
#
# `_conj(A)` returns the complex conjugate: out = (re, -im).
#
# The generic path (`flag_gems/ops/_conj.py`) launches a kernel that indexes the
# real/imag parts with `base = offsets * 2` and loads `base` (real) and
# `base + 1` (imag) as two SEPARATE stride-2 passes over the interleaved complex
# storage. On XPU a stride-2 gather is discrete access -> the block DMA engine
# degrades to per-element loads and bandwidth collapses. IR baseline
# `harness/perf_ir_4/ir-conj-dev5.log`: gems speedup 0.000-0.21
# ([4096,4096] 52ms, [1024,65536] 202ms), avg 0.0417.
#
# Fix: for a *contiguous* complex tensor the raw storage is one contiguous real
# stream [re0, im0, re1, im1, ...]. A single 1D contiguous kernel copies it while
# negating the odd (imaginary) lanes -> ONE stride-1 block DMA in and out (no
# strided access). This mirrors the already-landed `resolve_conj` fix. The
# `i % 2` test only selects which loaded VALUE to negate; it never appears in an
# address, so the load/store addresses stay affine (stride 1). Non-contiguous
# inputs fall back to the correct generic materialize path.
#
# Perf update (2026-09-10): the fixed BLOCK=8192 was replaced by an adaptive
# rule (`_conj_block_size`: 8192 for n2 < 2**20, 32768 otherwise). On the
# benchmark matrix ([1024,65536] complex64) 8192 achieves ~540 GB/s while
# 32768/65536 reach ~620-650 GB/s (fewer, larger programs -> less
# launch/loop overhead), but for n2 < 2**20 the smaller block wins
# (n2=32768: 6.2us vs 9.4us at 32768).
import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn

logger = logging.getLogger(__name__)


@triton.jit
def _conj_flat_kernel(fin, fout, n2, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    i = pid * BLOCK + tl.arange(0, BLOCK)
    m = i < n2
    x = tl.load(fin + i, mask=m)
    out = tl.where((i % 2) == 1, -x, x)
    tl.store(fout + i, out, mask=m)


@triton.jit
def _conj_flat_copy_kernel(fin, fout, n2, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    i = pid * BLOCK + tl.arange(0, BLOCK)
    m = i < n2
    x = tl.load(fin + i, mask=m)
    tl.store(fout + i, x, mask=m)


def _flatten_storage(input: torch.Tensor) -> torch.Tensor:
    """Return a contiguous 1D view of the interleaved real/imag storage.

    ``view_as_real`` strips the complex dtype, so materializing a
    non-contiguous input goes through the vendor ``copy_`` for a *real* tensor
    (the complex ``copy_`` is rejected by the vendor on this stack).
    """
    rv = torch.view_as_real(input)
    if not rv.is_contiguous():
        rv = rv.contiguous()
    return rv.reshape(-1)


def _conj_block_size(n2: int) -> int:
    # Picked from a BLOCK sweep with the harness do_bench (warmup=1000ms) on
    # the benchmark matrix: for n2 < 2**20 the smaller BLOCK (fewer lanes per
    # program, more programs) wins (e.g. n2=32768: 6.2us vs 9.4us at 32768);
    # for n2 >= 2**20 the larger BLOCK wins by ~15-17% (n2=33.5M: 419us vs
    # 501us at 8192).
    return 32768 if n2 >= 1 << 20 else 8192


def _conj_from_storage(input: torch.Tensor) -> torch.Tensor:
    """Materialize ``conj`` of a tensor whose storage is the physical value."""
    # explicitly contiguous: for a transposed/sliced input, empty_like would
    # otherwise preserve its strided layout and view_as_real(...).reshape(-1)
    # would detach from the output (returns a copy, kernel writes to a temp).
    out = torch.empty_like(input, memory_format=torch.contiguous_format)
    fin = _flatten_storage(input)
    fout = torch.view_as_real(out).reshape(-1)
    n2 = fin.numel()

    BLOCK = _conj_block_size(n2)
    grid = (triton.cdiv(n2, BLOCK),)
    with torch_device_fn.device(input.device):
        _conj_flat_kernel[grid](fin, fout, n2, BLOCK=BLOCK, num_warps=8)

    return out


def _conj(input: torch.Tensor) -> torch.Tensor:
    logger.debug("GEMS_KUNLUNXIN CONJ")
    if not input.is_complex():
        raise RuntimeError("_conj only supports complex tensors")

    if input.is_conj():
        # ATen ``_conj``: conj(conj(x)) = x, i.e. the result is the physical
        # storage of ``input`` (a conj-bit tensor's logical value is
        # conj(storage), and _conj of that is storage).  The lazy conj bit does
        # not change the raw storage, so a plain storage copy (no lane
        # negation) yields exactly the physical value.  This also avoids
        # ``resolve_conj()`` whose complex32 fallback goes through
        # ``torch.complex`` and recurses back into ``_conj`` on this stack.
        out = torch.empty_like(input, memory_format=torch.contiguous_format)
        fin = _flatten_storage(input)
        fout = torch.view_as_real(out).reshape(-1)
        n2 = fin.numel()
        BLOCK = _conj_block_size(n2)
        grid = (triton.cdiv(n2, BLOCK),)
        with torch_device_fn.device(input.device):
            _conj_flat_copy_kernel[grid](fin, fout, n2, BLOCK=BLOCK, num_warps=8)
        return out

    return _conj_from_storage(input)
