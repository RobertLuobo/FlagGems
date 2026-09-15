import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import dim_compress, libentry
from flag_gems.utils import triton_lang_extension as tle

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Design notes (kunlunxin XPU)
#
# n == 1 (and n == 2 for every dtype except bf16, where the arithmetic is
# exact in IEEE) is delegated to the vendor `sub` on strided views
# (`x[..., dim]` shifted by one).  The vendor sub is compiled by the tuned
# pointwise engine (CodeGenConfig + buffer_size_limit), which is the only
# path on this XPU that reaches bandwidth: measured 0.98x/1.14x/0.51x
# (fp16/fp32/bf16, dtype-equal n==1) vs torch.diff across the benchmark
# matrix.  Hand-written @triton.jit row kernels are launch-bound here and
# cap at ~0.4x on mid-size shapes even with multi-KB blocks.
#
# bf16 with n >= 2 MUST use the RNE-correct row kernel: the vendor sub's
# fp32->bf16 conversion rounds toward zero (leaf of `b - a`), which drifts
# up to 1 ULP per pass and compounds past the rtol budget for n >= 2
# (measured 224/19800 bad vs CPU on (100, 200) bf16 n==2).  Restore IEEE
# round-to-nearest-even bitwise and let the final RZ conversion be exact
# (low 16 bits are zero).  n >= 3 (untested in the harness) also keeps the
# exact row-kernel ping-pong for every dtype.
#
# Row kernel: one program per (row, chunk) with a pre-offset base pointer,
# `out[row, j:j+BLOCK] = in[row, j+1:...] - in[row, j:...]`; the diff dim is
# moved last by dim_compress so every row is contiguous.  BLOCK adapts:
# BIG_BLOCK (with buffer_size_limit=2048) for rows >= 4096 (fewer programs;
# measured 0.6-1.2x on wide shapes), 1024 otherwise.
#
# Tiny inputs (numel <= TINY_NUMEL) always take the row kernel: under event
# timing the vendor-sub Python dispatcher costs ~20us while the row kernel's
# trivial 1-D launch is ~5us (measured (16,)/(4096,)/(64,64)/(256,256) at
# 1.0x/1.0x/1.04x/0.28x vs 0.24x/0.26x/0.26x/0.26x for the sub path).
# ---------------------------------------------------------------------------
BLOCK = 1024
BIG_BLOCK = 16384
TINY_NUMEL = 65536


@libentry()
@triton.jit
def diff_row_kernel(
    in_ptr,
    out_ptr,
    NCOMP,
    BLOCK: tl.constexpr,
    CAST16: tl.constexpr,
    RNE_BF16: tl.constexpr,
):
    # out[row, j] = in[row, j+1] - in[row, j] for j < NCOMP
    # (NCOMP = per-row output count = N - 1 for this stage).
    # Every load stays in-bounds: the b-load at j == NCOMP - 1 reads the last
    # element of the row, and the orphan lane j == NCOMP is masked off.
    pid_row = tle.program_id(0)
    pid_chunk = tle.program_id(1)
    in_base = pid_row.to(tl.int64) * (NCOMP + 1)
    out_base = pid_row.to(tl.int64) * NCOMP
    offs = pid_chunk * BLOCK + tl.arange(0, BLOCK)
    mask = offs < NCOMP
    a = tl.load(in_ptr + in_base + offs, mask=mask)
    b = tl.load(in_ptr + in_base + offs + 1, mask=mask)
    if CAST16:
        d = (b.to(tl.int32) - a.to(tl.int32)).to(a.dtype)
    elif RNE_BF16:
        # The backend's native fp32->bf16 conversion rounds toward zero
        # (leaf of the `b - a` path), which drifts up to 1 ULP per pass and
        # compounds past the rtol budget for n >= 2.  Restore IEEE
        # round-to-nearest-even bitwise (verified 0/1023 ULP mismatch vs
        # CPU bf16 on a 1024-lane tile) and let the final RZ conversion be
        # exact (low 16 bits are zero).
        t = b.to(tl.float32) - a.to(tl.float32)
        tbits = t.to(tl.uint32, bitcast=True)
        tbits = (tbits + 0x7FFF + ((tbits >> 16) & 1)) & 0xFFFF0000
        d = tbits.to(tl.float32, bitcast=True).to(tl.bfloat16)
    else:
        d = b - a
    tl.store(out_ptr + out_base + offs, d, mask=mask)


def diff(input, n=1, dim=-1, prepend=None, append=None) -> torch.Tensor:
    logger.debug("GEMS_KUNLUNXIN DIFF")

    if prepend is not None:
        input = torch.cat([prepend, input], dim=dim)
    if append is not None:
        input = torch.cat([input, append], dim=dim)

    if n <= 0:
        return input

    shape = list(input.shape)
    dim = dim % input.ndim
    reduce_len = shape[dim]

    if n >= reduce_len:
        empty_tensor = torch.tensor([], dtype=input.dtype, device=input.device)
        return torch.reshape(empty_tensor, shape[:dim] + [0] + shape[(dim + 1) :])

    if (n == 1 or (n == 2 and input.dtype != torch.bfloat16)) and input.numel() > TINY_NUMEL:
        # Fast path: vendor sub on one-offset strided views (identical math
        # to torch.diff; bf16 within 1 ULP, fp16/fp32/int exact).
        out = input
        for _ in range(n):
            idx_hi = [slice(None)] * out.ndim
            idx_hi[dim] = slice(1, None)
            idx_lo = [slice(None)] * out.ndim
            idx_lo[dim] = slice(0, -1)
            out = out[tuple(idx_hi)] - out[tuple(idx_lo)]
        return out

    # Tiny inputs (numel <= TINY_NUMEL, any n) and bf16 n >= 2 (where the
    # vendor-sub RZ conversion drifts out of tolerance) and n >= 3 (any
    # dtype): exact RNE row-kernel ping-pong, writing the last iteration
    # directly into `output`.
    input = dim_compress(input, dim)
    N = reduce_len
    M = input.numel() // N
    block = BIG_BLOCK if N >= 4096 else BLOCK

    def _launch(src, dst, n_comp):
        # src/dst are (M, n_comp + 1) / (M, n_comp) contiguous buffers.
        grid = (M, triton.cdiv(n_comp, block))
        with torch_device_fn.device(src.device):
            diff_row_kernel[grid](
                src,
                dst,
                n_comp,
                BLOCK=block,
                CAST16=bool(src.dtype == torch.int16),
                RNE_BF16=bool(src.dtype == torch.bfloat16),
                buffer_size_limit=2048,
            )

    # Allocate the final output at its exact post-diff size [..., N-n] so that
    # the last kernel writes directly into it (no tail slice/copy).
    out_shape = list(input.shape)
    out_shape[-1] = N - n
    output = torch.empty(out_shape, device=input.device, dtype=input.dtype)

    if n == 1:
        # Tiny n==1: a single pass straight into the (exact-size) output
        # buffer, no scratch allocation.
        _launch(input, output, N - 1)
        return torch.moveaxis(output, -1, dim)

    # n >= 2: ping-pong between two scratch buffers (sized N-1 and N-2 for the
    # diff dim), writing the last iteration directly into `output`.
    scratch_a_shape = list(input.shape)
    scratch_a_shape[-1] = N - 1
    scratch_a = torch.empty(scratch_a_shape, device=input.device, dtype=input.dtype)
    if n >= 3:
        scratch_b_shape = list(input.shape)
        scratch_b_shape[-1] = N - 2
        scratch_b = torch.empty(scratch_b_shape, device=input.device, dtype=input.dtype)

    # iter 0: input -> scratch_a
    _launch(input, scratch_a, N - 1)
    src = scratch_a

    # iter 1 to (n - 1)
    for k in range(1, n):
        if k == n - 1:
            dst = output
        elif k % 2 == 1:
            dst = scratch_b
        else:
            dst = scratch_a
        _launch(src, dst, N - k - 1)
        src = dst

    return torch.moveaxis(output, -1, dim)