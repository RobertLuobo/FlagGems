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
# Fix (no libtuner, fixed BLOCK): drive one program per (row, chunk) with a
# pre-offset base pointer so each program does a purely contiguous 1D block-DMA
# `out[row, j:j+BLOCK] = in[row, j+1:...] - in[row, j:...]`. A fixed BLOCK=8192
# beats an N-adaptive block on XPU (large tiles stay well utilized; smaller
# tiles regress small-N cases). 1D inputs keep the fast flat-DMA path.
BLOCK = 1024


@libentry()
@triton.jit
def diff_flat_kernel(
    in_ptr,
    out_ptr,
    N_COMP,
    BLOCK: tl.constexpr,
    CAST16: tl.constexpr,
):
    # out[p] = in[p+1] - in[p] for p < N_COMP (N_COMP = numel_in - 1).
    # in[.] is a contiguous 1-D stream; the caller guarantees the only
    # out-of-range access (in[N_COMP]) is masked off.
    pid = tle.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N_COMP
    a = tl.load(in_ptr + offs, mask=mask)
    b = tl.load(in_ptr + offs + 1, mask=mask)
    if CAST16:
        d = (b.to(tl.int32) - a.to(tl.int32)).to(a.dtype)
    else:
        d = b - a
    tl.store(out_ptr + offs, d, mask=mask)


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

    # (M, N) contiguous with the diff dimension last.
    input = dim_compress(input, dim)
    N = reduce_len
    M = input.numel() // N
    total = M * N

    if total == 0:
        return torch.empty(
            shape[:dim] + [N - n] + shape[(dim + 1) :],
            dtype=input.dtype,
            device=input.device,
        )

    src = input.reshape(-1)

    def _launch_flat(s, d, n_comp):
        with torch_device_fn.device(s.device):
            diff_flat_kernel[(triton.cdiv(n_comp, FLAT_BLOCK),)](
                s,
                d,
                n_comp,
                BLOCK=FLAT_BLOCK,
                CAST16=bool(s.dtype == torch.int16),
            )

    if n == 1:
        buf = torch.empty(total, device=input.device, dtype=input.dtype)
        _launch_flat(src, buf, total - 1)
    else:
        # Ping-pong between two full-size scratch buffers; stage k writes
        # total-(k+1) valid elements, and consecutive stages are ordered on the
        # current stream (no host sync required).
        bufs = [
            torch.empty(total, device=input.device, dtype=input.dtype)
            for _ in range(2)
        ]
        for k in range(n):
            n_comp = total - (k + 1)
            _launch_flat(src, bufs[k % 2], n_comp)
            src = bufs[k % 2]
        buf = src

    # n >= 2: ping-pong between two scratch buffers, writing the last iteration
    # directly into `output` (size N-n).
    scratch_a_shape = list(input.shape)
    scratch_a_shape[-1] = N - 1
    scratch_a = torch.empty(scratch_a_shape, device=input.device, dtype=input.dtype)
    if n >= 3:
        scratch_b_shape = list(input.shape)
        scratch_b_shape[-1] = N - 2
        scratch_b = torch.empty(scratch_b_shape, device=input.device, dtype=input.dtype)

    _launch(input, scratch_a, N, N - 1, N)
    torch_device_fn.synchronize()
    src, src_stride = scratch_a, N - 1

    for k in range(1, n):
        if k == n - 1:
            dst, dst_stride = output, N - n
        elif k % 2 == 1:
            dst, dst_stride = scratch_b, N - 2
        else:
            dst, dst_stride = scratch_a, N - 1
        _launch(src, dst, src_stride, dst_stride, N - k)
        torch_device_fn.synchronize()
        src, src_stride = dst, dst_stride

    return torch.moveaxis(output, -1, dim)
