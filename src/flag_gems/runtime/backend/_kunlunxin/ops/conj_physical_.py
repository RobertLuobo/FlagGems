# Kunlunxin (XPU) override of `conj_physical_` (in-place).
#
# `conj_physical_(A)` for complex input negates the imaginary parts in place:
# out = (re, -im); for real input it is the identity.
#
# The generic path (`flag_gems/ops/conj_physical_.py`) already uses the flat
# interleaved-lane trick, but launches with BLOCK_SIZE=1024. On XPU the
# launch/block overhead dominates: measured gems latency (complex64, in-place,
# i.e. 1 full read + 1 full write of 2*numel floats) drops ~3.5-5.0x by moving
# to BLOCK=8192/num_warps=8:
#   [2048,2048]  0.4845ms -> 0.1052ms
#   [128,512,256] 1.9156ms -> 0.3836ms
#   [512,1024]   0.0654ms -> 0.0185ms
# This mirrors the already-landed `_conj` override (same flat kernel design,
# same tuned config). The `i % 2` test only selects which loaded VALUE to
# negate; it never appears in an address, so the load/store addresses stay
# affine (stride 1, one block DMA in and out). Note: the torch native
# reference has no working complex64 conj_physical_ kernel on XPU
# (`CUDA error: invalid device function`), so complex dtypes are excluded
# from the benchmark comparison for this vendor (see benchmark file).
import logging

import torch
import triton
import triton.language as tl

from flag_gems.utils import libentry

logger = logging.getLogger(__name__)


@libentry()
@triton.jit
def conj_physical__kernel(ptr, n_real_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_real_elements
    x = tl.load(ptr + offsets, mask=mask)
    # even offsets are real parts (kept), odd offsets are imag parts (negated)
    y = tl.where((offsets % 2) == 1, -x, x)
    tl.store(ptr + offsets, y, mask=mask)


def conj_physical_(input: torch.Tensor) -> torch.Tensor:
    logger.debug("GEMS_KUNLUNXIN CONJ_PHYSICAL_")
    if not input.is_complex():
        return input

    if not input.is_contiguous():
        raise RuntimeError(
            "conj_physical_ only supports contiguous tensors. "
            "Please call .contiguous() before this operation."
        )

    # view complex64/128 as a flat float32/64 buffer of 2 * numel elements
    flat = torch.view_as_real(input).view(-1)
    n = flat.numel()

    BLOCK_SIZE = 8192
    grid = (triton.cdiv(n, BLOCK_SIZE),)
    conj_physical__kernel[grid](
        flat, n, BLOCK_SIZE=BLOCK_SIZE, num_warps=8
    )

    return input