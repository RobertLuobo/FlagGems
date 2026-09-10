# Kunlunxin (XPU) override of _euclidean_dist.
#
# _euclidean_dist(x1, x2) computes pairwise Euclidean distances:
#   out[i, j] = ||x1[i] - x2[j]||_2 ,  x1:(N,D)  x2:(M,D)  out:(N,M)
#
# The generic KernelGen kernel (src/flag_gems/ops/_euclidean_dist.py) launches
# grid=(N, M): ONE program per output element, each re-loading the full D-length
# rows of BOTH x1 and x2, doing a D-reduction and storing a single scalar. On XPU
# that is launch-bound (N*M tiny programs) + O(N*M*D) redundant x1 reloads ->
# [128,256]x[128,256] (=32768 programs) gems latency ~3.0ms (speedup ~0.02),
# [64,128] ~0.105ms (speedup ~0.39 at the benchmark shapes).
#
# Fix (this file): the XPU backend lowers 2D tiles and axis-1 reductions well
# (1D per-output kernels are ~5x slower: [128,256] 1.54-2.09ms), so each program
#   * owns BM consecutive x1 rows and a CHUNK-wide slice of x2 columns,
#   * loads the x2 [CHUNK, BLOCK_D] tile ONCE via one 2D load and reuses it
#     across all BM rows (kills the redundant x1/x2 reloads),
#   * performs BM axis-1 reductions of the [CHUNK, BLOCK_D] diff (XPU lowers
#     this to a single 2D load + 2D reduction, no per-row serial loop).
# Measured (2026-09-09, card 7, fp32, iters=200):
#   [64,128]x[64,128]:  0.105ms (HEAD)  ->  0.0385ms  (~2.7x)
#   [128,256]x[128,256]: 1.487ms (HEAD)  ->  0.2930ms  (~5.1x)
# Config: BM=8, CHUNK=16 (D>=256 or D==64) / CHUNK=64 (other D), num_warps=4
# (swept). The [CHUNK,BLOCK_D] tile is kept <= ~32KB: BM=16/CHUNK=32 at D=128
# triggers an illegal-memory-access that poisons the device context, and the
# exactly-square [64,64] tile (CHUNK=D=64) fails TritonXPUCoreTiling.
import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as tle

logger = logging.getLogger("flag_gems").getChild(__name__.lstrip("."))


@libentry()
@triton.jit
def _euclidean_dist_kernel(
    x1_ptr,
    x2_ptr,
    out_ptr,
    N,
    M,
    D,
    stride_x1,
    stride_x2,
    stride_out,
    CHUNK: tl.constexpr,
    BM: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_c = tle.program_id(0)
    pid_r = tle.program_id(1)
    d = tl.arange(0, BLOCK_D)
    d_mask = d < D
    m = pid_c * CHUNK + tl.arange(0, CHUNK)
    m_mask = m < M
    x2_vals = tl.load(
        x2_ptr + m[:, None] * stride_x2 + d[None, :],
        mask=m_mask[:, None] & d_mask[None, :],
        other=0.0,
    ).to(tl.float32)
    for b in tl.static_range(BM):
        n = pid_r * BM + b
        n_ok = n < N
        x1_vals = tl.load(
            x1_ptr + n * stride_x1 + d,
            mask=n_ok & d_mask,
            other=0.0,
        ).to(tl.float32)
        diff = x1_vals[None, :] - x2_vals
        dist = tl.sqrt(tl.sum(diff * diff, axis=1))
        tl.store(
            out_ptr + n * stride_out + m,
            dist,
            mask=m_mask & n_ok,
        )


def _euclidean_dist(x1, x2):
    logger.debug("GEMS_KUNLUNXIN _EUCLIDEAN_DIST")

    assert x1.ndim == 2, "x1 must be a 2D tensor"
    assert x2.ndim == 2, "x2 must be a 2D tensor"
    assert x1.shape[1] == x2.shape[1], "x1 and x2 must have the same number of columns"

    N, D = x1.shape
    M = x2.shape[0]

    x1 = x1.contiguous()
    x2 = x2.contiguous()
    output = torch.empty((N, M), dtype=x1.dtype, device=x1.device)

    if N == 0 or M == 0:
        return output

    BM = 8
    BLOCK_D = min(triton.next_power_of_2(D), 1024)
    # Keep the working 2D x2 tile within ~16K elements (~64KB fp32); larger
    # tiles can trigger an IMA that poisons the device context on this backend.
    # CHUNK=64 with BLOCK_D=64 (exactly-square [64,64] tile) fails
    # TritonXPUCoreTiling, so D==64 uses CHUNK=16 instead.
    max_chunk = max(1, 16384 // max(BLOCK_D, 1))
    if D >= 256 or D == 64:
        CHUNK = 16
    else:
        CHUNK = 64
    CHUNK = min(CHUNK, max_chunk)

    with torch_device_fn.device(x1.device):
        grid = (triton.cdiv(M, CHUNK), triton.cdiv(N, BM))
        _euclidean_dist_kernel[grid](
            x1,
            x2,
            output,
            N,
            M,
            D,
            x1.stride(0),
            x2.stride(0),
            output.stride(0),
            CHUNK=CHUNK,
            BM=BM,
            BLOCK_D=BLOCK_D,
        )

    return output
