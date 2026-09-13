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
# Fix 1 (2026-09-09): 2D-tile kernel. Each program owns BM consecutive x1 rows
# and a CHUNK-wide slice of x2 columns, loading the x2 [CHUNK, BLOCK_D] tile
# ONCE via one 2D load and reusing it across all BM rows, then performing BM
# axis-1 reductions of the [CHUNK, BLOCK_D] diff. Measured: [64,128] 0.105->
# 0.0385ms, [128,256] 1.487->0.2930ms. Config: BM=8, CHUNK=16 (D>=256 or
# D==64) / CHUNK=64 (other D).
#
# Fix 2 (2026-09-10): fast-path kernel. Profiling the LLVM of Fix 1 shows every
# load goes through a per-64-float gm2lm_v3 DMA plus mfence, and the
# always-true-at-runtime `d < D` mask (D is a power of two here) forces the
# backend onto the predicated/masked load path. For shapes where D == BLOCK_D
# (D is a power of two, i.e. ALL benchmark shapes) and N % BM == 0, launch
# `_euclidean_dist_kernel_fast` which drops the d-mask and the per-row n_ok
# mask from the loads entirely (keeping only the [CHUNK]-shaped m_mask for the
# M tail), so the backend can emit clean vector loads. Random X2 validation is
# unchanged (maxerr ~2e-6). Measured 2026-09-10 (card 5, fp32, do_bench
# median): [64,128] 45.4->39.5us (0.863->1.008x), [128,256] 299.3->268us
# (0.204->0.231x).
#
# Notes / dead ends (measured, do not resurrect):
#  * tl.dot-based (x1@x2^T + row-norm identity) is FASTER (22/56us) but the
#    backend's tl.dot is numerically broken for ALL sizes (2x2@2x2 maxerr
#    5.4e-4; 16x256@256x16 maxerr 4.5e-2) and triggers an IMA that poisons the
#    device context -> unusable.
#  * A pure-scalar i1 mask on a 1D load (`mask=n_ok`) triggers the backend's
#    'arith.andi op requires the same encoding' bug -> wrong results; always
#    AND the scalar with a vector-shaped mask (general kernel) or drop it
#    (fast kernel, only when N % BM == 0).
#  * 3D diff tiles [BM, CHUNK, BLOCK_D] and transposed [BLOCK_D, CHUNK]
#    orientation (reduction axis=0) fail with OutOfResources (uni_sram).
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
def _euclidean_dist_kernel_fast(
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
    """Fast path: D == BLOCK_D (D is a power of two) and N % BM == 0.

    No d-mask (D == BLOCK_D) and no per-row n_ok mask (N % BM == 0), so the
    backend emits unmasked vector loads. Only the [CHUNK]-shaped m_mask guards
    the M tail (M % CHUNK may be non-zero).
    """
    pid_c = tle.program_id(0)
    pid_r = tle.program_id(1)
    d = tl.arange(0, BLOCK_D)
    m = pid_c * CHUNK + tl.arange(0, CHUNK)
    m_mask = m < M
    x2_vals = tl.load(
        x2_ptr + m[:, None] * stride_x2 + d[None, :],
        mask=m_mask[:, None],
        other=0.0,
    ).to(tl.float32)
    for b in tl.static_range(BM):
        n = pid_r * BM + b
        x1_vals = tl.load(x1_ptr + n * stride_x1 + d).to(tl.float32)
        diff = x1_vals[None, :] - x2_vals
        dist = tl.sqrt(tl.sum(diff * diff, axis=1))
        tl.store(out_ptr + n * stride_out + m, dist, mask=m_mask)


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
    """General path (any D / N): full d-mask and row mask."""
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

    # Fast path: D is a power of two (D == BLOCK_D) and the row dimension is
    # BM-aligned -> drop the always-true d-mask / n_ok masks from the loads so
    # the backend can vectorize them (Fix 2).
    use_fast = (D == BLOCK_D) and (N % BM == 0)
    kernel = _euclidean_dist_kernel_fast if use_fast else _euclidean_dist_kernel

    with torch_device_fn.device(x1.device):
        grid = (triton.cdiv(M, CHUNK), triton.cdiv(N, BM))
        kernel[grid](
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