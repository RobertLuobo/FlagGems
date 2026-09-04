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

import torch
import triton
import triton.language as tl

from flag_gems.ops.scaled_mm import (
    _check_inputs,
    _normalize_bias,
    _normalize_scale,
    _resolve_out_dtype,
)
from flag_gems.runtime import torch_device_fn
from flag_gems.runtime.backend._kunlunxin.ops.mm import (
    _restore_matmul_fast_mode,
    _set_matmul_fast_mode,
)
from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as ext

logger = logging.getLogger(__name__)

# M*N*K budget: above this, fp16/bf16 inputs use the fused Triton kernel
# below; everything else uses the native-mm chain.  Measured on XPU
# (do_bench median, transposed mat2, 2026-09-04): the native XDNN mm chain
# is faster for small/medium shapes (its mm needs no host-side padding, and
# each torch.zeros/_copy_from costs ~10-20us on this stack) and for fp32 at
# >= 1024^3 (0.076 vs 0.096ms, 0.123 vs 0.265ms), while the fused kernel
# wins for large fp16/bf16 GEMMs where the dot dominates: 256x4096x4096
# fp16 0.394->0.151ms (2.6x) / 0.386->0.268ms (1.4x), bf16 0.384->0.220ms
# (1.7x).  The budget sits between 1024^3 (2^30, chain wins) and
# 256x4096x4096 (2^32, fused wins).
_FUSED_MM_BUDGET = 1 << 30

# XPU tile probe (see _kunlunxin/ops/mm.py): 256-tile is the floor for
# M, N > 512; small shapes are launch-bound and prefer the 128-tile 4-warp
# config.  num_stages stays at the backend default (2).
GROUP_M = 8


def _block_m(M):
    return 128 if M <= 512 else 256


def _block_n(N):
    return 128 if N <= 512 else 256


def _block_k(M, N):
    if M <= 512 and N <= 512:
        return 128
    return 256


@libentry()
@triton.jit
def _scaled_mm_kernel(
    A,
    B,
    ScaleA,
    ScaleB,
    Bias,
    C,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    ACC_DTYPE: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    # Fused C = (A @ B) * sa[row] * sb[col] (+ bias), all in a single kernel.
    #
    # XPU safety (mirrors _kunlunxin/ops/mm.py): TritonXPU mis-lowers both
    # masked loads and masked stores whose addresses leave the allocation
    # (intermittent "illegal memory access", status 700, and silently wrong
    # values on fp16/bf16), so there is no load/store mask at all.  The host
    # pads A/B to K_pad = cdiv(K, BLOCK_K)*BLOCK_K, C to (M_pad, N_pad) =
    # (cdiv(M, BLOCK_M)*BLOCK_M, cdiv(N, BLOCK_N)*BLOCK_N) and the scale/bias
    # buffers to (M_pad,)/(N_pad,); row/col indices wrap through % for the
    # A/B/C tiles only, and the garbage pad rows/cols are excluded by the
    # host-side view.
    pid = ext.program_id(0)
    grid_m = tl.cdiv(M, BLOCK_M)
    grid_n = tl.cdiv(N, BLOCK_N)
    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // group_size

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    ram = rm % M
    rbn = rn % N
    rk = tl.arange(0, BLOCK_K)

    A = A + (ram[:, None] * stride_am + rk[None, :] * stride_ak)
    B = B + (rk[:, None] * stride_bk + rbn[None, :] * stride_bn)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_DTYPE)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        a = tl.load(A)
        b = tl.load(B)
        acc += tl.dot(a, b, out_dtype=ACC_DTYPE, allow_tf32=False)
        A += BLOCK_K * stride_ak
        B += BLOCK_K * stride_bk

    acc = acc.to(tl.float32)

    sa = tl.load(ScaleA + rm)
    sb = tl.load(ScaleB + rn)
    acc = acc * sa[:, None] * sb[None, :]

    if HAS_BIAS:
        bias = tl.load(Bias + rn).to(tl.float32)
        acc = acc + bias[None, :]

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    C = C + (rm[:, None] * stride_cm + rn[None, :] * stride_cn)
    tl.store(C, acc.to(C.dtype.element_ty))


def _scaled_mm_fused(a, mat2, scale_a, scale_b, bias, out, M, K, N):
    """Fused kernel launch on (possibly padded) row-major buffers.

    Mirrors _kunlunxin/ops/mm.py ``_pad_k``/``_padded_or_direct``: K- and
    C-padding make the unmasked loads/stores in-bounds by construction; the
    copy-back goes through the native ``_copy_from`` engine (gems never
    overrides it, so strided views such as ``c[:M, :N]`` are safe).
    """
    blk_m = _block_m(M)
    blk_n = _block_n(N)
    blk_k = _block_k(M, N)
    kp = triton.cdiv(K, blk_k) * blk_k
    if (a.stride(0), a.stride(1)) != (K, 1) or kp != K:
        ap = torch.zeros((M, kp), device=a.device, dtype=a.dtype)
        torch.ops.aten._copy_from(a, ap[:, :K], False)
        a = ap
    if (mat2.stride(0), mat2.stride(1)) != (N, 1) or kp != K:
        bp = torch.zeros((kp, N), device=mat2.device, dtype=mat2.dtype)
        torch.ops.aten._copy_from(mat2, bp[:K, :], False)
        mat2 = bp

    num_warps = 4 if (M <= 512 and N <= 512) else 8

    # Scale/bias are read by plain (unmasked, un-wrapped) row/col indices, so
    # the host pads them to the same extents as the C buffer.  Scalar scales
    # are broadcast to the full row/col extent.
    def _pad_scale(scale, extent):
        if scale.numel() == 1:
            return torch.full((extent,), float(scale), dtype=torch.float32, device=scale.device)
        buf = torch.empty((extent,), dtype=torch.float32, device=scale.device)
        torch.ops.aten._copy_from(scale, buf[: scale.numel()], False)
        return buf

    mp = triton.cdiv(M, blk_m) * blk_m
    np_ = triton.cdiv(N, blk_n) * blk_n
    sa_buf = _pad_scale(scale_a, mp)
    sb_buf = _pad_scale(scale_b, np_)
    bias_buf = None
    if bias is not None:
        bias_buf = torch.zeros((np_,), dtype=torch.float32, device=bias.device)
        torch.ops.aten._copy_from(bias, bias_buf[: N], False)

    def launch(c):
        grid = (
            triton.cdiv(M, blk_m) * triton.cdiv(N, blk_n),
        )
        _scaled_mm_kernel[grid](
            a,
            mat2,
            sa_buf,
            sb_buf,
            bias_buf if bias_buf is not None else a,
            c,
            M,
            N,
            kp,
            a.stride(0),
            a.stride(1),
            mat2.stride(0),
            mat2.stride(1),
            c.stride(0),
            c.stride(1),
            ACC_DTYPE=tl.float32,
            HAS_BIAS=bias_buf is not None,
            BLOCK_M=blk_m,
            BLOCK_N=blk_n,
            BLOCK_K=blk_k,
            GROUP_M=GROUP_M,
            num_warps=num_warps,
        )

    saved = _set_matmul_fast_mode(a.dtype, M, N, kp)
    try:
        if mp == M and np_ == N and out.stride(1) == 1:
            with torch_device_fn.device(a.device):
                launch(out)
            return out
        c = torch.empty((mp, np_), device=a.device, dtype=out.dtype)
        with torch_device_fn.device(a.device):
            launch(c)
        torch.ops.aten._copy_from(c[:M, :N], out, False)
    finally:
        _restore_matmul_fast_mode(saved)
    return out


def _scaled_mm_chain(self, mat2, scale_a, scale_b, bias, out, M, K, N):
    """Default path: native mm (no f32 upcast for fp32 inputs) plus
    elementwise scale/bias and the final cast/copy.  Measured faster than
    the fused Triton kernel for small/medium shapes (native XDNN mm has no
    host-padding overhead) and for fp32 at >= 1024^3; see _FUSED_MM_BUDGET."""
    result = torch.mm(self.to(torch.float32), mat2.to(torch.float32))
    if scale_a.numel() == 1:
        result = result * scale_a
    else:
        result = result * scale_a.reshape(M, 1)
    if scale_b.numel() == 1:
        result = result * scale_b
    else:
        result = result * scale_b.reshape(1, N)
    if bias is not None:
        result = result + bias
    out.copy_(result.to(out.dtype))
    return out


def _scaled_mm_impl(self, mat2, scale_a, scale_b, bias, out_dtype, out):
    _check_inputs(self, mat2)
    M, K = self.shape
    N = mat2.shape[1]
    output_dtype = _resolve_out_dtype(self, out_dtype, out)

    if out is None:
        out = torch.empty((M, N), dtype=output_dtype, device=self.device)
    elif out.shape != (M, N):
        raise RuntimeError("Incompatible output shape")
    if M == 0 or N == 0:
        return out

    scale_a, _ = _normalize_scale(scale_a, M, is_left_scale=True)
    scale_b, _ = _normalize_scale(scale_b, N, is_left_scale=False)
    bias = _normalize_bias(bias, N)

    if self.dtype != torch.float32 and M * N * K > _FUSED_MM_BUDGET:
        return _scaled_mm_fused(self, mat2, scale_a, scale_b, bias, out, M, K, N)
    return _scaled_mm_chain(self, mat2, scale_a, scale_b, bias, out, M, K, N)


def scaled_mm(
    self,
    mat2,
    scale_a,
    scale_b,
    bias=None,
    scale_result=None,
    out_dtype=None,
    use_fast_accum=False,
):
    logger.debug("GEMS_KUNLUNXIN SCALED_MM")
    return _scaled_mm_impl(self, mat2, scale_a, scale_b, bias, out_dtype, None)


def scaled_mm_out(
    self,
    mat2,
    scale_a,
    scale_b,
    bias=None,
    scale_result=None,
    out_dtype=None,
    use_fast_accum=False,
    *,
    out,
):
    logger.debug("GEMS_KUNLUNXIN SCALED_MM_OUT")
    return _scaled_mm_impl(self, mat2, scale_a, scale_b, bias, out_dtype, out)