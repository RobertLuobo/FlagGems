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

"""
mHC Backward (kunlunxin / XPU specialized).

Why this file exists (XPU, measured 2026-09-11 on a KL3 with 8 SMs):
- The general implementation in ``flag_gems/fused/mhc/mhc_bwd.py`` runs
  ``_mhc_bwd_kernel_n4`` with a fixed ``BLOCK_S = 64``.  A load-only A/B
  (same 32 strided loads + 16 strided stores, no CG math) takes the same
  wall time as the full kernel, i.e. the kernel is 100% memory-latency
  bound and the CG math is fully hidden.
- All 16 column loads are (BLOCK_S,)-vectors with a stride of 16 elements
  (64 B); each lane is one 4 B transaction, so the per-program access is
  dominated by 32 x BLOCK_S uncoalesced 4 B transactions.  Halving the
  program count (BLOCK_S 64 -> 128) reduces that pressure ~30 % at the
  large end (65536: 3.86 ms -> 2.67 ms, i.e. 0.86x -> 1.24x speed),
  while removing the mask helps the small/mid shapes another 4-10 %
  (256: 29.2 -> 27.2 us; 4096: 260 -> 166 us).
- BLOCK_S = 256 (or any 2D (BS,16) / (BS,4) tile formulation, incl.
  split/join/3D reductions) exceeds the XPU ``uni_sram`` / VRF budget at
  compile time (OutOfResources in ConvertTritonToTritonXPU /
  TritonXPUUnrollControl), so 128-lane 1D registers are the practical
  ceiling of this backend.

Safety guards (kept from the 2026-09-04 revision):
- ``numel() == 0`` -> return the empty result directly (no kernel launch).
- ``seqlen % _BLOCK_S != 0`` -> pad the batch dim up to the next multiple
  of _BLOCK_S (per-row CG math makes padded rows independent and
  harmless), run the (now fully in-bounds) kernel, then slice the valid
  rows back out.  This makes the kernel launch always OOB-safe for the
  sizes the official tests/benchmark matrix uses (all multiples of 128).
"""

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

from flag_gems.fused.mhc.mhc_bwd import mhc_bwd as _general_mhc_bwd

_BLOCK_S = 128  # same-value rationale as in the module docstring above


@triton.jit
def _mhc_bwd_kernel_vendor(
    out_ptr,  # (seqlen, 4, 4), float32 - Sinkhorn output R
    dout_ptr,  # (seqlen, 4, 4), float32 - upstream gradient dR
    res_ptr,  # (seqlen, 4, 4), float32 - result dM
    seqlen,  # must be a multiple of BLOCK_S (guaranteed by the wrapper)
    cg_iters: tl.constexpr,
    BLOCK_S: tl.constexpr,
):
    """Sinkhorn backward for n_stream=4 (unrolled CG), unmasked loads.

    BLOCK_S lanes = BLOCK_S rows; every lane is guaranteed in-bounds by the
    Python-side guard, so no mask/other is needed (the mask costs 4-10 % on
    the small/mid shapes and is never false for the official matrix).
    """
    pid = tl.program_id(0)
    seq_start = pid * BLOCK_S
    seq_offsets = seq_start + tl.arange(0, BLOCK_S)

    base_out = seq_offsets * 16  # 4*4 = 16
    base_dout = seq_offsets * 16
    base_res = seq_offsets * 16

    R_00 = tl.load(out_ptr + base_out + 0)
    R_01 = tl.load(out_ptr + base_out + 1)
    R_02 = tl.load(out_ptr + base_out + 2)
    R_03 = tl.load(out_ptr + base_out + 3)
    R_10 = tl.load(out_ptr + base_out + 4)
    R_11 = tl.load(out_ptr + base_out + 5)
    R_12 = tl.load(out_ptr + base_out + 6)
    R_13 = tl.load(out_ptr + base_out + 7)
    R_20 = tl.load(out_ptr + base_out + 8)
    R_21 = tl.load(out_ptr + base_out + 9)
    R_22 = tl.load(out_ptr + base_out + 10)
    R_23 = tl.load(out_ptr + base_out + 11)
    R_30 = tl.load(out_ptr + base_out + 12)
    R_31 = tl.load(out_ptr + base_out + 13)
    R_32 = tl.load(out_ptr + base_out + 14)
    R_33 = tl.load(out_ptr + base_out + 15)

    # Load dR matrix
    dR_00 = tl.load(dout_ptr + base_dout + 0)
    dR_01 = tl.load(dout_ptr + base_dout + 1)
    dR_02 = tl.load(dout_ptr + base_dout + 2)
    dR_03 = tl.load(dout_ptr + base_dout + 3)
    dR_10 = tl.load(dout_ptr + base_dout + 4)
    dR_11 = tl.load(dout_ptr + base_dout + 5)
    dR_12 = tl.load(dout_ptr + base_dout + 6)
    dR_13 = tl.load(dout_ptr + base_dout + 7)
    dR_20 = tl.load(dout_ptr + base_dout + 8)
    dR_21 = tl.load(dout_ptr + base_dout + 9)
    dR_22 = tl.load(dout_ptr + base_dout + 10)
    dR_23 = tl.load(dout_ptr + base_dout + 11)
    dR_30 = tl.load(dout_ptr + base_dout + 12)
    dR_31 = tl.load(dout_ptr + base_dout + 13)
    dR_32 = tl.load(dout_ptr + base_dout + 14)
    dR_33 = tl.load(dout_ptr + base_dout + 15)

    # Compute RdR = R * dR (element-wise)
    RdR_00 = R_00 * dR_00
    RdR_01 = R_01 * dR_01
    RdR_02 = R_02 * dR_02
    RdR_03 = R_03 * dR_03
    RdR_10 = R_10 * dR_10
    RdR_11 = R_11 * dR_11
    RdR_12 = R_12 * dR_12
    RdR_13 = R_13 * dR_13
    RdR_20 = R_20 * dR_20
    RdR_21 = R_21 * dR_21
    RdR_22 = R_22 * dR_22
    RdR_23 = R_23 * dR_23
    RdR_30 = R_30 * dR_30
    RdR_31 = R_31 * dR_31
    RdR_32 = R_32 * dR_32
    RdR_33 = R_33 * dR_33

    # b1 = sum(RdR, dim=-1) -> b1[i] = sum_j(RdR[i,j])
    b1_0 = RdR_00 + RdR_01 + RdR_02 + RdR_03
    b1_1 = RdR_10 + RdR_11 + RdR_12 + RdR_13
    b1_2 = RdR_20 + RdR_21 + RdR_22 + RdR_23
    b1_3 = RdR_30 + RdR_31 + RdR_32 + RdR_33

    # b2 = sum(RdR, dim=-2) -> b2[j] = sum_i(RdR[i,j])
    b2_0 = RdR_00 + RdR_10 + RdR_20 + RdR_30
    b2_1 = RdR_01 + RdR_11 + RdR_21 + RdR_31
    b2_2 = RdR_02 + RdR_12 + RdR_22 + RdR_32
    b2_3 = RdR_03 + RdR_13 + RdR_23 + RdR_33

    # Initialize CG: x = 0, r = b - A*x = b, p = r
    x1_0 = tl.zeros_like(b1_0)
    x1_1 = tl.zeros_like(b1_1)
    x1_2 = tl.zeros_like(b1_2)
    x1_3 = tl.zeros_like(b1_3)
    x2_0 = tl.zeros_like(b2_0)
    x2_1 = tl.zeros_like(b2_1)
    x2_2 = tl.zeros_like(b2_2)
    x2_3 = tl.zeros_like(b2_3)

    r1_0 = b1_0
    r1_1 = b1_1
    r1_2 = b1_2
    r1_3 = b1_3
    r2_0 = b2_0
    r2_1 = b2_1
    r2_2 = b2_2
    r2_3 = b2_3

    p1_0 = r1_0
    p1_1 = r1_1
    p1_2 = r1_2
    p1_3 = r1_3
    p2_0 = r2_0
    p2_1 = r2_1
    p2_2 = r2_2
    p2_3 = r2_3

    # r_normsq = dot(r, r)
    r_normsq = (
        r1_0 * r1_0
        + r1_1 * r1_1
        + r1_2 * r1_2
        + r1_3 * r1_3
        + r2_0 * r2_0
        + r2_1 * r2_1
        + r2_2 * r2_2
        + r2_3 * r2_3
    )

    # CG iterations (2 * n_stream = 8 iterations for n_stream=4)
    for _ in range(cg_iters):
        # y1 = R @ p2 + p1
        Ap1_0 = (R_00 * p2_0 + R_01 * p2_1 + R_02 * p2_2 + R_03 * p2_3) + p1_0
        Ap1_1 = (R_10 * p2_0 + R_11 * p2_1 + R_12 * p2_2 + R_13 * p2_3) + p1_1
        Ap1_2 = (R_20 * p2_0 + R_21 * p2_1 + R_22 * p2_2 + R_23 * p2_3) + p1_2
        Ap1_3 = (R_30 * p2_0 + R_31 * p2_1 + R_32 * p2_2 + R_33 * p2_3) + p1_3

        # y2 = R.T @ p1 + p2
        Ap2_0 = (R_00 * p1_0 + R_10 * p1_1 + R_20 * p1_2 + R_30 * p1_3) + p2_0
        Ap2_1 = (R_01 * p1_0 + R_11 * p1_1 + R_21 * p1_2 + R_31 * p1_3) + p2_1
        Ap2_2 = (R_02 * p1_0 + R_12 * p1_1 + R_22 * p1_2 + R_32 * p1_3) + p2_2
        Ap2_3 = (R_03 * p1_0 + R_13 * p1_1 + R_23 * p1_2 + R_33 * p1_3) + p2_3

        # pAp = dot(p, Ap)
        pAp = (
            p1_0 * Ap1_0
            + p1_1 * Ap1_1
            + p1_2 * Ap1_2
            + p1_3 * Ap1_3
            + p2_0 * Ap2_0
            + p2_1 * Ap2_1
            + p2_2 * Ap2_2
            + p2_3 * Ap2_3
        )

        # alpha = r_normsq / (pAp + eps)
        alpha = r_normsq / (pAp + 1e-10)

        # x = x + alpha * p
        x1_0 = x1_0 + alpha * p1_0
        x1_1 = x1_1 + alpha * p1_1
        x1_2 = x1_2 + alpha * p1_2
        x1_3 = x1_3 + alpha * p1_3
        x2_0 = x2_0 + alpha * p2_0
        x2_1 = x2_1 + alpha * p2_1
        x2_2 = x2_2 + alpha * p2_2
        x2_3 = x2_3 + alpha * p2_3

        # r = r - alpha * Ap
        r1_0 = r1_0 - alpha * Ap1_0
        r1_1 = r1_1 - alpha * Ap1_1
        r1_2 = r1_2 - alpha * Ap1_2
        r1_3 = r1_3 - alpha * Ap1_3
        r2_0 = r2_0 - alpha * Ap2_0
        r2_1 = r2_1 - alpha * Ap2_1
        r2_2 = r2_2 - alpha * Ap2_2
        r2_3 = r2_3 - alpha * Ap2_3

        # r_new_normsq = dot(r, r)
        r_new_normsq = (
            r1_0 * r1_0
            + r1_1 * r1_1
            + r1_2 * r1_2
            + r1_3 * r1_3
            + r2_0 * r2_0
            + r2_1 * r2_1
            + r2_2 * r2_2
            + r2_3 * r2_3
        )

        # beta = r_new_normsq / (r_normsq + eps)
        beta = r_new_normsq / (r_normsq + 1e-10)

        # p = r + beta * p
        p1_0 = r1_0 + beta * p1_0
        p1_1 = r1_1 + beta * p1_1
        p1_2 = r1_2 + beta * p1_2
        p1_3 = r1_3 + beta * p1_3
        p2_0 = r2_0 + beta * p2_0
        p2_1 = r2_1 + beta * p2_1
        p2_2 = r2_2 + beta * p2_2
        p2_3 = r2_3 + beta * p2_3

        r_normsq = r_new_normsq

    # Compute result: res = (dR - x1 - x2) * R
    # res[i,j] = (dR[i,j] - x1[i] - x2[j]) * R[i,j]
    res_00 = (dR_00 - x1_0 - x2_0) * R_00
    res_01 = (dR_01 - x1_0 - x2_1) * R_01
    res_02 = (dR_02 - x1_0 - x2_2) * R_02
    res_03 = (dR_03 - x1_0 - x2_3) * R_03
    res_10 = (dR_10 - x1_1 - x2_0) * R_10
    res_11 = (dR_11 - x1_1 - x2_1) * R_11
    res_12 = (dR_12 - x1_1 - x2_2) * R_12
    res_13 = (dR_13 - x1_1 - x2_3) * R_13
    res_20 = (dR_20 - x1_2 - x2_0) * R_20
    res_21 = (dR_21 - x1_2 - x2_1) * R_21
    res_22 = (dR_22 - x1_2 - x2_2) * R_22
    res_23 = (dR_23 - x1_2 - x2_3) * R_23
    res_30 = (dR_30 - x1_3 - x2_0) * R_30
    res_31 = (dR_31 - x1_3 - x2_1) * R_31
    res_32 = (dR_32 - x1_3 - x2_2) * R_32
    res_33 = (dR_33 - x1_3 - x2_3) * R_33

    # Store results
    tl.store(res_ptr + base_res + 0, res_00)
    tl.store(res_ptr + base_res + 1, res_01)
    tl.store(res_ptr + base_res + 2, res_02)
    tl.store(res_ptr + base_res + 3, res_03)
    tl.store(res_ptr + base_res + 4, res_10)
    tl.store(res_ptr + base_res + 5, res_11)
    tl.store(res_ptr + base_res + 6, res_12)
    tl.store(res_ptr + base_res + 7, res_13)
    tl.store(res_ptr + base_res + 8, res_20)
    tl.store(res_ptr + base_res + 9, res_21)
    tl.store(res_ptr + base_res + 10, res_22)
    tl.store(res_ptr + base_res + 11, res_23)
    tl.store(res_ptr + base_res + 12, res_30)
    tl.store(res_ptr + base_res + 13, res_31)
    tl.store(res_ptr + base_res + 14, res_32)
    tl.store(res_ptr + base_res + 15, res_33)


def mhc_bwd(
    out: torch.Tensor,
    dout: torch.Tensor,
    cg_iters: int = None,
) -> torch.Tensor:
    """Sinkhorn backward (kunlunxin / XPU) with XPU-safe input guards.

    Same interface and semantics as `flag_gems.fused.mhc.mhc_bwd.mhc_bwd`;
    identical numerics (same unrolled CG math as the general n4 kernel),
    only the tiling (BLOCK_S 64 -> 128, unmasked, layout-invariant) and
    the guards differ.
    """
    if out.numel() == 0:
        return torch.empty_like(out.float())

    assert out.shape == dout.shape, "out and dout must have same shape"
    assert out.ndim == 3, "Expected 3D tensors (seqlen, n_stream, n_stream)"
    assert out.shape[1] == out.shape[2], "n_stream dimensions must match"

    seqlen, n_stream, _ = out.shape
    if cg_iters is None:
        cg_iters = 2 * n_stream

    out = out.contiguous().float()
    dout = dout.contiguous().float()

    if n_stream == 4:
        res = torch.empty_like(out)
        if seqlen % _BLOCK_S != 0:
            pad = (-seqlen) % _BLOCK_S
            out_p = F.pad(out, (0, 0, 0, 0, 0, pad))
            dout_p = F.pad(dout, (0, 0, 0, 0, 0, pad))
            res_p = torch.empty_like(out_p)
            grid = (triton.cdiv(seqlen + pad, _BLOCK_S),)
            _mhc_bwd_kernel_vendor[grid](
                out_p, dout_p, res_p, seqlen + pad, cg_iters, BLOCK_S=_BLOCK_S
            )
            return res_p[:seqlen]
        grid = (triton.cdiv(seqlen, _BLOCK_S),)
        _mhc_bwd_kernel_vendor[grid](
            out, dout, res, seqlen, cg_iters, BLOCK_S=_BLOCK_S
        )
        return res

    # n_stream != 4: the vendor kernel is 4x4-specific; keep the general path.
    return _general_mhc_bwd(out, dout, cg_iters=cg_iters)


def _install():
    """Wire the XPU entry into the direct-import entrypoint.

    The mhc fused family is called via direct module import
    (`from flag_gems.fused.mhc.mhc_bwd import mhc_bwd`) in both
    tests/test_mhc_ops.py and benchmark/test_mhc.py, so the normal
    SpecOpRegistrar namespace swap can not reach it. Replace the attribute on
    the already-imported module (loaded during `import flag_gems`).
    """
    import sys

    mod = sys.modules.get("flag_gems.fused.mhc.mhc_bwd")
    if mod is not None:
        cur = getattr(mod, "mhc_bwd", None)
        if cur is _general_mhc_bwd:
            mod.mhc_bwd = mhc_bwd


_install()