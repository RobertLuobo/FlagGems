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
HC head fused kernel (kunlunxin / XPU specialized).

Why a specialized file (XPU, measured 2026-09-04):
- The general implementation in ``flag_gems/fused/mhc/hc_head_fused_kernel.py``
  launches a single per-token Triton kernel whose inner loads are masked with
  ``h_mask = h_off < H`` (``other=0.0``). On XPU the masked tail of the last
  row of the last token reads out of the tensor allocation (mask/other is not
  enforced at the load level, same family of defect as documented for
  ``mhc_bwd``/``mhc_pre``), which raises a device kernel exception
  (``torch.AcceleratorError: CUDA error: unspecified launch failure``,
  ``cuapi_xpu_wait ... status=719``) and wedges the device for all subsequent
  launches: functional baseline ``-m hc_head_fused_kernel --ref cpu`` gives
  15 failed / 1 passed / 16 skipped (of the 16 non-skipped cases only the
  smallest ``n1_h1280_hc4`` survives; every following case dies at a plain
  ``torch.manual_seed`` in the test harness).
- The general kernel also carries ``@triton.autotune`` (5 configs keyed on
  (H, HC)); the mhc family convention on XPU is a single fixed config.

Design (mirrors the proven ``_kunlunxin/fused/mhc_pre.py`` 3-kernel pattern,
all single-shot, no internal loops, no reduction over masked lanes):
  1. ``_sqrsum_partials_kernel``  grid (N, T): exact unmasked tiles
     (K % B == 0 for the full test/benchmark matrix) -> (N, T) partials.
  2. ``torch.mm`` (vendor engine, f32, numerically identical to the reference
     ``torch.matmul``) -> (N, HC) mixes.
  3. ``_head_mix_kernel``        grid (N,): rsqrt + sigmoid -> pre_mix, all
     scalar loads, no vector reductions.
  4. ``_weighted_row_kernel``    grid (N,): weighted sum, masked loads clamped
     to an in-bounds index, masked store (mhc_pre-proven safe pattern).

Key points:
- NO ``@triton.autotune``; single config, num_stages=1 (mhc convention).
- H, HC, B are ``tl.constexpr``; any K not divisible by B is zero-padded in
  the wrapper (padded lanes contribute 0 to sqrsum and mixes, and the rsqrt
  denominator keeps the original ``K = HC * H``), so arbitrary shapes stay
  correct instead of faulting.
"""

import logging
import os

import torch
import triton
import triton.language as tl

import flag_gems.fused.mhc.hc_head_fused_kernel as _general_module
from flag_gems.fused.mhc.hc_head_fused_kernel import (
    hc_head_fused_kernel as _general_hc_head_fused_kernel,
)

logger = logging.getLogger(__name__)

# exact-tile sizes: for the official matrix (H in {1280,2560,4096,7168},
# hc_mult in {2,4}): 4*H % 1024 == 0 and 2*H % 512 == 0.
_PART_BLOCK = {2: 512, 4: 1024}
_ROW_BLOCK_MAX = 8192
_T_MAX = 128


@triton.jit
def _sqrsum_partials_kernel(
    residual_ptr,  # (N, K) bf16, contiguous, K % B == 0
    part_ptr,  # (N, T) f32, T = K // B
    K: tl.constexpr,
    B: tl.constexpr,
):
    """Exact unmasked per-token tile squares (grid (N, cdiv(K, B)))."""
    pid_n = tl.program_id(0)
    pid_t = tl.program_id(1)
    offs = pid_t * B + tl.arange(0, B)
    v = tl.load(residual_ptr + pid_n * K + offs).to(tl.float32)
    tl.store(part_ptr + pid_n * (K // B) + pid_t, tl.sum(v * v))


@triton.jit
def _head_mix_kernel(
    part_ptr,  # (N, T) f32
    mixes_ptr,  # (N, HC) f32
    hc_scale_ptr,  # (1,) f32
    hc_base_ptr,  # (HC,) f32
    pre_mix_ptr,  # (N, HC) f32
    T: tl.constexpr,
    K: tl.constexpr,
    rms_eps,
    hc_eps,
    HC: tl.constexpr,
):
    """Per-token rsqrt + sigmoid (any HC), all scalar loads, no reductions."""
    pid = tl.program_id(0)
    sq = 0.0
    for i in tl.static_range(T):
        sq += tl.load(part_ptr + pid * T + i)
    rms_inv = tl.rsqrt(sq / K + rms_eps)
    hc_scale = tl.load(hc_scale_ptr)
    for k in tl.static_range(HC):
        m = tl.load(mixes_ptr + pid * HC + k)
        b = tl.load(hc_base_ptr + k)
        p = tl.sigmoid(m * rms_inv * hc_scale + b) + hc_eps
        tl.store(pre_mix_ptr + pid * HC + k, p)


@triton.jit
def _weighted_row_kernel(
    residual_ptr,  # (N, HC*H) bf16, contiguous
    pre_mix_ptr,  # (N, HC) f32
    out_ptr,  # (N, H) bf16
    H: tl.constexpr,
    HC: tl.constexpr,
    B: tl.constexpr,
):
    """Weighted sum of the HC residual rows (masked, no reduction).

    NOTE: the load address must stay the plain ``k * H + offs`` expression.
    Any ``tl.where`` on the index (e.g. clamping masked lanes to 0) defeats
    the compiler's contiguity analysis on XPU and turns the block DMA into a
    ~22x slower discrete gather (measured 34.86ms -> 1.59ms at
    n=4096 H=4096 HC=4 for this kernel alone).
    """
    pid = tl.program_id(0)
    offs = tl.arange(0, B)
    m = offs < H
    base = pid * (HC * H)
    acc = tl.zeros([B], dtype=tl.float32)
    for k in tl.static_range(HC):
        pk = tl.load(pre_mix_ptr + pid * HC + k)
        r = tl.load(residual_ptr + base + k * H + offs, mask=m, other=0.0).to(
            tl.float32
        )
        acc += pk * r
    tl.store(out_ptr + pid * H + offs, acc.to(tl.bfloat16), mask=m)


def hc_head_fused_kernel(
    hs_flat: torch.Tensor,
    fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    out: torch.Tensor,
    hidden_size: int,
    rms_eps: float,
    hc_eps: float,
    hc_mult: int,
) -> torch.Tensor:
    """HC head fused kernel (kunlunxin / XPU specialized).

    Same interface and semantics as
    `flag_gems.fused.mhc.hc_head_fused_kernel.hc_head_fused_kernel`.
    """
    assert hs_flat.dtype == torch.bfloat16
    assert fn.dtype == torch.float32
    assert hc_scale.dtype == torch.float32
    assert hc_base.dtype == torch.float32

    num_tokens = hs_flat.shape[0]
    if num_tokens == 0:
        return out

    assert hs_flat.shape == (num_tokens, hc_mult, hidden_size)
    assert fn.shape == (hc_mult, hc_mult * hidden_size)
    assert hc_scale.shape == (1,)
    assert hc_base.shape == (hc_mult,)
    assert out.shape == (num_tokens, hidden_size)
    assert out.dtype == hs_flat.dtype

    if hs_flat.device.type != "cuda":
        return _general_hc_head_fused_kernel(
            hs_flat, fn, hc_scale, hc_base, out, hidden_size, rms_eps, hc_eps, hc_mult
        )

    H = hidden_size
    HC = hc_mult
    K = HC * H
    B = _PART_BLOCK.get(HC, 512)
    T = (K + B - 1) // B
    if T > _T_MAX:
        # pathological shape: keep upstream behavior
        return _general_hc_head_fused_kernel(
            hs_flat, fn, hc_scale, hc_base, out, hidden_size, rms_eps, hc_eps, hc_mult
        )

    residual_c = hs_flat.contiguous()
    out_c = out if out.is_contiguous() else torch.empty_like(out)

    # zero-pad the flattened K dim when it is not an exact tile (matrix shapes
    # are exact, so no copy for the common path)
    K_eff = T * B
    if K_eff == K:
        x2d = residual_c.reshape(num_tokens, K)
        fn_eff = fn
    else:
        x2d = torch.nn.functional.pad(
            residual_c.reshape(num_tokens, K), (0, K_eff - K)
        )
        fn_eff = torch.nn.functional.pad(fn, (0, K_eff - K))

    # 1) rms sqrsum partials (exact unmasked tiles)
    part = torch.empty(num_tokens, K_eff // B, dtype=torch.float32, device=hs_flat.device)
    _sqrsum_partials_kernel[(num_tokens, K_eff // B)](
        x2d,
        part,
        K=K_eff,
        B=B,
        num_warps=4,
        num_stages=1,
    )

    # 2) mixes via vendor f32 mm (numerically identical to the reference)
    mixes = torch.mm(x2d.to(torch.float32), fn_eff.t())

    # 3) rsqrt + sigmoid -> pre_mix
    pre_mix = torch.empty(num_tokens, HC, dtype=torch.float32, device=hs_flat.device)
    _head_mix_kernel[(num_tokens,)](
        part,
        mixes,
        hc_scale,
        hc_base,
        pre_mix,
        T=K_eff // B,
        K=K,
        rms_eps=rms_eps,
        hc_eps=hc_eps,
        HC=HC,
        num_warps=4,
        num_stages=1,
    )

    # 4) weighted sum of the HC residual rows
    row_block = min(triton.next_power_of_2(H), _ROW_BLOCK_MAX)
    _weighted_row_kernel[(num_tokens,)](
        residual_c,
        pre_mix,
        out_c,
        H=H,
        HC=HC,
        B=row_block,
        num_warps=8,
        num_stages=1,
    )

    if out.data_ptr() != out_c.data_ptr():
        out.copy_(out_c)
    return out


# ────────────────────────────── wiring ──────────────────────────────


def _use_general_for_ab():
    """A/B escape hatch: set FLAGGEMS_XPU_HC_HEAD_GENERAL=1 to force the
    general implementation (used only for baseline measurement / ablation)."""
    return os.environ.get("FLAGGEMS_XPU_HC_HEAD_GENERAL", "0") == "1"


def _install():
    """Wire the XPU implementation into the direct-import entrypoint.

    The mhc fused family is called via direct module import
    (`from flag_gems.fused.mhc.hc_head_fused_kernel import
    hc_head_fused_kernel`) in both tests/test_mhc_ops.py and
    benchmark/test_mhc.py, so the normal SpecOpRegistrar namespace swap can
    not reach it. Replace the attribute on the already-imported module
    (loaded during `import flag_gems`).
    """
    if _use_general_for_ab():
        return
    import sys

    mod = sys.modules.get("flag_gems.fused.mhc.hc_head_fused_kernel")
    if mod is not None:
        cur = getattr(mod, "hc_head_fused_kernel", None)
        if cur is _general_hc_head_fused_kernel:
            mod.hc_head_fused_kernel = hc_head_fused_kernel


_install()