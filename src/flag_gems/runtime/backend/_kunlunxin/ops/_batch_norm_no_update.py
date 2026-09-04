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

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry, tl_extra_shim

logger = logging.getLogger(__name__)
rsqrt = tl_extra_shim.rsqrt


# NOTE (kunlunxin / XPU inference rewrite, 2026-08-17):
# The previous flat kernel indexed per-lane channel stats as
# `channels = (offsets // INNER) % C` and loaded running_mean/running_var/weight/bias
# through that per-lane gather, which the XPU compiler lowered to slow discrete
# accesses (measured ~6-12 ms/call for the small benchmark shapes, ~0.009x speedup).
#
# In the natural [N, C, S] contiguous layout each (n, c) slice is a run of S
# CONTIGUOUS elements sharing ONE channel. So the (inference) no-update kernel maps
# one program to each (n, c) slice (grid = N*C, same pattern as the batch_norm
# 3-stage normalize kernel): stats/affine are loaded ONCE per program as scalars,
# and the data tiles are contiguous block-DMA (masked only when S % TILE_S != 0).
# TILE_S < 64 is deliberately avoided: the XPU compiler miscompiles scalar+
# small-tile broadcast math for TILE<=32 (wrong results, verified).
#
# `_batch_norm_no_update_kernel` below keeps the original grid=N*C / TILE_S=4096
# shape: it is shared by `batch_norm.py` and
# `_native_batch_norm_legit_no_training.py` (unchanged consumers).
#
# NOTE (kunlunxin / XPU batch-fusion rewrite, 2026-09-04):
# Timing breakdown on P800 (do_bench, fp32) shows the per-(n,c)-slice launch pays
# a large per-program cost: 256 programs x 1 tile ~= 88us while 32 programs x 8
# sequential tiles ~= 30us for the SAME 256 slice-tiles (16K elements). The array
# is launch/wave-bound, not bandwidth-bound at these sizes: the card runs ~1 wave
# of ~96 programs efficiently and 256 programs cost ~3 waves. So for
# spatial_dim <= 8192 we FUSE the batch dimension in `_batch_norm_no_update_fused_kernel`:
# one program handles NB consecutive n-slices of the same channel
# (grid = C*ceil(N/NB) programs, NB = N/2 for C>=16, NB = N/4 for C<16, NB = N if
# N < 8), keeping each (n, c) data tile a contiguous block-DMA run. The tile also
# switches to an exact-fit policy (S<=512 -> T=512, S<=1024 -> T=1024, else
# min(next_pow2(S), 4096)): the 4096-lane masked tile wastes 75-99% of its lanes
# when S is small. For spatial_dim > 8192 (bandwidth-bound) the per-slice grid is
# kept. Measured A/B (fp32, 2026-09-04, see harness/solution/performance/
# batch_norm_no_update_perf.md): (16,16,64) 88.7us -> 29.7us (2.99x),
# (16,16,1024) 87.0us -> 31.5us (2.76x), (16,16,8,48) 90.1us -> 28.7us (3.14x),
# (16,16,4098) 133.7us -> 101.6us (1.32x), (16,8,128,128) 94.5us -> ~95us
# (unchanged per-slice path).

BNNU_TILE_S = 4096  # shared per-slice tile (kept for batch_norm.py consumers)
BNNU_MAX_PROGRAMS = 4096
BNNU_BIG_S = 8192  # spatial_dim > BNNU_BIG_S uses the per-slice (grid=N*C) path
BNNU_MAX_NB = 256  # cap sequential n-iters per program


def _bnu_tile_s(spatial_dim):
    """Exact-fit XPU tile for the batch-fused no-update kernel."""
    if spatial_dim <= 0:
        return 1, False
    if spatial_dim <= 512:
        return 512, (spatial_dim % 512) != 0
    if spatial_dim <= 1024:
        return 1024, (spatial_dim % 1024) != 0
    tile = min(triton.next_power_of_2(spatial_dim), 4096)
    return tile, (spatial_dim % tile) != 0


def _bnu_n_batch_groups(batch_dim, feat_dim, spatial_dim):
    """Number of n-groups; each program handles ceil(batch_dim/groups) n-slices.

    Small spatial dims are launch/wave-bound: ~32 programs (a single wave) is the
    measured optimum; large spatial dims are bandwidth-bound: keep per-slice.
    """
    if spatial_dim > BNNU_BIG_S or batch_dim <= 1:
        return batch_dim
    if batch_dim < 8:
        return 1
    groups = 4 if feat_dim < 16 else 2
    groups = max(groups, -(-batch_dim // BNNU_MAX_NB))
    return min(batch_dim, groups)


# NOTE (kunlunxin / XPU exact-tile rewrite, 2026-09-04):
# `_batch_norm_no_update_kernel` below keeps the *signature* used by the shared
# batch_norm inference callers (grid = N*C, TILE_S = BNNU_TILE_S constexpr) but
# picks the tile width at RUNTIME from spatial_dim (uniform branch): 512 lanes for
# S <= 512, 1024 lanes for S <= 1024, TILE_S otherwise. The 4096-lane masked tile
# wastes 75-99% of its lanes for the small-S shapes that dominate the
# batch_norm_no_update benchmark (see harness/solution/performance/
# batch_norm_no_update_perf.md); measured per-slice gains on P800 (fp32, 2026-09-04):
# (16,16,64) 88.0us -> ~75us, (16,16,8,48) 90.2us -> ~78us,
# (16,16,8,88) 91.8us -> ~80us, (16,16,128) 86.8us -> ~75us.
# S > 1024 keeps the caller's TILE_S path unchanged.


@libentry()
@triton.jit(do_not_specialize=["eps"])
def _batch_norm_no_update_kernel(
    input_pointer,  # [N*C, S] contiguous, flattened
    weight_pointer,  # [C] or unused
    bias_pointer,  # [C] or unused
    running_mean_pointer,  # [C]
    running_var_pointer,  # [C]
    output_pointer,
    feat_dim,
    spatial_dim,
    eps,
    HAS_WEIGHT: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    TILE_S: tl.constexpr,
    NEED_MASK: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    c = pid % feat_dim
    base = pid * spatial_dim

    mean = tl.load(running_mean_pointer + c).to(tl.float32)
    inv_std = rsqrt(tl.load(running_var_pointer + c).to(tl.float32) + eps)
    if HAS_WEIGHT:
        weight = tl.load(weight_pointer + c).to(tl.float32)
    else:
        weight = 1.0
    if HAS_BIAS:
        bias = tl.load(bias_pointer + c).to(tl.float32)
    else:
        bias = 0.0

    if spatial_dim <= 512:
        for off in range(0, spatial_dim, 512):
            idx = off + tl.arange(0, 512)
            mask = idx < spatial_dim
            x = tl.load(input_pointer + base + idx, mask=mask).to(tl.float32)
            y = weight * (x - mean) * inv_std + bias
            tl.store(
                output_pointer + base + idx,
                y.to(output_pointer.dtype.element_ty),
                mask=mask,
            )
    elif spatial_dim <= 1024:
        for off in range(0, spatial_dim, 1024):
            idx = off + tl.arange(0, 1024)
            mask = idx < spatial_dim
            x = tl.load(input_pointer + base + idx, mask=mask).to(tl.float32)
            y = weight * (x - mean) * inv_std + bias
            tl.store(
                output_pointer + base + idx,
                y.to(output_pointer.dtype.element_ty),
                mask=mask,
            )
    else:
        for off in range(0, spatial_dim, TILE_S):
            idx = off + tl.arange(0, TILE_S)
            if NEED_MASK:
                mask = idx < spatial_dim
                x = tl.load(input_pointer + base + idx, mask=mask).to(tl.float32)
                y = weight * (x - mean) * inv_std + bias
                tl.store(
                    output_pointer + base + idx,
                    y.to(output_pointer.dtype.element_ty),
                    mask=mask,
                )
            else:
                x = tl.load(input_pointer + base + idx).to(tl.float32)
                y = weight * (x - mean) * inv_std + bias
                tl.store(output_pointer + base + idx, y.to(output_pointer.dtype.element_ty))


@libentry()
@triton.jit(do_not_specialize=["eps"])
def _batch_norm_no_update_fused_kernel(
    input_pointer,  # [N*C, S] contiguous, flattened
    weight_pointer,  # [C] or unused
    bias_pointer,  # [C] or unused
    running_mean_pointer,  # [C]
    running_var_pointer,  # [C]
    output_pointer,
    batch_dim,
    feat_dim,
    spatial_dim,
    eps,
    program_base,  # first program id of this launch (chunking)
    NB: tl.constexpr,  # consecutive n-slices handled per program
    HAS_WEIGHT: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    TILE_S: tl.constexpr,
    NEED_MASK: tl.constexpr,
):
    pid = program_base + tl.program_id(axis=0)
    c = pid % feat_dim
    n = (pid // feat_dim) * NB

    mean = tl.load(running_mean_pointer + c).to(tl.float32)
    inv_std = rsqrt(tl.load(running_var_pointer + c).to(tl.float32) + eps)
    if HAS_WEIGHT:
        weight = tl.load(weight_pointer + c).to(tl.float32)
    else:
        weight = 1.0
    if HAS_BIAS:
        bias = tl.load(bias_pointer + c).to(tl.float32)
    else:
        bias = 0.0

    n_end = n + NB
    if n_end > batch_dim:
        n_end = batch_dim
    for nn in range(n, n_end):
        base = (nn * feat_dim + c) * spatial_dim
        for off in range(0, spatial_dim, TILE_S):
            idx = off + tl.arange(0, TILE_S)
            if NEED_MASK:
                mask = idx < spatial_dim
                x = tl.load(input_pointer + base + idx, mask=mask).to(tl.float32)
                y = weight * (x - mean) * inv_std + bias
                tl.store(
                    output_pointer + base + idx,
                    y.to(output_pointer.dtype.element_ty),
                    mask=mask,
                )
            else:
                x = tl.load(input_pointer + base + idx).to(tl.float32)
                y = weight * (x - mean) * inv_std + bias
                tl.store(output_pointer + base + idx, y.to(output_pointer.dtype.element_ty))


def _batch_norm_no_update(
    input,
    weight=None,
    bias=None,
    running_mean=None,
    running_var=None,
    momentum=0.1,
    eps=1e-5,
):
    logger.debug("GEMS_KUNLUNXIN _BATCH_NORM_NO_UPDATE")
    if input.ndim < 2:
        raise RuntimeError("batch_norm expects input with at least 2 dimensions")
    if running_mean is None or running_var is None:
        raise RuntimeError(
            "running_mean and running_var are required for no-update batch_norm"
        )

    channels = input.shape[1]
    if running_mean.numel() != channels or running_var.numel() != channels:
        raise RuntimeError("running statistics must contain one value per channel")

    input_contiguous = input.contiguous()
    output = torch.empty_like(input_contiguous)
    n_elements = input_contiguous.numel()
    batch_dim = input.shape[0]
    n_slices = batch_dim * channels
    inner = n_elements // n_slices if n_slices > 0 else 0
    tile_s, need_mask = _bnu_tile_s(inner)
    n_groups = _bnu_n_batch_groups(batch_dim, channels, inner)
    nb = -(-batch_dim // n_groups) if n_groups > 0 else 1

    if n_elements > 0:
        input_flat = input_contiguous.reshape(-1)
        output_flat = output.reshape(-1)
        weight_pointer = input_flat if weight is None else weight
        bias_pointer = input_flat if bias is None else bias
        with torch_device_fn.device(input.device):
            # n_groups x channels programs, each handling NB consecutive n-slices;
            # chunk the program space when it exceeds the per-launch cap.
            programs = n_groups * channels
            for program_base in range(0, programs, BNNU_MAX_PROGRAMS):
                program_count = min(BNNU_MAX_PROGRAMS, programs - program_base)
                _batch_norm_no_update_fused_kernel[(program_count,)](
                    input_flat,
                    weight_pointer,
                    bias_pointer,
                    running_mean,
                    running_var,
                    output_flat,
                    batch_dim,
                    channels,
                    inner,
                    eps,
                    program_base,
                    NB=nb,
                    HAS_WEIGHT=weight is not None,
                    HAS_BIAS=bias is not None,
                    TILE_S=tile_s,
                    NEED_MASK=need_mask,
                    num_warps=4,
                    isCloseVectorization=True,
                    buffer_size_limit=2048,
                )

    save_mean = torch.empty((0,), dtype=input.dtype, device=input.device)
    save_var = torch.empty((0,), dtype=input.dtype, device=input.device)
    reserved = torch.empty((0,), dtype=torch.uint8, device=input.device)
    return output.view_as(input), save_mean, save_var, reserved