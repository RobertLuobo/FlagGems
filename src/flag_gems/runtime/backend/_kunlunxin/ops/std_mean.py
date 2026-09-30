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

import importlib

import torch
import triton
import triton.language as tl

from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as ext

_generic = importlib.import_module("flag_gems.ops.std_mean")


@libentry()
@triton.jit
def _std_mean_global_chunk_map_kernel(
    inp,
    scratch,
    N,
    CHUNK_SIZE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = ext.program_id(0)
    num_programs = tl.num_programs(0)
    compute_dtype = tl.float64 if inp.type.element_ty == tl.float64 else tl.float32
    base = pid * CHUNK_SIZE
    lane = tl.arange(0, BLOCK_SIZE)
    shift = tl.load(inp + base).to(compute_dtype)
    count = tl.minimum(tl.maximum(N - base, 0), CHUNK_SIZE).to(compute_dtype)

    delta_sum = tl.zeros((BLOCK_SIZE,), dtype=compute_dtype)
    value_sum = tl.zeros((BLOCK_SIZE,), dtype=compute_dtype)
    for start in range(0, CHUNK_SIZE, BLOCK_SIZE):
        local = start + lane
        offs = base + local
        mask = (local < CHUNK_SIZE) & (offs < N)
        vals = tl.load(inp + offs, mask=mask, other=shift).to(compute_dtype)
        delta = vals - shift
        delta_sum += tl.where(mask, delta, 0.0)
        value_sum += tl.where(mask, vals, 0.0)
    delta_total = tl.sum(delta_sum)
    mean = _generic._stable_mean_from_shift(
        shift, delta_total, tl.sum(value_sum), count
    )

    squared_deviation_sum = tl.zeros((BLOCK_SIZE,), dtype=compute_dtype)
    for start in range(0, CHUNK_SIZE, BLOCK_SIZE):
        local = start + lane
        offs = base + local
        mask = (local < CHUNK_SIZE) & (offs < N)
        vals = tl.load(inp + offs, mask=mask, other=mean).to(compute_dtype)
        deviation = vals - mean
        squared_deviation_sum += tl.where(mask, deviation * deviation, 0.0)
    m2 = tl.sum(squared_deviation_sum)
    m2 = tl.where(m2 < 0.0, 0.0, m2)
    tl.store(scratch + pid, mean)
    tl.store(scratch + num_programs + pid, m2)
    tl.store(scratch + 2 * num_programs + pid, count)


_orig_run_global = _generic._run_global


def _run_global(
    inp,
    out_std,
    out_mean,
    N,
    denominator,
    denominator_low_bits,
    denominator_high_bits,
    use_fp64_denominator,
    is_complex,
):
    # XPU3 miscompiles the generic grid-stride multi-program global map kernel:
    # its masked ``tl.sum`` count accumulation over-counts by a fixed amount and
    # its per-lane m2 reduction returns wrong partials whenever a program owns a
    # partially-masked tail block or the grid-stride loop iterates more than once
    # (both trigger only once N exceeds the single-CTA threshold). The single-CTA,
    # complex, and small paths are correct, so defer them to the generic routine
    # and only replace the real multi-program branch with a contiguous-chunk map
    # kernel (arithmetic count, per-program shift preserved for outlier
    # robustness) feeding the unchanged generic finish kernel.
    use_ascend_real_split = (
        _generic.runtime_device.vendor_name == "ascend"
        and not is_complex
        and inp.dtype in (torch.float16, torch.bfloat16, torch.float32)
    )
    if is_complex or use_ascend_real_split or N <= _generic._GLOBAL_SINGLE_CTA_LIMIT:
        return _orig_run_global(
            inp,
            out_std,
            out_mean,
            N,
            denominator,
            denominator_low_bits,
            denominator_high_bits,
            use_fp64_denominator,
            is_complex,
        )

    block_size = _generic._BLOCK_SIZE
    num_programs = min(triton.cdiv(N, block_size), _generic._MAX_GLOBAL_PROGRAMS)
    chunk_size = triton.cdiv(triton.cdiv(N, num_programs), block_size) * block_size
    num_programs = triton.cdiv(N, chunk_size)
    partial_dtype = torch.float64 if inp.dtype == torch.float64 else torch.float32
    scratch = torch.empty(
        (num_programs * 3,), dtype=partial_dtype, device=inp.device
    )
    _std_mean_global_chunk_map_kernel[(num_programs,)](
        inp,
        scratch,
        N,
        CHUNK_SIZE=chunk_size,
        BLOCK_SIZE=block_size,
    )
    _generic._std_mean_global_finish_kernel[(1,)](
        scratch,
        out_std,
        out_mean,
        N,
        denominator,
        denominator_low_bits,
        denominator_high_bits,
        USE_FP64_DENOMINATOR=use_fp64_denominator,
        NUM_PARTIALS=num_programs,
        BLOCK_SIZE=triton.next_power_of_2(num_programs),
    )


# Route every std_mean full-tensor reduction through the corrected global path.
_generic._run_global = _run_global


def std_mean_correction(inp, dim=None, *, correction=None, keepdim=False):
    return _generic.std_mean_correction(
        inp, dim, correction=correction, keepdim=keepdim
    )


__all__ = ["std_mean_correction"]
