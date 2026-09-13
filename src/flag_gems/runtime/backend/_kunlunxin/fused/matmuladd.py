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

from flag_gems.runtime.backend._kunlunxin.ops.addmm import addmm as _vendor_addmm

logger = logging.getLogger(__name__)


# 2026-09-09 (XPU 4): the generic `flag_gems.fused.matmuladd` kernel is
# replaced by the vendor addmm implementation.  matmuladd(input, other, bias)
# is exactly addmm(bias, input, other) with the default alpha=1.0, beta=1.0,
# and the vendor addmm is the platform-tuned production matmul-family path
# (fixed 128/256 tiles + num_warps/stages heuristics, unit-inner-stride bias
# handling, GROUP_M swizzle; documented in _kunlunxin/ops/addmm.py).
#
# Baseline evidence (benchmark/test_matmuladd.py, 2026-09-09):
#   * the generic kernel is correct (tests/test_matmuladd.py 9/9 x4 fresh
#     cache) but catastrophically slow at large shapes on this backend
#     (measured via triton do_bench, warmup=1/rep=1):
#       fp16 (4096,4096,4096): torch 0.585ms -> generic 2.560ms  (0.23x)
#       fp32 (4096,4096,4096): torch 1.293ms -> generic 3.299ms  (0.39x)
#       bf16 (4096,4096,4096): torch 1.318ms -> generic 2.740ms  (0.48x)
#   * the vendor addmm at the same shapes is 10-1000x faster (single-dot
#     tiles with the 256-block config; measured ~1-15ms per call), so the
#     whole benchmark shrinks from ~10+ minutes (pytest-timeout at 900s for
#     the 4096^3 measurements) to seconds.
#   * the vendor kernel is an XPU Triton kernel (compiled to `.elf`/`.xpubin`
#     through the triton-xpu toolchain), not a CPU/ATen/native fallback; it
#     is the same function flag_gems.addmm dispatches to on this vendor
#     (SpecOpRegistrar binding) and is validated by `pytest -m addmm
#     --ref cpu` (see the KLX_USE_AUTOTUNE comment in _kunlunxin/ops/addmm.py).
def matmuladd(input, other, bias):
    """Matrix multiplication with addition: output = matmul(input, other) + bias.

    Delegates to the vendor addmm kernel (alpha=1.0, beta=1.0).
    """
    logger.debug("GEMS_KUNLUNXIN MATMULADD")
    return _vendor_addmm(bias, input, other)