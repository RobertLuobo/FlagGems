# Kunlunxin (XPU) override of `negative`.
#
# `negative` is functionally identical to `neg` (both compute -x), and kunlunxin
# already ships a tuned override for neg (`_kunlunxin/ops/neg.py`, a
# `pointwise_dynamic` with the XPU-tuned CodeGenConfig). But `negative` was NOT
# overridden, so it fell back to the generic hand-written kernel
# (`ops/negative.py`) with a fixed `BLOCK_SIZE=1024` + `grid=cdiv(n,1024)` and no
# XPU tuning -> launch-bound / discrete slow path on large shapes (IR baseline
# `harness/perf_ir_3/ir-negative-dev5.log`: large shapes gems 0.04-0.17, plus
# per-shape first-compile warmup spikes to 240-390ms).
#
# Fix: reuse the exact tuned neg recipe by delegating to it. Zero algorithm
# change (both are -x), zero correctness risk.
import logging

import torch

from .neg import neg_func

logger = logging.getLogger(__name__)


def negative(A):
    logger.debug("GEMS_KUNLUNXIN NEGATIVE")
    # Host-side pre-allocation: handing `out0` over to pointwise_dynamic makes
    # prepare_args skip its per-call dtype-promotion walk + `empty_like`, which is
    # the dominant cost at launch-bound shapes. It is only valid while the
    # promoted result dtype is provably A.dtype: a single floating-point operand
    # with promotion method DEFAULT promotes to itself, so the guard is exact
    # (integer/bool/complex inputs keep the generic path).
    if A.is_floating_point():
        return neg_func(A, out0=torch.empty_like(A))
    return neg_func(A)
