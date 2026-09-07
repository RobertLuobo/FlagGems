# Kunlunxin (XPU) override of lift_fresh.
#
# aten::lift_fresh(Tensor(a) self) -> Tensor(a) is the autograd boundary
# "lift" identity.  The schema's alias annotation (a) on BOTH the input and the
# output declares that the result aliases the input storage, and torch's own
# CompositeExplicitAutograd kernel simply returns `self` unchanged (verified:
# torch.ops.aten.lift_fresh(x) is x on the reference backend; the torch
# baseline in benchmark/test_lift_fresh.py is a shape-independent ~3.2us
# no-op, while a data-copy kernel would scale with numel).
#
# The previous implementation materialized a full on-device copy
# (pointwise_dynamic identity kernel under the copy-family CodeGenConfig),
# which measured 0.037-2.35ms on 16M-655M element tensors
# (0.0012x-0.086x torch, ~450-800x slower than the no-op reference) and
# additionally detached the result from the input storage, deviating from the
# alias contract.  This override is a pure pass-through, mirroring the sibling
# aten::lift implementation (src/flag_gems/ops/lift.py).
import logging

import torch

logger = logging.getLogger("flag_gems").getChild(__name__.lstrip("."))


def lift_fresh(x: torch.Tensor):
    """Implements aten::lift_fresh(Tensor self) -> Tensor.

    ``lift_fresh`` lifts its argument into a "fresh" autograd leaf without
    touching the data: the returned tensor is the input itself (same storage,
    same values), exactly as torch's native CompositeExplicitAutograd kernel
    (``return self``) and the sibling ``ops/lift.py`` pass-through do.
    """
    logger.debug("GEMS_KUNLUNXIN LIFT_FRESH")
    return x
