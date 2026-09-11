# Kunlunxin (XPU) override of lift_out.
#
# aten::lift.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!) is an autograd
# metadata op whose only data effect is a device copy of `self` into `out`
# (the functional test compares the contents of `out` against a reference
# copy; the reference side shows the same copy semantics).
#
# The generic implementation (src/flag_gems/ops/lift.py) does
# `out.copy_(A)`, which inside use_gems() re-dispatches to the Kunlunxin
# Triton `copy_` kernel.  A controlled A/B on the benchmark matrix
# (12 shapes x fp16/fp32/bf16, median of 200 reps) shows the Triton path
# measures 1.2x-3.1x slower than the vendor copy engine: e.g. fp32
# (4096, 4096) 0.138ms vs 0.090ms, fp16 (64, 64, 256) 0.079ms vs 0.026ms.
#
# Fix: mirror the proven "same key" recipe used by slice_backward / resize /
# constant_pad_nd (and the existing _kunlunxin amin/amax/aminmax/all/any
# overrides): call torch.ops.aten._copy_from directly.  flag_gems does NOT
# override `_copy_from`, so it reaches the vendor's native strided copy engine
# on-device, for all dtypes/ranks/strides (no CPU fallback involved).
import logging

import torch

logger = logging.getLogger("flag_gems").getChild(__name__.lstrip("."))


def lift_out(A, *, out=None):
    """Implements aten::lift.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!).

    Copies ``A`` into ``out`` via the vendor native copy engine and returns
    ``out``, matching the reference semantics of the generic implementation.
    """
    logger.debug("GEMS_KUNLUNXIN LIFT_OUT")
    if out is None:
        out = torch.empty_like(A, memory_format=torch.contiguous_format)
    torch.ops.aten._copy_from(A, out, False)
    return out
