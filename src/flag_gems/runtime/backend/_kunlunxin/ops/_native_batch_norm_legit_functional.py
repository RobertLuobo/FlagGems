# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

import logging

from .batch_norm import batch_norm

logger = logging.getLogger(__name__)


def _native_batch_norm_legit_functional(
    input,
    weight=None,
    bias=None,
    running_mean=None,
    running_var=None,
    training=False,
    momentum=0.1,
    eps=1e-5,
):
    logger.debug("GEMS_KUNLUNXIN _NATIVE_BATCH_NORM_LEGIT_FUNCTIONAL")
    # The functional variant's contract (per the CPU aten reference used by the
    # --ref cpu tests): the returned running stats are the UPDATED values for
    # every float dtype (fp16/bf16 included) and running_var is folded with the
    # UNBIASED batch variance (var * count / (count - 1)). `batch_norm` defaults
    # to the torch@XPU F.batch_norm semantics (fp32-only in-place update, biased
    # var) for `_batch_norm_impl_index`; the two flags override it here.
    output, save_mean, save_invstd = batch_norm(
        input,
        weight,
        bias,
        running_mean,
        running_var,
        training,
        momentum,
        eps,
        update_running_all_dtypes=True,
        unbiased_running_var=True,
    )
    return output, save_mean, save_invstd, running_mean, running_var
