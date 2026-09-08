# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

# NOTE: hand-written flat-1D kernels instead of the pointwise_dynamic codegens.
#  - The generic codegen (512-tile) was the baseline's slow path (~0.77x average:
#    per-element integer-division indexing + always-true predicated access).
#  - The vendor codegen (kunlunAutoGrid) derives an unbounded 1D tile
#    tile = next_power_of_2(cdiv(numel, 12)): (16,128,64,1280) and (4096,4096)
#    land on 2^24 entries and deterministically hang the device (NOC idle
#    timeout, observed 2026-09-08 on XPU 5; same red-zone hazard as
#    special_chebyshev_polynomial_w_out recorded the same day).
#  - Masked loads must NOT pass `other=` (mis-lowered on this backend: a handful
#    of interior lanes return `other`/zero -> U_n(0) garbage; the sibling
#    generic codegen also omits `other`).  Masked-off lanes load garbage but are
#    never stored (store is masked), which is safe.
# Math: U_0=1, U_1=2x, U_k=2x*U_{k-1}-U_{k-2}, selected per element by the
# (integer, guard-validated [0,5]) degree n; computed in fp32.

import logging

import torch

from flag_gems.ops.special_chebyshev_polynomial_u import (
    special_chebyshev_polynomial_u_kernel,
)

logger = logging.getLogger(__name__)


def special_chebyshev_polynomial_u(x, n):
    logger.debug("GEMS_KUNLUNXIN SPECIAL_CHEBYSHEV_POLYNOMIAL_U")
    if isinstance(n, torch.Tensor):
        n = n.to(device=x.device, dtype=torch.int32)
        if torch.any((n < 0) | (n > 5)).item():
            raise ValueError("Chebyshev polynomial order n must be in [0, 5]")
    else:
        n_min = n_max = int(n)
        n = torch.empty((), dtype=torch.int32, device=x.device)
        n.fill_(n_min)
        if n_max > 5 or n_min < 0:
            raise ValueError(
                f"Chebyshev polynomial order n must be in [0, 5], "
                f"got values in [{n_min}, {n_max}]"
            )

    return special_chebyshev_polynomial_u_kernel(x, n)
