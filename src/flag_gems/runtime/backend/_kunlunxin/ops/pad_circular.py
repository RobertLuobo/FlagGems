# Copyright 2026, The FlagOS Contributors.
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

logger = logging.getLogger(__name__)


def _pad_circular_dim(t: torch.Tensor, d: int, pl: int, pr: int) -> torch.Tensor:
    """Circular-pad a single dim d of t by (pl, pr) using affine narrow + copy.

    out[k] = t[(k - pl) mod S] for k in [0, So).  Because |pl|,|pr| <= S the wrap
    happens at most once, so the output splits into three contiguous affine
    segments (left-wrap / core / right-wrap). Each segment is a stride-1 copy
    (torch.narrow view + Tensor.copy_ == _copy_from), avoiding the discrete
    gather that miscompiles / is bandwidth-bound on XPU3.
    """
    S = t.shape[d]
    So = S + pl + pr
    out_shape = list(t.shape)
    out_shape[d] = So
    out = torch.empty(out_shape, dtype=t.dtype, device=t.device)

    # core: output positions k where (k - pl) in [0, S)
    k_core_lo = max(0, pl)
    k_core_hi = min(So, pl + S)
    core_len = k_core_hi - k_core_lo
    if core_len > 0:
        out.narrow(d, k_core_lo, core_len).copy_(
            t.narrow(d, k_core_lo - pl, core_len)
        )
    # left wrap: k in [0, pl) -> source t[k - pl + S]
    if pl > 0:
        out.narrow(d, 0, pl).copy_(t.narrow(d, S - pl, pl))
    # right wrap: k in [pl + S, So) -> source t[k - pl - S]
    if pr > 0:
        out.narrow(d, pl + S, pr).copy_(t.narrow(d, 0, pr))
    return out


def _pad_circular(x: torch.Tensor, pad):
    logger.debug("GEMS_KUNLUNXIN _PAD_CIRCULAR")

    ndim = x.dim()
    if len(pad) % 2 != 0:
        raise ValueError("padding length must be even")
    npad = len(pad) // 2
    if npad < 1 or npad > 3:
        raise ValueError(
            "flag_gems _pad_circular supports padding the last 1, 2 or 3 dimensions"
        )
    if ndim <= npad:
        raise ValueError("input must have at least one non-padded (batch) dimension")

    x = x.contiguous()
    in_shape = list(x.shape)
    for i in range(npad):
        d = ndim - 1 - i
        pl = int(pad[2 * i])
        pr = int(pad[2 * i + 1])
        if in_shape[d] + pl + pr <= 0:
            raise ValueError(
                "negative padding removed the whole dimension "
                f"{d}: input {in_shape[d]} pad ({pl}, {pr})"
            )
        if pl > in_shape[d] or pr > in_shape[d]:
            raise ValueError("Padding value causes wrapping around more than once.")

    out = x
    for i in range(npad):
        d = ndim - 1 - i
        pl = int(pad[2 * i])
        pr = int(pad[2 * i + 1])
        out = _pad_circular_dim(out, d, pl, pr)
    return out.contiguous()
