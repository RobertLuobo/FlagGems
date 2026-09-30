import logging

import torch
import triton

from flag_gems.ops.native_channel_shuffle import _native_channel_shuffle_kernel
from flag_gems.runtime import torch_device_fn

logger = logging.getLogger("flag_gems.ops.native_channel_shuffle")


def native_channel_shuffle(input: torch.Tensor, groups: int) -> torch.Tensor:
    logger.debug("GEMS_KUNLUNXIN NATIVE_CHANNEL_SHUFFLE")
    x = input
    if not x.is_contiguous():
        x = x.contiguous()

    if x.ndim < 2:
        raise ValueError(
            f"Input must have at least 2 dimensions (N, C, ...), got {x.ndim}"
        )

    C = x.shape[1]
    HW = 1
    for d in x.shape[2:]:
        HW *= d

    g = int(groups)
    assert g > 0, "groups must be > 0"
    assert C % g == 0, f"C ({C}) must be divisible by groups ({g})"

    out = torch.empty_like(x)
    numel = x.numel()
    if numel == 0:
        return out

    cpg = C // g

    BLOCK = 1024
    grid = (triton.cdiv(numel, BLOCK),)
    with torch_device_fn.device(x.device):
        _native_channel_shuffle_kernel[grid](
            x,
            out,
            numel,
            C,
            HW,
            cpg,
            g,
            BLOCK=BLOCK,
        )
    return out
