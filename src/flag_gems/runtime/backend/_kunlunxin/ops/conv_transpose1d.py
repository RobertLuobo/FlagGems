import logging

import torch

logger = logging.getLogger(__name__)

# Composite-implicit dispatch keyset: redispatching with only this keyset skips
# the composite wrapper and lands directly on the XPU/XDNN device kernel.
_FALLBACK_KEYSET = torch._C.DispatchKeySet(torch._C.DispatchKey.CompositeImplicitAutograd)


def _single(value):
    return [value] if isinstance(value, int) else value


def conv_transpose1d(
    input,
    weight,
    bias=None,
    stride=1,
    padding=0,
    output_padding=0,
    groups=1,
    dilation=1,
):
    logger.debug("GEMS_KUNLUNXIN CONV_TRANSPOSE1D")
    # Dispatch straight to the XDNN device kernel, bypassing the composite
    # wrapper. Native dtypes are passed through untouched: the XDNN kernel
    # supports fp16/bf16/fp32 directly, so no upcast (which would halve
    # throughput on the XPU) is required.
    output = torch.ops.aten.conv_transpose1d.default.redispatch(
        _FALLBACK_KEYSET,
        input,
        weight,
        bias,
        _single(stride),
        _single(padding),
        _single(output_padding),
        groups,
        _single(dilation),
    )
    return output
