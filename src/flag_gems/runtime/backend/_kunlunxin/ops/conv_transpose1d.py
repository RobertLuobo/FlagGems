import logging

import torch

logger = logging.getLogger(__name__)

_FALLBACK_KEYSET = torch._C.DispatchKeySet(
    torch._C.DispatchKey.CompositeImplicitAutograd
)


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
    # The vendor XDNN conv_transpose1d loads fp16/bf16 inputs and accumulates in
    # fp32 internally (verified: passing an already-upcast fp32 input yields
    # bit-identical output to the native low-precision path), so the historical
    # explicit upcast only added two extra copies and a second kernel launch per
    # call.  Pass the tensors through unchanged.
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
