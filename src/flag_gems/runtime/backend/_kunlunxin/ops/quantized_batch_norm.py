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
import importlib
import logging

import torch

logger = logging.getLogger(__name__)

_generic = importlib.import_module("flag_gems.ops.quantized_batch_norm")

# The generic Triton kernels and functional launcher are correct on XPU: the
# functional variant allocates its result through ``_empty_affine_quantized``
# with the requested qparams, so its quantizer is already right. Reuse it
# verbatim.
quantized_batch_norm = _generic.quantized_batch_norm


def _stamp_quantizer(out, scale, zero_point):
    """Set ``out``'s per-tensor quantizer in place without moving its storage.

    On XPU the QuantizedCUDA ``_copy_from`` bridge copies only the integer
    payload and leaves the destination quantizer stale, so the generic
    ``out.copy_(result)`` cannot update ``out``'s scale/zero_point. No
    ``set_quantizer_`` op/attr is exposed and ``set_``/``quantize_per_tensor.out``
    both preserve the existing quantizer. The one in-place lever is ``.data``,
    which swaps the TensorImpl (quantizer included) while keeping the same
    Python object; a following ``set_`` re-points the object back at its own
    storage/offset/strides and keeps the freshly-stamped quantizer. ``.data``
    is bound to a 1-element ``_empty_affine_quantized`` carrying the target
    qparams -- a pure allocation, no quantized math.
    """
    storage = out.untyped_storage()
    offset = out.storage_offset()
    size = tuple(out.shape)
    stride = out.stride()
    donor = torch._empty_affine_quantized(
        (1,),
        scale=float(scale),
        zero_point=int(zero_point),
        dtype=out.dtype,
        device=out.device,
    )
    out.data = donor
    out.set_(storage, offset, size, stride)
    return out


def quantized_batch_norm_out(
    input,
    weight=None,
    bias=None,
    mean=None,
    var=None,
    eps=1e-05,
    output_scale=1.0,
    output_zero_point=0,
    *,
    out=None,
):
    logger.debug("GEMS_KUNLUNXIN QUANTIZED_BATCH_NORM_OUT")
    logger.debug("GEMS QUANTIZED_BATCH_NORM_OUT")

    # Reuse the generic out-variant for all validation, resize and the integer
    # payload write (its final ``out.copy_(result)`` reaches the vendor
    # ``_copy_from`` bridge, which moves the integers correctly through
    # ``out``'s own strides/offset).
    out = _generic.quantized_batch_norm_out(
        input,
        weight,
        bias,
        mean,
        var,
        eps,
        output_scale,
        output_zero_point,
        out=out,
    )

    # Repair the quantizer the device copy left stale. For an empty input ATen
    # keeps the *input's* qparams (the functional path returns ``input.clone()``);
    # otherwise the requested output qparams.
    if input.numel() == 0:
        _stamp_quantizer(out, input.q_scale(), input.q_zero_point())
    else:
        _stamp_quantizer(out, output_scale, output_zero_point)
    return out
