import logging

import torch
import triton
from _kunlunxin.utils.codegen_config_utils import CodeGenConfig

from flag_gems.ops.copy import copy_ as _gems_copy_

from ..utils.pointwise_dynamic import pointwise_dynamic
from .copy import _copy_flat_kernel, _pick_flat_block
from .expand_copy import _launch_bcast

logger = logging.getLogger("flag_gems").getChild(__name__.lstrip("."))

# The flat gather kernel behind `_launch_bcast` addresses 6 dimensions; deeper
# layouts fall back to the rank-general generic gems copy.
_MAX_FLAT_DIM = 6


config_ = CodeGenConfig(
    512,
    (65536, 65536, 65536),
    32,
    True,
    prefer_1d_tile=True,
    buffer_size_limit=4096,
    isCloseVectorization=False,
    unroll_num=8,
)


@pointwise_dynamic(is_tensor=[True], promotion_methods=[(0, "DEFAULT")], config=config_)
@triton.jit
def _squeeze_copy_flat(src):
    return src


# `pointwise_dynamic` splits a 1d tile over 12 CTAs (`kunlunAutoGrid` is off in
# `config_`), so the per-CTA tile is `next_pow2(cdiv(n_elements, 12))` lanes.
# On this backend a program whose tile is only 256 B wide costs ~0.83 us of
# device time regardless of payload, versus ~0.11 us once the tile reaches
# 1 KiB (XPU per-program cost is set by the tile *byte* width).  Measured on a
# (1024, 1) fp16 input that is 128 lanes = 256 B per CTA, i.e. 12 x 0.83 us
# ~= 10.8 us of device time to move 2 KiB -- 5.5x slower than the reference,
# while the same kernel at 4 KiB tiles costs ~2.2 us.
#
# Below ~12 KiB of payload the 1d tile is therefore always byte-starved, and the
# bounded-tile flat block DMA (the kernel that backs the vendor `copy_`) is
# strictly better.  Above it the 12-CTA pointwise path keeps winning (its large
# tile beats a fixed 64 KiB block by 12-32% at >= 1 M elements), so the two are
# selected by size.  The threshold is exactly the point where the pointwise tile
# stops being narrower than 2 KiB, for any element size:
#   tile_bytes = next_pow2(cdiv(n, 12)) * itemsize < 2048  <=>  n * itemsize < 12288
_FLAT_COPY_BYTE_LIMIT = 12 * 1024


def squeeze_copy(x: torch.Tensor) -> torch.Tensor:
    """Return a copy of ``x`` with every size-1 dimension removed.

    ``aten::squeeze_copy`` (no-dim overload) never aliases its input and never
    reorders elements, so the work is: allocate ``squeezed_shape`` and copy
    ``x.numel()`` elements flat.
    """
    logger.debug("GEMS_KUNLUNXIN SQUEEZE_COPY")
    squeezed_shape = tuple(s for s in x.shape if s != 1)
    out = torch.empty(squeezed_shape, dtype=x.dtype, device=x.device, layout=x.layout)
    n_elements = out.numel()
    if n_elements == 0:
        return out

    if x.is_contiguous():
        src = x
    else:
        # Materialise a contiguous source on gems' own Triton kernels.
        # `x.contiguous()` must NOT be used here: it goes through the native
        # `clone`, which redispatches to the vendor `copy_`, whose strided
        # pointwise branch (`copy_slice`) is both 90-500x slower than a flat
        # block DMA and raises KL_XID_KERNEL_EXCEPTION (status 700) on
        # transposed 2-byte shapes (a (512, 512) fp16 `.t()` wedges
        # deterministically).  `aten::_copy_from` is not used either (gems
        # operators may not delegate their work to it).  `_launch_bcast` is
        # the flat, block-tiled gather behind `expand_copy`: it decodes the
        # flat destination index against `x`'s real strides, so a transposed /
        # stepped / narrowed source is normalised without any vendor copy.
        # This only normalises the layout; the copy the operator is about
        # still runs on the Triton kernels below.
        src = torch.empty(x.shape, dtype=x.dtype, device=x.device)
        if x.dim() <= _MAX_FLAT_DIM:
            _launch_bcast(x.shape, x.stride(), x, src, src.numel())
        else:
            _gems_copy_(src, x)

    if n_elements * x.element_size() <= _FLAT_COPY_BYTE_LIMIT:
        block_size = _pick_flat_block(n_elements)
        _copy_flat_kernel[(triton.cdiv(n_elements, block_size),)](
            src,
            out,
            n_elements,
            BLOCK_SIZE=block_size,
            NEED_MASK=(n_elements % block_size != 0),
            num_warps=32,
            unroll_num=8,
            buffer_size_limit=1024,
        )
    else:
        # `view` (not `reshape`): all views here are metadata-only, and
        # `Tensor.reshape` additionally dispatches through the gems-registered
        # `aten::_reshape_alias`.
        _squeeze_copy_flat(src.view(-1), out0=out.view(-1))
    return out
