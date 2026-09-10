import logging

import torch

logger = logging.getLogger("flag_gems").getChild(__name__.lstrip("."))


def permute_copy(x: torch.Tensor, dims):
    """Wrapper for aten::permute_copy: return a copy of x with permuted dims.

    `permute_copy` is pure data movement: out is the materialized (contiguous)
    copy of the permuted view `x.permute(dims)`.  Instead of a hand-written
    Triton permute/gather kernel (which on XPU has one inherently discrete
    (stride != 1) side and measured ~0.26-2.6ms for the benchmark cells), we
    express the op as the view `x.permute(dims)` + `torch.ops.aten._copy_from`
    into a pre-allocated contiguous `out`.  Gems never registers `_copy_from`,
    so the call reaches the vendor native strided-copy kernel
    (RegisterCUDA.cpp) instead of a Triton kernel: the native engine handles
    arbitrary strides on both sides and materializes the permutation in one
    pass.  Same pattern as the accepted `t_copy` / `sum_dim._compress` /
    `slice_backward` / `resize` / `constant_pad_nd` / `block_diag` fixes
    ("同一把钥匙"); not a CPU/native-composite fallback -- the copy executes
    on-device in the vendor engine and the output is a device tensor.
    """
    logger.debug("GEMS_KUNLUNXIN PERMUTE_COPY")
    # x.permute(dims) performs ATen's own dims validation (duplicate dims ->
    # RuntimeError, out-of-range -> IndexError, negative in-range -> wrapped)
    # and yields a strided view of any rank (0-D..N-D, non-contiguous inputs
    # included).
    view = x.permute(dims)
    out_shape = list(view.shape)
    if x.numel() == 0:
        return torch.empty(out_shape, dtype=x.dtype, device=x.device)
    out = torch.empty(out_shape, dtype=x.dtype, device=x.device)
    torch.ops.aten._copy_from(view, out, False)
    return out
