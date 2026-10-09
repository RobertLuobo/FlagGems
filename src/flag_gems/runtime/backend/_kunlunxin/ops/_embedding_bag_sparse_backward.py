import logging

import torch

logger = logging.getLogger(__name__)


def _generic_impl(*args):
    from flag_gems.ops._embedding_bag_sparse_backward import (
        _embedding_bag_sparse_backward as _generic,
    )

    return _generic(*args)


def _embedding_bag_sparse_backward(
    grad: torch.Tensor,
    indices: torch.Tensor,
    offsets: torch.Tensor,
    offset2bag: torch.Tensor,
    bag_size: torch.Tensor,
    num_weights: int,
    scale_grad_by_freq: bool,
    mode: int,
    per_sample_weights: torch.Tensor = None,
    padding_idx: int = -1,
) -> torch.Tensor:
    """Sparse-COO embedding_bag backward matching aten value count.

    aten derives the number of output values from ``offset2bag`` (via
    ``grad.index_select(0, offset2bag)``), not from ``indices``. When the
    forward is run with ``sparse=True`` the runtime may return an empty
    ``offset2bag`` (and a zero ``bag_size``), in which case aten produces an
    empty (nnz=0) sparse gradient. The generic kernel instead sizes its work
    from ``indices`` and reads ``offset2bag`` out of bounds. Align the per-sample
    inputs to ``offset2bag`` before delegating so the valid-length and empty
    cases both match the reference.
    """
    logger.debug("GEMS_KUNLUNXIN _EMBEDDING_BAG_SPARSE_BACKWARD")

    n = offset2bag.numel()
    if n != indices.numel():
        indices = indices[:n]
        if per_sample_weights is not None:
            per_sample_weights = per_sample_weights[:n]

    return _generic_impl(
        grad,
        indices,
        offsets,
        offset2bag,
        bag_size,
        num_weights,
        scale_grad_by_freq,
        mode,
        per_sample_weights,
        padding_idx,
    )
