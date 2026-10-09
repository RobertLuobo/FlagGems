import logging

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)

# Bounded, backend-proven flat tile (matches pad_sequence._PAD_BLOCK); a plain
# contiguous load/store block that stays clear of the tile hazards recorded for
# this backend.
_PAD_BLOCK = 4096


@triton.jit
def _ppseq_fill_kernel(out_ptr, n, value, BLOCK: tl.constexpr):
    """Write ``value`` into every element of the contiguous ``n``-element
    buffer. Flat data-parallel fill (gems Triton replacement for torch.full)."""
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    tl.store(
        out_ptr + offs,
        tl.full((BLOCK,), value, dtype=out_ptr.dtype.element_ty),
        mask=mask,
    )


@triton.jit
def _ppseq_scatter_kernel(
    src_ptr,
    dst_ptr,
    dst_row,
    feature: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Scatter packed row ``r`` into its padded destination row.

    Program ``(r, c)`` moves the contiguous ``feature`` block of packed row
    ``r`` (source rows are contiguous) into the contiguous destination row
    ``dst_row[r]`` of the padded output. Both source and destination feature
    blocks are contiguous (no striding), so no masked/scattered padding store
    is involved -- the padded tail is handled entirely by the pre-fill above.
    """
    pid_r = tl.program_id(0)
    pid_c = tl.program_id(1)
    offs = pid_c * BLOCK + tl.arange(0, BLOCK)
    mask = offs < feature
    out_row = tl.load(dst_row + pid_r)
    vals = tl.load(src_ptr + pid_r * feature + offs, mask=mask)
    tl.store(dst_ptr + out_row * feature + offs, vals, mask=mask)


def _pad_packed_sequence(data, batch_sizes, batch_first, padding_value, total_length):
    """Pad a packed variable-length sequence back into a dense tensor (XPU).

    XPU rewrite of the generic implementation: the generic kernel scatters each
    (t, b) row and fills the padded tail with a masked/scalar ``tl.store`` of
    ``padding_value``. That scattered padding store is not honoured on this
    backend (same symptom recorded for pad_sequence), so the ``torch.empty``
    output keeps garbage in every padded position -> maxdiff ~1e1.

    Here the output is built directly in the final (T, B, *) / (B, T, *) layout,
    pre-filled with ``padding_value`` via a flat gems Triton fill kernel (covers
    every padded element), then every packed row is copied into its destination
    row through a contiguous gems Triton scatter kernel driven by a CPU-computed
    destination-row index. No numeric / data-movement torch fallback is used
    (``torch.empty`` for allocation and tiny host-side metadata arithmetic only).
    """
    logger.debug("GEMS_KUNLUNXIN _PAD_PACKED_SEQUENCE")

    num_steps = batch_sizes.numel()
    max_batch = int(batch_sizes[0].item())

    total_steps = num_steps
    if total_length > 0:
        total_steps = max(total_steps, int(total_length))

    # Host-side metadata (tensors are tiny and already CPU-resident): exclusive
    # prefix sum of batch_sizes (packed offset per step), per-element lengths,
    # and the per-packed-row destination index in the padded output.
    batch_sizes_cpu = batch_sizes.to(dtype=torch.int64, device="cpu")
    time_offsets_cpu = torch.zeros_like(batch_sizes_cpu)
    time_offsets_cpu[1:] = torch.cumsum(batch_sizes_cpu[:-1], dim=0)
    lengths = torch.searchsorted(
        torch.flip(batch_sizes_cpu, [0]),
        torch.arange(max_batch, dtype=torch.int64),
        right=True,
    )
    lengths = (num_steps - lengths).to(torch.int64)

    total = int(batch_sizes_cpu.sum().item())

    feature_shape = data.shape[1:]
    feature_size = 1
    for s in feature_shape:
        feature_size *= s

    if batch_first:
        out_shape = (max_batch, total_steps, *feature_shape)
    else:
        out_shape = (total_steps, max_batch, *feature_shape)

    output = torch.empty(out_shape, dtype=data.dtype, device=data.device)
    n = output.numel()
    if n == 0:
        return output, lengths.cpu()

    # Pre-fill the whole buffer so every padded position is written.
    grid_fill = (triton.cdiv(n, _PAD_BLOCK),)
    _ppseq_fill_kernel[grid_fill](
        output.reshape(-1), n, padding_value, BLOCK=_PAD_BLOCK, num_warps=4
    )

    if total == 0:
        return output, lengths.cpu()

    # Per packed row r -> (t, b); destination flat-row index in `output`.
    t_per_row = torch.repeat_interleave(
        torch.arange(num_steps, dtype=torch.int64), batch_sizes_cpu
    )
    b_per_row = torch.arange(total, dtype=torch.int64) - time_offsets_cpu[t_per_row]
    if batch_first:
        dst_row_cpu = b_per_row * total_steps + t_per_row
    else:
        dst_row_cpu = t_per_row * max_batch + b_per_row
    dst_row = dst_row_cpu.to(data.device)

    BLOCK = min(_PAD_BLOCK, max(1, triton.next_power_of_2(feature_size)))
    grid_copy = (total, triton.cdiv(feature_size, BLOCK))
    _ppseq_scatter_kernel[grid_copy](
        data.reshape(-1),
        output.reshape(-1),
        dst_row,
        feature=feature_size,
        BLOCK=BLOCK,
        num_warps=4,
    )

    return output, lengths.cpu()
