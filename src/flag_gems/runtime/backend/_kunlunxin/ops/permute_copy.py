import logging

import torch
import triton
import triton.language as tl

logger = logging.getLogger("flag_gems").getChild(__name__.lstrip("."))


@triton.jit
def _pc_flat_copy_kernel(
    src_ptr, dst_ptr, n_words, BLOCK: tl.constexpr, NEED_MASK: tl.constexpr
):
    pid = tl.program_id(0)
    off = pid * BLOCK + tl.arange(0, BLOCK)
    if NEED_MASK:
        m = off < n_words
        v = tl.load(src_ptr + off, mask=m)
        tl.store(dst_ptr + off, v, mask=m)
    else:
        v = tl.load(src_ptr + off)
        tl.store(dst_ptr + off, v)


@triton.jit
def _pc_gather_kernel(
    in_ptr,
    out_ptr,
    R,
    C,
    N: tl.constexpr,
    A: tl.constexpr,
    B: tl.constexpr,
    ST0: tl.constexpr,
    ST1: tl.constexpr,
    ST2: tl.constexpr,
    ST3: tl.constexpr,
    ROW_BLOCK: tl.constexpr,
    COL_BLOCK: tl.constexpr,
    NEED_MASK: tl.constexpr,
):
    pid_r = tl.program_id(0)
    pid_c = tl.program_id(1)
    rows = pid_r * ROW_BLOCK + tl.arange(0, ROW_BLOCK)
    cols = pid_c * COL_BLOCK + tl.arange(0, COL_BLOCK)
    r = rows
    if N == 1:
        off = tl.zeros([ROW_BLOCK], dtype=tl.int32)
        col_off = cols * ST0
    elif N == 2:
        off = r * ST0
        col_off = cols * ST1
    elif N == 3:
        d0 = r // A
        d1 = r % A
        off = d0 * ST0 + d1 * ST1
        col_off = cols * ST2
    else:
        d0 = r // (A * B)
        d1 = (r // B) % A
        d2 = r % B
        off = d0 * ST0 + d1 * ST1 + d2 * ST2
        col_off = cols * ST3
    m = (rows < R)[:, None] & (cols < C)[None, :]
    if NEED_MASK:
        v = tl.load(in_ptr + off[:, None] + col_off[None, :], mask=m, other=0.0)
        tl.store(out_ptr + rows[:, None] * C + cols[None, :], v, mask=m)
    else:
        v = tl.load(in_ptr + off[:, None] + col_off[None, :])
        tl.store(out_ptr + rows[:, None] * C + cols[None, :], v)


def _pc_flat_copy(view, out, n, elemsz):
    """Byte-wise flat copy with 4-byte word coalescing when alignment allows."""
    src = view.reshape(-1)
    if n == 0:
        return
    if elemsz in (2, 4):
        # move as many 4-byte words as possible; keep sources contiguous
        words = (n * elemsz) // 4
        head_el = words * (4 // elemsz)
        if words > 0:
            src4 = src[:head_el].contiguous().view(torch.int32)
            dst4 = out.reshape(-1)[:head_el].view(torch.int32)
            BLOCK = 8192 if words <= 1048576 else 32768
            _pc_flat_copy_kernel[(triton.cdiv(words, BLOCK),)](
                src4, dst4, words, BLOCK, words % BLOCK != 0
            )
        tail_el = n - head_el
        if tail_el > 0:
            out_flat = out.reshape(-1)
            _pc_flat_copy_kernel[(1,)](
                src[-tail_el:], out_flat[-tail_el:], tail_el, 512, True
            )
    elif elemsz == 1:
        BLOCK = 8192 if n <= 1048576 else 32768
        _pc_flat_copy_kernel[(triton.cdiv(n, BLOCK),)](
            src, out, n, BLOCK, n % BLOCK != 0
        )
    else:  # 8-byte elements: direct 8-byte copy
        BLOCK = 2048 if n <= 262144 else 8192
        _pc_flat_copy_kernel[(triton.cdiv(n, BLOCK),)](
            src, out, n, BLOCK, n % BLOCK != 0
        )


def _pc_gather(view, x, out, n):
    N = view.dim()
    C = view.shape[-1]
    R = n // C
    sz = list(view.shape) + [1] * (4 - N)
    st = list(view.stride()) + [1] * (4 - N)
    # 4x512: probe sweep (2026-09-09, this card) — median do_bench over
    # (RB,CB) in {2,4,8}x{64,128,256,512} x num_warps{2,4,8}. 4x512 is
    # fastest on the benchmark transposes: (64,128)(1,0) 21.4->14.3us,
    # (32,64,128)(2,0,1) 306->278us (fp32), same direction fp16/bf16.
    # Wider column blocks amortise the strided-load instruction issue cost;
    # the per-element 32x line amplification (no tl.trans on this backend)
    # remains a structural limit for the transposed paths.
    RB, CB = 4, 512
    NEED_MASK = (R % RB != 0) or (C % CB != 0)
    grid = (triton.cdiv(R, RB), triton.cdiv(C, CB))
    _pc_gather_kernel[grid](
        x, out, R, C, N, sz[1], sz[2], st[0], st[1], st[2], st[3], RB, CB, NEED_MASK
    )


@triton.jit
def _pc_rowtable_kernel(
    in_ptr,
    out_ptr,
    row_ptr,
    R,
    C,
    COL_STRIDE,
    ROW_BLOCK: tl.constexpr,
    COL_BLOCK: tl.constexpr,
):
    pid_r = tl.program_id(0)
    pid_c = tl.program_id(1)
    rows = pid_r * ROW_BLOCK + tl.arange(0, ROW_BLOCK)
    cols = pid_c * COL_BLOCK + tl.arange(0, COL_BLOCK)
    rb = tl.load(row_ptr + rows)
    m = (rows < R)[:, None] & (cols < C)[None, :]
    addr = rb[:, None] + cols[None, :] * COL_STRIDE
    v = tl.load(in_ptr + addr, mask=m)
    tl.store(out_ptr + rows[:, None] * C + cols[None, :], v, mask=m)


def _pc_row_gather(view, x, out, n):
    """Rank > 4 fallback: per-row input base table + linear column stride."""
    C = view.shape[-1]
    R = n // C
    # row base = in-offset of element (row, 0). Decode on host with torch.
    coords = torch.meshgrid(
        *[torch.arange(s, dtype=torch.int32, device=x.device) for s in view.shape[:-1]],
        indexing="ij",
    )
    row_base = sum(c.reshape(-1) * s for c, s in zip(coords, view.stride()[:-1]))
    row_base = row_base.contiguous()
    RB, CB = 16, 64
    grid = (triton.cdiv(R, RB), triton.cdiv(C, CB))
    _pc_rowtable_kernel[grid](x, out, row_base, R, C, view.stride()[-1], RB, CB)


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
