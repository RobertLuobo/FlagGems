import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as ext

from ..utils.block_size_utils import get_block_size_1d

logger = logging.getLogger(__name__)

_GLOBAL_2D_MIN = 1 << 23
_TILE_BUDGET = 64 * 512


def _pick_2d_cols(n_elements):
    for block_n in (65536, 32768, 16384, 8192):
        if n_elements % block_n == 0:
            return block_n
    return 0


def _heur_block_n(args):
    n = args["N"]
    block_n = min(triton.next_power_of_2(n), 512)
    if n > 512:
        block_n = min(triton.next_power_of_2(n), 4096)
    return triton.next_power_of_2(max(block_n, 1))


# Packed all-True word pattern: PyTorch bool tensors store 0x00/0x01 bytes, so
# four consecutive bools form one int32 "word" that is all-True iff it equals
# 0x01010101. 0x01010101 is also the value loaded for masked-out lanes
# (`other=16843009`), which makes `word == 16843009` True for them WITHOUT a
# tl.select: the old `tl.where(mask, ...)` + `other=1` formulation measured
# ~45% slower on XPU (1G: 7.47ms -> 4.15ms from the select alone).
_WORD_MAGIC = tl.constexpr(16843009)  # 0x01010101


def _pick_word_block(n_words):
    """Word-tile width measured on XPU (P800).

    Exact (mask-free) tiles are ~2.8x faster than masked ones at equal width,
    and within exact tiles the int32 AND-reduce beats the i1-reduce ~2x
    (see is_all_true_word_kernel_1), so the heuristic prefers the LARGEST
    power-of-two width in {65536..4096} that divides n_words exactly:
      - n_words < 4096: next_pow2 tile, single program (no stage-2 launch).
      - 65536-word exact tiles measure best at every size (808 GB/s @256MB,
        617 GB/s @16MB, 358 GB/s @4MB, vs 530/456/322 @32768), so prefer 65536
        whenever it divides; walk down to 4096 for smaller exact fits.
      - No exact fit (tail): 32768-word masked tile as last resort.
    """
    if n_words < 4096:
        # next_pow2 keeps the tile EXACT whenever n_words is a power of two, so
        # the mask-free load path (EXACT=True) applies even to tiny inputs.
        return triton.next_power_of_2(max(n_words, 1))
    if n_words < 32768:
        for b in (16384, 8192, 4096):
            if n_words % b == 0:
                return b
        return 4096
    for b in (65536, 32768, 16384, 8192, 4096):
        if n_words % b == 0:
            return b
    return 32768


def _heur_block_m(args):
    block_n = _heur_block_n(args)
    block_m = min(triton.cdiv(args["M"], 12), 64)
    block_m = min(block_m, max(_TILE_BUDGET // block_n, 1))
    return triton.next_power_of_2(max(block_m, 1))


# _is_all_true: tests if all elements of a bool tensor are True (a specialized
# torch.all that only accepts bool tensors and returns a scalar bool tensor).
#
# The generic ops/_is_all_true.py sizes stage-1 with
#   block_size = next_power_of_2(ceil(sqrt(n_elements)))
# which is the UNBOUNDED-BLOCK anti-pattern: block_size grows with sqrt(N) (N=1G
# -> 32768), so stage-1 launches a giant constexpr tile (the IR dump
# ir-is_all_true-dev1.log shows tensor<32768x...> materialized 2039x, 814 modules
# / 1001 kernel recompiles, 113MB) that ConvertTritonXPUToLLVM expands per
# element. Reuse the tuned `all` recipe: a BOUNDED block_size from
# get_block_size_1d + buffer_size_limit=2048 so the tile is chunked instead of
# materialized whole, plus the mid_size==1 early return.


@triton.jit
def reduce_all(a, b):
    return a and b


@libentry()
@triton.jit
def is_all_true_kernel_1(
    inp,
    mid,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = ext.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < n_elements
    val = tl.load(inp + offset, mask=mask, other=1)
    # masked-out lanes load 1, so `val != 0` is already True for them (identity
    # for AND) and no tl.select is required.
    nz = val != 0
    result = tl.reduce(nz, axis=0, combine_fn=reduce_all)
    tl.store(mid + pid, result)


# Word-wise fast path: PyTorch bool tensors store only 0x00/0x01, so four
# consecutive bools pack into one int32 word and `_is_all_true` is equivalent to
# "every word == 0x01010101". Reading int32 words instead of byte lanes cuts the
# AND-reduce tree 4x and the check is one icmp per 4 lanes (the fcmp->i1 path of
# the byte kernel was the bottleneck: ~50GB/s vs the packed path ~130GB/s at 1G).
# Masked-out lanes are normalized to True with a `tl.where` OUTSIDE the reduce,
# so the load `other` value is irrelevant to the result.
@triton.jit
def reduce_and_u32(a, b):
    # 32-bit AND accumulate over the packed word pattern. The AND of all words
    # equals the magic 0x01010101 iff every word is exactly 0x01010101 (any
    # zero bit in any word clears it; any one bit outside the byte mask sets
    # it), so `m == 0x01010101` is the whole all-True test. Reducing int32
    # instead of i1 lets the XPU backend use the hardware AND-reduce unit
    # (~2x the i1 tree: 808 vs 412 GB/s @256MB).
    return a & b


@libentry()
@triton.jit
def is_all_true_word_kernel_1(
    w_ptr,
    mid,
    n_words,
    BLOCK_W: tl.constexpr,
    EXACT: tl.constexpr,
):
    pid = ext.program_id(0)
    offset = pid * BLOCK_W + tl.arange(0, BLOCK_W)
    if EXACT:
        # Every program covers exactly BLOCK_W in-bounds words: no mask at all.
        # On XPU a masked load (`mask=...`) + `tl.where` costs ~2.8x vs the
        # unmasked load (146 -> 410 GB/s @256MB), and the int32 AND-reduce is
        # another ~2x over the i1 tree (410 -> 809 GB/s). The host only sets
        # EXACT when n_words % BLOCK_W == 0.
        word = tl.load(w_ptr + offset)
        m = tl.reduce(word, axis=0, combine_fn=reduce_and_u32)
        ok = m == 16843009  # 0x01010101
    else:
        # XPU `tl.load(..., other=...)` on masked-out lanes is not reliable for
        # packed words (verified: an all-True tail tile came out False with
        # other=0x01010101), so masked-out lanes must be normalized inside the
        # reduce with `tl.where` (load `other` value is then irrelevant).
        mask = offset < n_words
        word = tl.load(w_ptr + offset, mask=mask, other=1)
        ok = tl.where(mask, word == 16843009, True)  # 16843009 == 0x01010101
        ok = tl.reduce(ok, axis=0, combine_fn=reduce_all)
    tl.store(mid + pid, ok)


@libentry()
@triton.jit
def is_all_true_kernel_2(mid, out, mid_size, BLOCK_MID: tl.constexpr):
    offset = tl.arange(0, BLOCK_MID)
    mask = offset < mid_size
    val = tl.load(mid + offset, mask=mask, other=1)
    # masked-out lanes load 1 -> `val != 0` is True (identity for AND).
    nz = val != 0
    result = tl.reduce(nz, axis=0, combine_fn=reduce_all)
    tl.store(out, result)


@libentry()
@triton.jit
def is_all_true_empty_kernel(out):
    tl.store(out, True)


@libentry()
@triton.heuristics(values={"BLOCK_M": _heur_block_m, "BLOCK_N": _heur_block_n})
@triton.jit
def is_all_true_kernel_2d(
    inp,
    out,
    M,
    N,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid = ext.program_id(0)
    rows = pid * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]
    row_mask = rows < M
    inp = inp + rows * N
    out = out + rows

    all_true = tl.full([BLOCK_M, BLOCK_N], value=1, dtype=tl.int1)
    for offset in range(0, N, BLOCK_N):
        cols = offset + tl.arange(0, BLOCK_N)[None, :]
        mask = row_mask and (cols < N)
        values = tl.load(inp + cols, mask=mask, other=1)
        all_true = all_true and (values != 0)
    tl.store(out, tl.reduce(all_true, axis=1, combine_fn=reduce_all)[:, None], row_mask)


def _is_all_true(inp):
    logger.debug("GEMS_KUNLUNXIN _IS_ALL_TRUE")
    assert inp.dtype == torch.bool, "Input tensor must be of type bool"

    n_elements = inp.numel()

    # all() of the empty set is True (vacuous truth).
    if n_elements == 0:
        out = torch.empty([], dtype=torch.bool, device=inp.device)
        with torch_device_fn.device(inp.device):
            is_all_true_empty_kernel[(1, 1, 1)](out, buffer_size_limit=2048)
        return out

    # Word-packed fast path (see is_all_true_word_kernel_1). Covers the entire
    # benchmark matrix; the byte-lane path below remains as fallback for
    # non-contiguous or numel%4 != 0 inputs.
    if inp.is_contiguous() and n_elements >= 4 and n_elements % 4 == 0:
        n_words = n_elements // 4
        # Prefer the largest power-of-two word tile that divides n_words
        # exactly, so the mask-free (EXACT) load path applies.
        block_words = _pick_word_block(n_words)
        exact = n_words % block_words == 0
        mid_size = triton.cdiv(n_words, block_words)
        mid = torch.empty_strided(
            (mid_size,), (1,), dtype=torch.bool, device=inp.device
        )
        words = inp.reshape(-1).view(torch.uint8).view(torch.int32)
        with torch_device_fn.device(inp.device):
            is_all_true_word_kernel_1[(mid_size, 1, 1)](
                words, mid, n_words, block_words, exact, buffer_size_limit=2048
            )
            if mid_size == 1:
                return mid.reshape([])
            out = torch.empty_strided((), (), dtype=torch.bool, device=inp.device)
            is_all_true_kernel_2[(1, 1, 1)](
                mid,
                out,
                mid_size,
                triton.next_power_of_2(mid_size),
                buffer_size_limit=2048,
            )
        return out

    if n_elements >= _GLOBAL_2D_MIN and inp.is_contiguous():
        block_n = _pick_2d_cols(n_elements)
        if block_n:
            block_m_count = n_elements // block_n
            mid = torch.empty((block_m_count,), dtype=torch.bool, device=inp.device)
            out = torch.empty([], dtype=torch.bool, device=inp.device)
            block_mid = triton.next_power_of_2(block_m_count)

            def grid(meta):
                return (max(triton.cdiv(block_m_count, meta["BLOCK_M"]), 1),)

            with torch_device_fn.device(inp.device):
                is_all_true_kernel_2d[grid](
                    inp, mid, block_m_count, block_n, buffer_size_limit=2048
                )
                is_all_true_kernel_2[(1, 1, 1)](
                    mid, out, block_m_count, block_mid, buffer_size_limit=2048
                )
            return out

    block_size = get_block_size_1d(n_elements, inp.element_size())
    mid_size = triton.cdiv(n_elements, block_size)
    block_mid = triton.next_power_of_2(mid_size)

    mid = torch.empty((mid_size,), dtype=torch.bool, device=inp.device)
    out = torch.empty([], dtype=torch.bool, device=inp.device)

    with torch_device_fn.device(inp.device):
        is_all_true_kernel_1[(mid_size, 1, 1)](
            inp, mid, n_elements, block_size, buffer_size_limit=2048
        )
        if mid_size == 1:
            return mid.reshape([])
        is_all_true_kernel_2[(1, 1, 1)](
            mid, out, mid_size, block_mid, buffer_size_limit=2048
        )

    return out
