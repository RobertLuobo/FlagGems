import logging
import math

import torch
import triton
import triton.language as tl

from flag_gems.ops.sparse_sampled_addmm import _broadcast_sparse_csr
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as ext

logger = logging.getLogger(__name__)


_SAMPLED_ADDMM_DTYPES = {
    torch.float16,
    torch.bfloat16,
    torch.float32,
    torch.float64,
}

# The dense scratch buffer is padded so that *every* tile in the autotune pool
# divides it exactly.  That lets the GEMM kernel store its tile with no mask at
# all (masked 2D block stores are known to overrun on this backend).
#
# INVARIANT: every TILE_M / TILE_N in `_gemm_autotune` must divide _DENSE_ALIGN.
# If a tile with a non-256 factor (192, 384, ...) is ever added to that pool,
# _DENSE_ALIGN has to be adjusted too, otherwise the unmasked GEMM store runs
# past the end of `dense`.
_DENSE_ALIGN = 256
# Tile width for the fused gather pass.  Swept on device; 4096 wins for large
# per-batch nnz, 2048 for small ones (the MASKED tail is at most one block).
_COMBINE_BLOCK_SMALL = 2048
_COMBINE_BLOCK_LARGE = 4096
_COMBINE_LARGE_NNZ = 64 * 1024


def _combine_block(nnz_per_batch):
    return (
        _COMBINE_BLOCK_LARGE
        if nnz_per_batch > _COMBINE_LARGE_NNZ
        else _COMBINE_BLOCK_SMALL
    )


def _heur_group_m(args):
    if args["TILE_M"] > args["TILE_N"]:
        return 1
    return (args["M"] + args["TILE_M"] - 1) // args["TILE_M"]


def _heur_divisible_m(args):
    return args["M"] % args["TILE_M"] == 0


def _heur_divisible_n(args):
    return args["N"] % args["TILE_N"] == 0


def _heur_divisible_k(args):
    return args["K"] % args["TILE_K"] == 0


_gemm_autotune = triton.autotune(
    configs=[
        triton.Config({"TILE_M": 256, "TILE_N": 256, "TILE_K": 256}),
        triton.Config({"TILE_M": 256, "TILE_N": 256, "TILE_K": 128}),
        triton.Config({"TILE_M": 128, "TILE_N": 128, "TILE_K": 128}),
        triton.Config({"TILE_M": 16, "TILE_N": 16, "TILE_K": 16}),
    ],
    key=["M", "N", "K"],
)


@libentry()
@_gemm_autotune
@triton.heuristics(
    {
        "GROUP_M": _heur_group_m,
        "DIVISIBLE_M": _heur_divisible_m,
        "DIVISIBLE_N": _heur_divisible_n,
        "DIVISIBLE_K": _heur_divisible_k,
    }
)
@triton.jit
def _ssa_gemm_kernel(
    A,
    B,
    Out,
    M,
    N,
    K,
    OUT_ROW_STRIDE,
    OUT_BATCH_STRIDE,
    TILE_M: tl.constexpr,
    TILE_N: tl.constexpr,
    TILE_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    DIVISIBLE_M: tl.constexpr,
    DIVISIBLE_N: tl.constexpr,
    DIVISIBLE_K: tl.constexpr,
):
    pid_b = ext.program_id(2)
    b64 = pid_b.to(tl.int64)
    A += b64 * M * K
    B += b64 * K * N
    Out += b64 * OUT_BATCH_STRIDE

    pidx = ext.program_id(0)
    pidy = ext.program_id(1)

    if GROUP_M == 1:
        pid_m, pid_n = pidx, pidy
    else:
        gridx = ext.num_programs(0)
        gridy = ext.num_programs(1)
        pid = pidx + pidy * gridx
        num_CTA_per_group = gridy * GROUP_M
        group_id = pid // num_CTA_per_group
        inner_group_id = pid % num_CTA_per_group
        GROUP_SIZE = tl.where(
            (group_id * GROUP_M + GROUP_M) > gridx, gridx % GROUP_M, GROUP_M
        )
        pid_m = group_id * GROUP_M + inner_group_id % GROUP_SIZE
        pid_n = inner_group_id // GROUP_SIZE

    offs_m = pid_m * TILE_M + tl.arange(0, TILE_M)
    offs_n = pid_n * TILE_N + tl.arange(0, TILE_N)
    offs_k = tl.arange(0, TILE_K)

    if not DIVISIBLE_M:
        mask_m = offs_m < M
    if not DIVISIBLE_N:
        mask_n = offs_n < N

    a_ptrs = A + offs_m[:, None] * K + offs_k[None, :]
    b_ptrs = B + offs_k[:, None] * N + offs_n[None, :]

    acc = tl.zeros((TILE_M, TILE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, TILE_K)):
        if DIVISIBLE_K:
            if DIVISIBLE_M:
                mask_a = tl.full([TILE_M, TILE_K], value=1, dtype=tl.int1)
            else:
                mask_a = mask_m[:, None]
            if DIVISIBLE_N:
                mask_b = tl.full([TILE_K, TILE_N], value=1, dtype=tl.int1)
            else:
                mask_b = mask_n[None, :]
        else:
            mask_k_row = offs_k[None, :] < K - k * TILE_K
            mask_k_col = offs_k[:, None] < K - k * TILE_K
            if DIVISIBLE_M:
                mask_a = mask_k_row
            else:
                mask_a = mask_m[:, None] & mask_k_row
            if DIVISIBLE_N:
                mask_b = mask_k_col
            else:
                mask_b = mask_k_col & mask_n[None, :]

        a = tl.load(a_ptrs, mask_a, other=0.0)
        b_val = tl.load(b_ptrs, mask_b, other=0.0)
        a_ptrs += TILE_K
        b_ptrs += TILE_K * N
        acc += tl.dot(a, b_val, out_dtype=tl.float32, allow_tf32=False)

    o_ptrs = Out + offs_m[:, None] * OUT_ROW_STRIDE + offs_n[None, :]
    tl.store(o_ptrs, acc)


@libentry()
@triton.jit(do_not_specialize=["alpha", "beta"])
def _ssa_combine_kernel(
    row_ptr,
    col_ptr,
    dense_ptr,
    val_ptr,
    nnz_per_batch,
    start,
    OUT_ROW_STRIDE,
    OUT_BATCH_STRIDE,
    alpha,
    beta,
    BLOCK: tl.constexpr,
    MASKED: tl.constexpr,
):
    # Flat (per-batch) gather: element `off` of batch `b` is the row's `rows`
    # array index, its column comes from `col`, and `dense` holds the dense
    # product.  out_val[off] = alpha * dense[b, row, col] + beta * val[off].
    # The row identity comes from a precomputed int32 `rows` array (one entry
    # per element) instead of per-row programs; the per-row kernel launch
    # overhead is the dominant cost on this backend.
    b = ext.program_id(1).to(tl.int64)
    off = start + ext.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    base = b * nnz_per_batch
    if MASKED:
        keep = off < nnz_per_batch
        safe = tl.where(keep, off, 0)
        r = tl.load(row_ptr + base + safe).to(tl.int64)
        c = tl.load(col_ptr + base + safe).to(tl.int64)
        d = tl.load(
            dense_ptr + b * OUT_BATCH_STRIDE + r * OUT_ROW_STRIDE + c,
            mask=keep,
            other=0.0,
        )
        v = tl.load(val_ptr + base + safe)
        res = alpha * d + beta * v.to(tl.float32)
        tl.store(val_ptr + base + off, res.to(v.dtype), mask=keep)
    else:
        r = tl.load(row_ptr + base + off).to(tl.int64)
        c = tl.load(col_ptr + base + off).to(tl.int64)
        d = tl.load(dense_ptr + b * OUT_BATCH_STRIDE + r * OUT_ROW_STRIDE + c)
        v = tl.load(val_ptr + base + off)
        res = alpha * d + beta * v.to(tl.float32)
        tl.store(val_ptr + base + off, res.to(v.dtype))


def _align_up(x, a):
    return ((x + a - 1) // a) * a


def _sparse_sampled_addmm_impl(input, mat1, mat2, *, beta=1.0, alpha=1.0, out=None):
    if input.layout != torch.sparse_csr:
        raise RuntimeError(
            f"sparse_sampled_addmm: Expected input to have sparse csr layout, "
            f"but got {input.layout}"
        )
    if mat1.layout != torch.strided:
        raise RuntimeError(
            f"sparse_sampled_addmm: Expected mat1 to have strided layout, "
            f"but got {mat1.layout}"
        )
    if mat2.layout != torch.strided:
        raise RuntimeError(
            f"sparse_sampled_addmm: Expected mat2 to have strided layout, "
            f"but got {mat2.layout}"
        )
    if out is not None and out.layout != torch.sparse_csr:
        raise RuntimeError(
            f"sparse_sampled_addmm: Expected out to have sparse csr layout, "
            f"but got {out.layout}"
        )

    if input.dtype not in _SAMPLED_ADDMM_DTYPES:
        raise RuntimeError(
            f"sparse_sampled_addmm: Expected input to be floating-point, "
            f"but got {input.dtype}"
        )
    if input.dtype != mat1.dtype or input.dtype != mat2.dtype:
        raise RuntimeError(
            f"sparse_sampled_addmm: Expected all inputs to have the same dtype, "
            f"but got input={input.dtype}, mat1={mat1.dtype}, mat2={mat2.dtype}"
        )

    if input.dense_dim() != 0:
        raise RuntimeError("sparse_sampled_addmm: Expected non-hybrid input tensor")
    if out is not None and out.dense_dim() != 0:
        raise RuntimeError("sparse_sampled_addmm: Expected non-hybrid out tensor")

    if mat1.dim() < 2 or mat2.dim() < 2:
        raise RuntimeError(
            "sparse_sampled_addmm: Expected mat1 and mat2 to be at least 2-D matrices"
        )

    batch_dims = mat1.shape[:-2]
    M, K = mat1.shape[-2:]
    N = mat2.shape[-1]

    if mat2.shape[:-2] != batch_dims:
        raise RuntimeError(
            "sparse_sampled_addmm: Expected mat1 and mat2 to have the same batch size"
        )
    if input.dim() > 2 and input.shape[:-2] != batch_dims:
        raise RuntimeError(
            "sparse_sampled_addmm: Expected input and mat1 to have the same batch size"
        )
    if input.shape[-2] != M or input.shape[-1] != N:
        raise RuntimeError(
            "sparse_sampled_addmm: input.shape[-2:] must match (M, N) of mat1 @ mat2"
        )
    if mat2.shape[-2] != K:
        raise RuntimeError(
            "sparse_sampled_addmm: mat1 and mat2 shapes cannot be multiplied"
        )

    out_shape = batch_dims + (M, N)
    B = math.prod(batch_dims) if batch_dims else 1

    nnz_per_batch = input._nnz()
    nnz = nnz_per_batch * B

    if out is None:
        out = _broadcast_sparse_csr(input, out_shape)
    else:
        if out.shape != out_shape:
            raise RuntimeError(
                f"sparse_sampled_addmm: Expected out shape {out_shape}, got {out.shape}"
            )
        if out._nnz() != nnz_per_batch:
            raise RuntimeError(
                f"sparse_sampled_addmm: Expected out nnz per batch {nnz_per_batch}, "
                f"got {out._nnz()}"
            )
        if out is not input:
            out.copy_(_broadcast_sparse_csr(input, out_shape))

    if mat1.numel() == 0 or mat2.numel() == 0 or nnz == 0 or alpha == 0.0 or K == 0:
        out.values().mul_(beta)
        return out

    mat1_f = mat1.contiguous().reshape(B, M, K)
    mat2_f = mat2.contiguous().reshape(B, K, N)
    val_f = out.values().reshape(nnz)
    crow_2d = out.crow_indices().reshape(B, M + 1).contiguous()
    col_2d = out.col_indices().reshape(B, nnz_per_batch).contiguous()

    # Padded dense scratch: the GEMM writes every tile unmasked into it, and the
    # combine pass gathers only the (row, col) slots that are structurally
    # non-zero, so the padding is never read.
    Mp = _align_up(M, _DENSE_ALIGN)
    Np = _align_up(N, _DENSE_ALIGN)
    dense = torch.empty((B, Mp, Np), dtype=torch.float32, device=input.device)

    # pos[b * nnz_per_batch + e] = flat offset of value e inside `dense`.
    # The row identity is materialized once per batch as an int32 array over
    # all non-zeros (row_arr[e]) with a single repeat_interleave; a flat
    # per-batch kernel then gathers dense[b, row_arr[e], col[e]] directly,
    # avoiding the per-row program launch chain that dominates on this backend.
    combine_block = _combine_block(nnz_per_batch)

    logger.debug(
        "GEMS_KUNLUNXIN SPARSE_SAMPLED_ADDMM, [shape info]: batch=%s, M=%s, N=%s, "
        "K=%s, nnz=%s, Mp=%s, Np=%s",
        B,
        M,
        N,
        K,
        nnz,
        Mp,
        Np,
    )

    grid_gemm = lambda meta: (  # noqa: E731
        triton.cdiv(M, meta["TILE_M"]),
        triton.cdiv(N, meta["TILE_N"]),
        B,
    )
    n_full = nnz_per_batch // combine_block
    tail = n_full * combine_block

    with torch_device_fn.device(input.device):
        # Row identity: row_arr[b * nnz_per_batch + e] == row of element e.
        lengths = (crow_2d[:, 1:] - crow_2d[:, :-1]).to(torch.int32).reshape(-1)
        rows = torch.arange(M, dtype=torch.int32, device=input.device)
        rows = rows.expand(B, M).reshape(-1)
        row_arr = torch.repeat_interleave(rows, lengths, output_size=nnz)
        _ssa_gemm_kernel[grid_gemm](
            mat1_f,
            mat2_f,
            dense,
            M,
            N,
            K,
            Np,
            Mp * Np,
        )
        if n_full > 0:
            _ssa_combine_kernel[(n_full, B)](
                row_arr,
                col_2d,
                dense,
                val_f,
                nnz_per_batch,
                0,
                Np,
                Mp * Np,
                alpha,
                beta,
                BLOCK=combine_block,
                MASKED=False,
            )
        if tail < nnz_per_batch:
            _ssa_combine_kernel[(1, B)](
                row_arr,
                col_2d,
                dense,
                val_f,
                nnz_per_batch,
                tail,
                Np,
                Mp * Np,
                alpha,
                beta,
                BLOCK=combine_block,
                MASKED=True,
            )

    return out


def sparse_sampled_addmm(input, mat1, mat2, *, beta=1.0, alpha=1.0):
    logger.debug("GEMS_KUNLUNXIN SPARSE_SAMPLED_ADDMM")
    return _sparse_sampled_addmm_impl(input, mat1, mat2, beta=beta, alpha=alpha)


def sparse_sampled_addmm_out(input, mat1, mat2, *, beta=1.0, alpha=1.0, out=None):
    logger.debug("GEMS_KUNLUNXIN SPARSE_SAMPLED_ADDMM_OUT")
    if out is None:
        raise TypeError("sparse_sampled_addmm(): out must be provided for out variant")
    return _sparse_sampled_addmm_impl(
        input, mat1, mat2, beta=beta, alpha=alpha, out=out
    )
