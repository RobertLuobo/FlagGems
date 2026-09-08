import torch
import triton
import triton.language as tl

from flag_gems.utils import libentry

from .linalg_ldl_factor import (
    _SMALL_LDL_MAX_N,
    _check_linalg_ldl_factor,
    _linalg_ldl_factor_v5,
    _linalg_ldl_factor_v6,
)


@libentry()
@triton.jit
def _ldl_init_info_kernel(info, num_batches):
    batch_idx = tl.program_id(0)
    if batch_idx < num_batches:
        tl.store(info + batch_idx, 0)


def ldl_factor_ex(A, hermitian=False, check_errors=False):
    """Kunlunxin LDL factorization (extended: LD, pivots, info).

    Reuses the same v5 (n <= 16, single-launch transposed-workspace) / v6
    (n > 16, fused per-column) kernels as ldl_factor; the extended op only
    adds the info output (0 == success) on top.  The old static
    `_ldl_factor_kernel` (MAX_SIZE=64 fully-scalar loops) was ~100x slower at
    n = 32 while producing identical results.
    """
    _check_linalg_ldl_factor(A, hermitian, check_errors)
    n = A.shape[-1]
    if n <= _SMALL_LDL_MAX_N:
        LD, pivots = _linalg_ldl_factor_v5(A)
    else:
        LD, pivots = _linalg_ldl_factor_v6(A)

    num_batches = A.numel() // (n * n)
    info = torch.empty(A.shape[:-2], dtype=torch.int32, device=A.device)
    _ldl_init_info_kernel[(num_batches,)](info, num_batches)
    return LD, pivots, info
