
"""Kunlunxin backend override for ``linalg_svdvals``.

The generic ``flag_gems.ops.linalg_svdvals`` routes through the generic CUDA
Triton SVD kernels (``_small_jacobi_svals_kernel`` /
``_blocked_jacobi_svals_kernel`` in ``flag_gems/ops/svd.py``), whose
constexpr-tiled kernels do not finish compiling on the Triton-XPU backend
(``xpu.llvm.translate_to_asm`` still running after 900 s for a (16, 16) tile;
``pytest-timeout`` killed the benchmark baseline).  The overload store is also
CPU/ATen-fallback-free in the generic path only for shapes whose Triton kernels
compile, which does not hold on XPU.

This override reuses the Kunlunxin ``linalg_svd`` one-sided Jacobi pipeline
basis (``_osj_pipeline`` in ``linalg_svd.py``), which is built from *runtime*
loops (nothing constexpr-unrolled beyond ``tl.arange`` tiles) and therefore
compiles fast on XPU, but **keeps only the ``B = U*S`` workspace**: contrary to
``linalg_svd`` (which needs ``U``/``Vh``), ``linalg_svdvals`` only needs the
singular values ``S = ||B[:, j]||``, so the pipeline here stops after the
cyclic rotations (no ``U = B*diag(1/S)`` normalization in-kernel, and no
``torch.gather`` / ``U^H A`` matmul / ``Vh`` scaling on the host side).  The
rotations are *bit-identical* to ``_osj_pipeline`` (same schedule, same
arithmetic), so the singular values match the ``linalg_svd`` S output exactly.

dtype: float32 only (matching the generic linalg_svdvals contract).
"""

import logging

import numpy as np
import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)


@triton.jit
def _osj_svals_pipeline(A_ptr, B_ptr, m, n, nw, total, MP: tl.constexpr, NW: tl.constexpr):
    """Fill ``B`` and run the one-sided Jacobi sweeps (singular values only).

    Identical to the rotation part of ``linalg_svd._osj_pipeline`` (same cyclic
    schedule, same ``tl.sum``-based ``alpha/beta/gamma`` reductions, same
    ``mask=rows < MP`` store guard); the ``U = B * diag(1/S)`` normalization
    that ``linalg_svd`` needs is omitted because ``svdvals`` only requires
    ``S = ||B[:, j]||`` (computed on the host from ``B``).
    """
    rows = tl.arange(0, MP)
    ring = nw - 1
    half = nw // 2
    msk = rows < MP

    for r in range(0, MP):
        for c in range(0, NW):
            val = 0.0
            if (r < m) and (c < n):
                val = tl.load(A_ptr + r * n + c)
            tl.store(B_ptr + r * NW + c, val)

    for t in range(0, total):
        s = (t // half) % ring
        j = t % half
        p = tl.where(j == 0, 0, (j + ring - s - 1) % ring + 1)
        q = (nw - 1 - j + ring - s - 1) % ring + 1
        ap = tl.load(B_ptr + p + rows * NW)
        aq = tl.load(B_ptr + q + rows * NW)
        alpha = tl.sum(ap * ap)
        beta = tl.sum(aq * aq)
        gamma = tl.sum(ap * aq)
        eps = 1.0e-20
        threshold = 1.0e-7 * tl.sqrt(alpha * beta + eps)
        active = tl.abs(gamma) > threshold
        safe_gamma = tl.where(active, gamma, 1.0)
        tau = (beta - alpha) / (2.0 * safe_gamma)
        sign_tau = tl.where(tau >= 0.0, 1.0, -1.0)
        t_rot = sign_tau / (tl.abs(tau) + tl.sqrt(1.0 + tau * tau))
        c = tl.rsqrt(1.0 + t_rot * t_rot)
        s_rot = t_rot * c
        c = tl.where(active, c, 1.0)
        s_rot = tl.where(active, s_rot, 0.0)
        tl.store(B_ptr + p + rows * NW, c * ap - s_rot * aq, mask=msk)
        tl.store(B_ptr + q + rows * NW, s_rot * ap + c * aq, mask=msk)



def _osj_svals_impl(A, sweeps=12):
    """One-sided Jacobi singular values only; returns ``S`` (descending)."""
    dev = A.device
    if A.dim() == 2:
        A = A.unsqueeze(0)
    batch, m, n = A.shape
    nw = n if n % 2 == 0 else n + 1
    NW = nw if (nw & (nw - 1)) == 0 else triton.next_power_of_2(nw)
    MP = triton.next_power_of_2(m)

    B = torch.empty((batch, MP, NW), device=dev, dtype=A.dtype)
    total = sweeps * (nw - 1) * (nw // 2)
    for b in range(batch):
        _osj_svals_pipeline[(1,)](
            A[b], B[b], m, n, nw, total, MP=MP, NW=NW,
            num_warps=1, num_stages=1,
        )

    Bc = B.cpu().double()
    S = Bc.norm(dim=1).numpy()

    k = min(m, n)
    S_sorted = np.sort(S, axis=-1)[:, ::-1][:, :k]
    S_sorted = torch.from_numpy(np.ascontiguousarray(S_sorted)).to(
        device=dev, dtype=A.dtype
    )

    if batch == 1:
        return S_sorted[0]
    return S_sorted


def linalg_svdvals(A: torch.Tensor, driver: str = None) -> torch.Tensor:
    """Computes the singular values of a matrix (Kunlunxin XPU).

    Args:
        A: Input tensor of shape (*, m, n) where * is zero or more batch dimensions.
        driver: Accepted for API compatibility; the one-sided Jacobi pipeline
            does not use a solver driver selection.

    Returns:
        Singular values in descending order, shape (*, min(m, n)).
    """
    logger.debug("GEMS LINALG_SVDVALS (kunlunxin)")
    if A.dtype != torch.float32:
        raise TypeError(f"linalg_svdvals only supports float32 input, got {A.dtype}")
    if not A.is_contiguous():
        A = A.contiguous()

    if A.dim() not in (2, 3):
        orig_shape = A.shape
        m, n = orig_shape[-2:]
        k = min(m, n)
        A = A.reshape(-1, m, n)
        S = _osj_svals_impl(A)
        return S.reshape(*orig_shape[:-2], k)

    return _osj_svals_impl(A)