# Copyright 2026 FlagOS Contributors
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

import logging

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)


@triton.jit
def _osj_pipeline(
    A_ptr, B_ptr, U_ptr, m, n, nw, total, MP: tl.constexpr, NW: tl.constexpr
):
    """Fill ``B``, run the one-sided Jacobi sweeps, and write ``U = B/S``."""
    rows = tl.arange(0, MP)
    ring = nw - 1
    half = nw // 2
    msk = rows < MP  # store mask: on this backend unmasked vector stores write
    # a fixed ~2KB window (the phantom lanes land in the memory block right
    # above the tensor); the mask limits the store to the real MP rows.

    # 1. fill: B[r, c] = A[r, c] if r < m and c < n else 0 (scalar stores)
    for r in range(0, MP):
        for c in range(0, NW):
            val = 0.0
            if (r < m) and (c < n):
                val = tl.load(A_ptr + r * n + c)
            tl.store(B_ptr + r * NW + c, val)

    # 2. cyclic one-sided Jacobi sweeps (flattened (sweep, s, j) schedule):
    #    rotate column pair (p, q) so that their inner product becomes zero.
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

    # 3. U = B * diag(1/S) with S = ||B[:, j]|| (vector stores only)
    for j in range(0, nw):
        v = tl.load(B_ptr + j + rows * NW)
        sv = tl.sqrt(tl.sum(v * v))
        inv = tl.where(sv > 1.0e-20, 1.0 / sv, 0.0)
        tl.store(U_ptr + j + rows * NW, v * inv, mask=msk)


def _osj_svd_impl(A, sweeps=12, full_matrices=False):
    """One-sided Jacobi SVD; returns ``(U, S, Vh)`` per torch convention."""
    dev = A.device
    if A.dim() == 2:
        A = A.unsqueeze(0)
    batch, m, n = A.shape
    nw = n if n % 2 == 0 else n + 1
    NW = nw if (nw & (nw - 1)) == 0 else triton.next_power_of_2(nw)
    MP = triton.next_power_of_2(m)

    # workspace (zero-padded) and U, both produced by the single pipeline
    # kernel (one launch per batch element, grid (1,)).
    B = torch.empty((batch, MP, NW), device=dev, dtype=A.dtype)
    U = torch.empty((batch, MP, NW), device=dev, dtype=A.dtype)
    total = sweeps * (nw - 1) * (nw // 2)
    for b in range(batch):
        _osj_pipeline[(1,)](
            A[b],
            B[b],
            U[b],
            m,
            n,
            nw,
            total,
            MP=MP,
            NW=NW,
            num_warps=1,
            num_stages=1,
        )

    # S = column norms of B, computed on host (scalar-store workaround);
    # the D2H copy is also the completion barrier for the pipeline kernel.
    Bc = B.cpu().double()
    S = Bc.norm(dim=1).to(device=dev, dtype=A.dtype)  # (batch, NW)

    k = min(m, n)
    S_sorted, idx = torch.sort(S, dim=-1, descending=True)
    S_sorted = S_sorted[:, :k]
    idxg = idx.unsqueeze(1).expand(-1, MP, -1)
    U = torch.gather(U, 2, idxg)[:, :m, :k].contiguous()

    # Vh = S^{-1} U^H A   (exact identity A = U diag(S) Vh when U orthonormal)
    UtA = torch.matmul(U.transpose(-2, -1), A)
    Vh = UtA * (1.0 / S_sorted).unsqueeze(-1)

    if full_matrices:
        if m > k:
            pad = torch.zeros((batch, m, m - k), device=dev, dtype=A.dtype)
            U = torch.cat([U, pad], dim=2)
        if n > k:
            pad = torch.zeros((batch, n - k, n), device=dev, dtype=A.dtype)
            Vh = torch.cat([Vh, pad], dim=1)
    Vh = Vh.contiguous()

    if batch == 1:
        return U[0], S_sorted[0], Vh[0]
    return U, S_sorted, Vh


def linalg_svd(A, full_matrices=True, *, driver=None):
    """Triton (XPU) implementation of ``torch.linalg.svd``.

    Returns ``(U, S, Vh)`` with ``A = U @ diag(S) @ Vh``.  Only ``float32``
    is supported (matches the generic ``linalg_svd`` contract).
    """
    logger.debug("GEMS LINALG_SVD (kunlunxin)")
    if A.dtype != torch.float32:
        raise TypeError(f"linalg_svd only supports float32 input, got {A.dtype}")
    return _osj_svd_impl(A, full_matrices=full_matrices)
