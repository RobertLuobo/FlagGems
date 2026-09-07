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

"""
mHC Backward entry guard (kunlunxin / XPU specialized).

Why this file exists (XPU, measured 2026-09-04):
- The general implementation in ``flag_gems/fused/mhc/mhc_bwd.py`` (which
  already passes the full mhc_bwd test / benchmark matrix on XPU with a
  dtype-equal Gems Speedup of ~29.9x) launches ``_mhc_bwd_kernel_n4`` with
  ``grid = (cdiv(seqlen, 64),)`` and relies on masked tail loads
  (``other=0.0``) for ``seqlen % 64 != 0``.
- On XPU the masked tail of ``_mhc_bwd_kernel_n4`` reads out of the tensor
  allocation (mask/other ignored at the load level, same family of defect as
  documented in the harness for reduction tails), which raises a device
  kernel exception (``Xid ... KL_XID_KERNEL_EXCEPTION``) and wedges the
  device for subsequent launches. ``seqlen == 0`` additionally produces a
  zero-sized grid launch, which also raises a kernel exception.
- The official test/benchmark matrix (seqlen in {256, 1024, 4096, 65536})
  is always a multiple of 64, so the exception is not exercised there; this
  guard is a robustness fix for out-of-matrix shapes.

Strategy (no Triton changes, behavior for the supported matrix is identical
to the general implementation, which is delegated to 1:1):
- ``numel() == 0`` -> return the empty result directly (no kernel launch).
- ``seqlen % 64 != 0`` -> pad the batch dim up to the next multiple of 64,
  run the general kernel (whose per-row CG math makes padded rows
  independent and harmless), then slice the valid rows back out.

NOTE (2026-09-04): this guard was written after the device-side kernel
exception it fixes; it is UNVERIFIED until the device is NOC-checked/reset
and the ``tests/`` and ``benchmark/`` paths are re-run.
"""

import torch

from flag_gems.fused.mhc.mhc_bwd import mhc_bwd as _general_mhc_bwd

_BLOCK_S = 64  # must match BLOCK_S in the general _mhc_bwd_kernel_n4


def mhc_bwd(
    out: torch.Tensor,
    dout: torch.Tensor,
    cg_iters: int = None,
) -> torch.Tensor:
    """Sinkhorn backward (kunlunxin / XPU) with XPU-safe input guards.

    Same interface and semantics as `flag_gems.fused.mhc.mhc_bwd.mhc_bwd`;
    delegates to the general implementation for shapes that are safe on XPU.
    """
    if out.numel() == 0:
        return torch.empty_like(out.float())

    seqlen = out.shape[0]
    if seqlen % _BLOCK_S != 0:
        pad = (-seqlen) % _BLOCK_S
        out_p = torch.nn.functional.pad(out, (0, 0, 0, 0, 0, pad))
        dout_p = torch.nn.functional.pad(dout, (0, 0, 0, 0, 0, pad))
        res = _general_mhc_bwd(out_p, dout_p, cg_iters=cg_iters)
        return res[:seqlen]

    return _general_mhc_bwd(out, dout, cg_iters=cg_iters)


def _install():
    """Wire the XPU entry into the direct-import entrypoint.

    The mhc fused family is called via direct module import
    (`from flag_gems.fused.mhc.mhc_bwd import mhc_bwd`) in both
    tests/test_mhc_ops.py and benchmark/test_mhc.py, so the normal
    SpecOpRegistrar namespace swap can not reach it. Replace the attribute on
    the already-imported module (loaded during `import flag_gems`).
    """
    import sys

    mod = sys.modules.get("flag_gems.fused.mhc.mhc_bwd")
    if mod is not None:
        cur = getattr(mod, "mhc_bwd", None)
        if cur is _general_mhc_bwd:
            mod.mhc_bwd = mhc_bwd


_install()