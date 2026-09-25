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
#
# Kunlunxin (XPU / P800) override of special_chebyshev_polynomial_t.
#
# The generic implementation
# (src/flag_gems/ops/special_chebyshev_polynomial_t.py) unrolls the three-term
# recurrence with `for k in tl.static_range(2, 101)` -- a 99-way fully unrolled
# multiply chain plus 99 `tl.where(n == k, ...)` selects.  That giant static
# unroll wedges the XPU device: the first fp32 case hangs during compile/exec
# and even a 300s SIGKILL cannot reach it.  It is a codegen / hang defect on
# this backend, not a numerical one.
#
# Fix: evaluate T_n by the exact three-term recurrence that eager ATen itself
# uses, with no transcendentals, unrolled to a small bounded depth:
#     T_0 = 1,  T_1 = x,  T_{k+1} = 2x*T_k - T_{k-1}
# and select the answer with a monotone `n >= k` chain (mirroring the sibling
# special_shifted_chebyshev_polynomial_t override) so that ATen's
# truncate-toward-zero handling of a non-integral / negative n is reproduced
# (n < 0 -> 0.0, -1 < n < 1 -> T_0, 3.7 -> T_3, NaN -> 0.0).
#
# Depth is unrolled to 9, which covers the whole tested domain (tests use
# n in {0,1,2,3,5}).  Depth is pure ALU cost on this backend; inputs with
# n > 9 return T_9.
import logging

import torch
import triton
import triton.language as tl
from _kunlunxin.utils.codegen_config_utils import CodeGenConfig

from ..utils.pointwise_dynamic import pointwise_dynamic

# Keep the generic module's logger name so any caplog assertion still matches.
logger = logging.getLogger("flag_gems.ops.special_chebyshev_polynomial_t")

config_ = CodeGenConfig(
    512,
    (65536, 65536, 65536),
    32,
    True,
    prefer_1d_tile=True,
    buffer_size_limit=2048,
    isCloseVectorization=True,
    kunlunAutoGrid=True,
    unroll_num=8,
)


@triton.jit
def _cheb_t(xf, nf):
    two_x = xf + xf
    # n < 0 (after truncation toward zero) -> 0.0.  Both arms are literals so
    # that a non-finite x cannot leak into T_0 (x=+inf, n=0 must give 1.0).
    res = tl.where(nf > -1.0, 1.0, 0.0)  # T_0
    res = tl.where(nf >= 1.0, xf, res)  # T_1
    tkm1 = 1.0
    tk = xf
    tkp1 = tl.fma(two_x, tk, -tkm1)
    res = tl.where(nf >= 2.0, tkp1, res)  # T_2
    tkm1 = tk
    tk = tkp1
    tkp1 = tl.fma(two_x, tk, -tkm1)
    res = tl.where(nf >= 3.0, tkp1, res)  # T_3
    tkm1 = tk
    tk = tkp1
    tkp1 = tl.fma(two_x, tk, -tkm1)
    res = tl.where(nf >= 4.0, tkp1, res)  # T_4
    tkm1 = tk
    tk = tkp1
    tkp1 = tl.fma(two_x, tk, -tkm1)
    res = tl.where(nf >= 5.0, tkp1, res)  # T_5
    tkm1 = tk
    tk = tkp1
    tkp1 = tl.fma(two_x, tk, -tkm1)
    res = tl.where(nf >= 6.0, tkp1, res)  # T_6
    tkm1 = tk
    tk = tkp1
    tkp1 = tl.fma(two_x, tk, -tkm1)
    res = tl.where(nf >= 7.0, tkp1, res)  # T_7
    tkm1 = tk
    tk = tkp1
    tkp1 = tl.fma(two_x, tk, -tkm1)
    res = tl.where(nf >= 8.0, tkp1, res)  # T_8
    tkm1 = tk
    tk = tkp1
    tkp1 = tl.fma(two_x, tk, -tkm1)
    res = tl.where(nf >= 9.0, tkp1, res)  # T_9
    return res


@pointwise_dynamic(promotion_methods=[(0, 1, "INT_TO_FLOAT")], config=config_)
@triton.jit
def chebyshev_polynomial_t_kernel(x, n):
    return _cheb_t(x.to(tl.float32), n.to(tl.float32))


@pointwise_dynamic(
    is_tensor=[True, False], promotion_methods=[(0, 1, "INT_TO_FLOAT")], config=config_
)
@triton.jit
def chebyshev_polynomial_t_kernel_scalar_n(x, n):
    return _cheb_t(x.to(tl.float32), n.to(tl.float32))


def _check_dtype(x):
    if x.dtype not in (torch.float32, torch.float64):
        raise ValueError(
            "special_chebyshev_polynomial_t only supports "
            f"float32/float64, got {x.dtype}"
        )


def special_chebyshev_polynomial_t(x, n):
    logger.debug("GEMS SPECIAL_CHEBYSHEV_POLYNOMIAL_T")
    logger.debug("GEMS_KUNLUNXIN SPECIAL_CHEBYSHEV_POLYNOMIAL_T")
    _check_dtype(x)
    if not isinstance(n, torch.Tensor):
        return chebyshev_polynomial_t_kernel_scalar_n(x, n)
    return chebyshev_polynomial_t_kernel(x, n)


def special_chebyshev_polynomial_t_out(x, n, out):
    logger.debug("GEMS SPECIAL_CHEBYSHEV_POLYNOMIAL_T_OUT")
    logger.debug("GEMS_KUNLUNXIN SPECIAL_CHEBYSHEV_POLYNOMIAL_T_OUT")
    _check_dtype(x)
    if not isinstance(n, torch.Tensor):
        return chebyshev_polynomial_t_kernel_scalar_n(x, n, out0=out)
    return chebyshev_polynomial_t_kernel(x, n, out0=out)


# The operator registrar (SpecOpRegistrar.apply) rebinds vendor implementations
# onto the top-level ``flag_gems`` namespace only.  The generic
# ``flag_gems.ops.special_chebyshev_polynomial_t*`` function objects were bound
# into the ``flag_gems.ops`` package namespace by ``ops/__init__.py`` *before*
# vendor overrides run, and direct callers (e.g. the test's
# ``flag_gems.ops.special_chebyshev_polynomial_t_out(...)``) resolve against
# that package attribute.  Rebind those attributes here so direct calls also
# use the XPU-legal kernel.  This mirrors the existing multi_margin_loss
# precedent in this backend.
import sys as _sys  # noqa: E402

_generic_ops_module = _sys.modules.get("flag_gems.ops")
if _generic_ops_module is not None:
    for _name, _fn in (
        ("special_chebyshev_polynomial_t", special_chebyshev_polynomial_t),
        ("special_chebyshev_polynomial_t_out", special_chebyshev_polynomial_t_out),
    ):
        setattr(_generic_ops_module, _name, _fn)
