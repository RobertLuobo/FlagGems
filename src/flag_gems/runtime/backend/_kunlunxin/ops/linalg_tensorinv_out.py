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
import importlib
import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as tle

logger = logging.getLogger(__name__)

_generic = importlib.import_module("flag_gems.ops.linalg_tensorinv")
check_inv_input = _generic.check_inv_input


@libentry()
@triton.jit
def _tinv_pivot_kernel(A, piv, N, k, BLOCK: tl.constexpr):
    rows = tl.arange(0, BLOCK)
    col = tl.load(A + rows * N + k, mask=rows < N, other=0.0).to(tl.float32)
    absc = tl.where((rows >= k) & (rows < N), tl.abs(col), -1.0)
    mx = tl.max(absc, axis=0)
    pr = tl.min(tl.where(absc == mx, rows, N), axis=0)
    tl.store(piv, pr)


@libentry()
@triton.jit
def _tinv_step_kernel(A_in, B_in, A_out, B_out, piv, N, k, BLOCK: tl.constexpr):
    r = tle.program_id(0)
    p = tl.load(piv)
    cols = tl.arange(0, BLOCK)
    cm = cols < N
    src = tl.where(r == k, p, tl.where(r == p, k, r))
    a_src = tl.load(A_in + src * N + cols, mask=cm, other=0.0).to(tl.float32)
    b_src = tl.load(B_in + src * N + cols, mask=cm, other=0.0).to(tl.float32)
    pivotval = tl.load(A_in + p * N + k).to(tl.float32)
    prow_a = tl.load(A_in + p * N + cols, mask=cm, other=0.0).to(tl.float32) / pivotval
    prow_b = tl.load(B_in + p * N + cols, mask=cm, other=0.0).to(tl.float32) / pivotval
    factor = tl.load(A_in + src * N + k).to(tl.float32)
    is_k = r == k
    a_out = tl.where(is_k, a_src / pivotval, a_src - factor * prow_a)
    b_out = tl.where(is_k, b_src / pivotval, b_src - factor * prow_b)
    tl.store(A_out + r * N + cols, a_out, mask=cm)
    tl.store(B_out + r * N + cols, b_out, mask=cm)


def linalg_tensorinv(A, ind=2, *, out=None):
    logger.debug("GEMS_KUNLUNXIN LINALG_TENSORINV")
    check_inv_input(A, ind)

    matrix_size = 1
    for i in range(ind):
        matrix_size *= A.shape[i]
    output_shape = A.shape[ind:] + A.shape[:ind]
    n = matrix_size

    A0 = (
        A.contiguous()
        .to(torch.float32)
        .reshape(n, n)
        .clone(memory_format=torch.contiguous_format)
    )
    A1 = torch.empty_like(A0)
    I0 = torch.eye(n, dtype=torch.float32, device=A.device).clone(
        memory_format=torch.contiguous_format
    )
    I1 = torch.empty_like(I0)
    As = [A0, A1]
    Is = [I0, I1]
    piv = torch.zeros(1, dtype=torch.int32, device=A.device)

    block = max(2, triton.next_power_of_2(n))
    with torch_device_fn.device(A.device):
        for k in range(n):
            cur = k % 2
            nxt = (k + 1) % 2
            _tinv_pivot_kernel[(1,)](As[cur], piv, n, k, BLOCK=block)
            _tinv_step_kernel[(n,)](
                As[cur], Is[cur], As[nxt], Is[nxt], piv, n, k, BLOCK=block
            )

    inverse = Is[n % 2]
    result = inverse.reshape(output_shape).to(A.dtype)
    if out is not None:
        out.copy_(result)
        return out
    return result


def linalg_tensorinv_out(A, ind=2, *, out=None):
    logger.debug("GEMS_KUNLUNXIN LINALG_TENSORINV_OUT")
    return linalg_tensorinv(A, ind=ind, out=out)
