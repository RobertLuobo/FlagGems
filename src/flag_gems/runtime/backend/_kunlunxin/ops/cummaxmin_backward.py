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

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)

ROW_BLOCK = 128


@libentry()
@triton.jit(do_not_specialize=["reduction_size", "n_rows"])
def rowown_backward_kernel(
    grad_ptr,
    indices_ptr,
    grad_input_ptr,
    reduction_size,
    inner_size,
    n_rows,
    ROW_BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    r = pid * ROW_BLOCK + tl.arange(0, ROW_BLOCK)
    rmask = r < n_rows
    outer = r // inner_size
    inner = r % inner_size
    base = outer * reduction_size * inner_size + inner
    for i in tl.range(reduction_size, num_stages=1):
        off = base + i * inner_size
        g = tl.load(grad_ptr + off, mask=rmask, other=0.0).to(tl.float32)
        t = tl.load(indices_ptr + off, mask=rmask, other=0)
        tgt = base + t * inner_size
        cur = tl.load(grad_input_ptr + tgt, mask=rmask, other=0.0)
        tl.store(grad_input_ptr + tgt, cur + g, mask=rmask)


def cummaxmin_backward(grad_output, input, indices, dim):
    logger.debug("GEMS_KUNLUNXIN CUMMAXMIN_BACKWARD")
    ndim = grad_output.ndim
    if dim < 0:
        dim = dim + ndim

    shape = list(grad_output.shape)
    reduction_size = shape[dim]

    grad_c = grad_output.contiguous()
    indices_c = indices.contiguous()

    inner_size = 1
    for i in range(dim + 1, ndim):
        inner_size *= shape[i]
    outer_size = 1
    for i in range(dim):
        outer_size *= shape[i]

    grad_input_f32 = torch.zeros(shape, dtype=torch.float32, device=grad_output.device)

    n_rows = outer_size * inner_size
    grid = (triton.cdiv(n_rows, ROW_BLOCK),)
    with torch_device_fn.device(grad_output.device):
        rowown_backward_kernel[grid](
            grad_c,
            indices_c,
            grad_input_f32,
            reduction_size,
            inner_size,
            n_rows,
            ROW_BLOCK,
        )

    return grad_input_f32.to(grad_output.dtype)
