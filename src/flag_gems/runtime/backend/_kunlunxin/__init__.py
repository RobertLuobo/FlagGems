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

from backend_utils import VendorDescriptor  # noqa: E402

vendor_info = VendorDescriptor(
    vendor_name="kunlunxin",
    device_name="cuda",
    device_query_cmd="xpu-smi",
    triton_extra_name="xpu",
    fp64_enabled=False,
)

CUSTOMIZED_UNUSED_OPS = (
    "cumsum",
    "randperm",
    "topk",
    "unique",
    # flag_gems' generic Python "slice.Tensor" impl (src/flag_gems/ops/slice.py)
    # is incompatible with this torch_xmlir-XPU build: the ATen dispatcher
    # invokes the registered Python kernel with 4 positional arguments
    # (self, dim, start, stop), omitting the defaulted `step`, so ANY slice
    # inside a use_gems() context raises
    #     TypeError: slice() missing 1 required positional argument: 'step'
    # (see e.g. tests/test_fill.py::test_fill_scalar_sliced_view /
    # test_fill_sliced_view_tensor, and the _kunlunxin/ops/*.py workarounds).
    # Additionally that generic impl materialises a COPY (torch.empty + copy_)
    # instead of a view, so it can never satisfy the aliasing semantics of
    # Tensor.__setitem__'s slice fast path (view = x.slice(...); view.fill_(v))
    # even when called with an explicit step. Use the native torch_xmlir
    # slice.Tensor (which is a correct zero-copy view) for this vendor.
    "slice",
)


__all__ = ["*"]
