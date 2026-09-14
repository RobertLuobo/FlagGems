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
    "slice",
    # The 097c718a (batch 20260914) vendor conv_transpose1d.py was only an
    # fp16/bf16-upcast wrapper around
    #   aten::conv_transpose1d.default.redispatch(CompositeImplicitAutograd, ...)
    # and was never imported by _kunlunxin/ops/__init__.py (dead code). The
    # generic flag_gems.ops.conv_transpose1d (triton autotune + tl.dot) has no
    # kunlunxin tune_configs.yaml entry and the conv family is unsupported on
    # this stack (cf. op_black_list.yaml: conv1d "All dtypes failed"), so
    # exclude it from use_gems() registration and keep the native ATen path.
    "conv_transpose1d",
)


__all__ = ["*"]
