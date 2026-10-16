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
    fp8_enabled=False,
)

# NOTE(2026-10-14): kunlunxin specializations disabled. The generic
# implementations of these ops must now be registered, so the exclusion list
# is empty. The original list is preserved in git history.
CUSTOMIZED_UNUSED_OPS = ()


# NOTE(2026-09-23): atanh_ is already registered by the generic _FULL_CONFIG
# (src/flag_gems/__init__.py, ("atanh_", atanh_), added by #6009), so the
# import-time monkey patch of GeneralOpRegistrar that used to live here was
# redundant. It was also harmful: vendor auto-detection imports every backend
# module (backend.get_vendor_infos -> importlib.import_module("_<vendor>")), so
# on non-kunlunxin platforms the patched __init__ fired during their use_gems()
# and pulled in _kunlunxin.ops, whose acos.py imports
# triton.language.extra.xpu.libdevice -> ModuleNotFoundError on vendors without
# the xpu triton extra. Removed; the kunlunxin atanh_ override still applies
# through the vendor ops mechanism (SpecOpRegistrar).

__all__ = ["*"]
