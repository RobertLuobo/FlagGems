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

# NOTE(2026-10-14): kunlunxin fused-operator specializations intentionally
# disabled. Attribute access is forwarded to the generic implementations in
# src/flag_gems/fused/. The original registry is preserved in git history.
#
# A module-level ``__getattr__`` (PEP 562) is used instead of a star-import so
# that ``inspect.getmembers(module, inspect.isfunction)`` -- which the backend
# uses to detect vendor-customized ops -- does not mistake the generic fused
# kernels for vendor customizations. The forwarder is a callable instance
# rather than a plain function so it is not itself reported.

from flag_gems import fused as _generic_fused


class _GenericFusedForwarder:
    def __call__(self, name):
        return getattr(_generic_fused, name)


__getattr__ = _GenericFusedForwarder()
