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

# NOTE(2026-10-14): kunlunxin operator specializations intentionally disabled.
# Every op path -- both the registered flag_gems.<op> and direct vendor-module
# access (e.g. ``from flag_gems.runtime.backend._kunlunxin import ops``) --
# resolves to the generic kernels in src/flag_gems/ops/. The vendor kernels on
# disk are no longer imported. The original registry is preserved in git
# history.
#
# Forwarding uses a module-level ``__getattr__`` (PEP 562) instead of a
# star-import: ``inspect.getmembers(module, inspect.isfunction)``, which the
# backend uses to detect vendor-customized ops, only sees names bound in the
# module namespace. With a star-import the generic kernels would be misreported
# as vendor customizations; with forwarding they are not. The forwarder is a
# callable instance rather than a plain function so it is not itself reported.

from flag_gems import ops as _generic_ops
from flag_gems.ops import __all__  # noqa: F401


class _GenericOpsForwarder:
    def __call__(self, name):
        return getattr(_generic_ops, name)


__getattr__ = _GenericOpsForwarder()
