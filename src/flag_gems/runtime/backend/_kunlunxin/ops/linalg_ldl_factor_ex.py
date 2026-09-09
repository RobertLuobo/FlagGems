"""Kunlunxin ldl_factor_ex (aten::linalg_ldl_factor_ex) vendor override.

The extended op shares the module that provides the base `ldl_factor` (see
`linalg_ldl_factor.py`, which returns (LD, pivots, info) with an fp32 work
space and full shape/dtype checks).  This module only keeps the
`ldl_factor_ex` name bound for `ops/__init__.py`.
"""

from .linalg_ldl_factor import ldl_factor_ex  # noqa: F401
