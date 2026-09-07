import pytest
import torch

from . import base


# Multi-backend: the previous `vendor_name not in ("nvidia", "thead")` skipif was stale.
# Verified on kunlunxin (XPU, 2026-08-31) with the body below unchanged: all 13 shape
# cells run SUCCESS, so no vendor gate is needed here.
@pytest.mark.special_shifted_chebyshev_polynomial_w
def test_special_shifted_chebyshev_polynomial_w():
    bench = base.BinaryPointwiseBenchmark(
        op_name="special_shifted_chebyshev_polynomial_w",
        torch_op=torch.special.shifted_chebyshev_polynomial_w,
        # PyTorch reference only supports float32 for this operator
        dtypes=[torch.float32],
    )
    bench.run()
