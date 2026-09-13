import pytest
import torch

from . import base


# ormqr benchmark
# ormqr applies Householder reflectors which have inherent sequential dependency.
# The fused Triton kernel excels on small-to-medium matrices where the entire
# active region fits in GPU SRAM (<=128 in both dimensions).
class OrmqrBenchmark(base.GenericBenchmark2DOnly):
    # Override default shapes to include representative sizes for Householder application:
    # small matrices (fused kernel path) and medium/large matrices (tiled path)
    # NOTE (Kunlunxin, 2026-09-11): (4096, 4096) and (1024, 65536) are excluded on
    # this backend - their reflector directions blow up to O(k * rows/64) sequential
    # kernel launches (k > 2048 per-reflector path, e.g. 2k=8192 launches/call at
    # k=4096), i.e. tens of minutes per single do_bench call, so neither baseline
    # nor the fixed implementation can complete a full run. Both the baseline and
    # the post-fix measurements use this reduced set (same convention as the
    # kunlunxin core_shapes comment "Big Shape Will Cause Timeout").
    DEFAULT_SHAPES = [
        (32, 32),
        (48, 48),
        (64, 64),
        (96, 96),
        (128, 128),
        (256, 256),
        (1024, 1024),
    ]

    def set_shapes(self, shape_file_path=None):
        self.shapes = self.DEFAULT_SHAPES

    def get_tflops(self, op, *args, **kwargs):
        # ormqr: multiply Q (m x m or n x n) with C (m x n)
        # Flops: 2 * m * n * min(m, n) for the matrix multiplication
        m, n = args[2].shape
        k = args[0].shape[-1]  # k = min(m, n) for ormqr
        return 2 * m * n * k


@pytest.mark.ormqr
def test_ormqr():
    def ormqr_input_fn(shape, dtype, device):
        m, n = shape
        k = min(m, n)
        # Generate valid Householder reflectors via QR decomposition
        a = torch.randn(m, k, dtype=dtype, device=device)
        input_tensor, tau = torch.geqrf(a)
        other = torch.randn(m, n, dtype=dtype, device=device)
        yield input_tensor, tau, other

    bench = OrmqrBenchmark(
        input_fn=ormqr_input_fn,
        op_name="ormqr",
        torch_op=torch.ormqr,
        # ormqr only supports float32 and float64 (LAPACK limitation, no half/bfloat16)
        dtypes=[torch.float32, torch.float64],
    )
    bench.run()
