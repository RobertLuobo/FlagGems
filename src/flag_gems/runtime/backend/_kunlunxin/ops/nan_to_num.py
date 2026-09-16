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

import functools
import logging
import os

import torch
import triton
import triton.language as tl
from _kunlunxin.utils.codegen_config_utils import CodeGenConfig
from flag_gems.runtime import torch_device_fn

from ..utils.pointwise_dynamic import pointwise_dynamic

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# tle.raw fast path for the large-shape in-place nan_to_num_ (P800 xpu3,
# cluster C payload in nan_to_num_raw.xpu).
#
# Why a raw payload: the compile-time select of the pointwise kernel keeps the
# stdlib `arith.select` (which the XPU backend lowers to a SCALAR-predicated
# select, ~0.38x on the large benchmark shapes) and the elementwise 3-step
# select chain never fuses into a single vector pass; the hand-written payload
# drives per-core GM2LM/LM2GM DMA and evaluates the nan/+-inf conditions with
# the hardware vvneq_*/vveq_* vector compares + a single masked-HOLD bitwise
# and per select (m ? y : x == vvand_*_mh(y, ONES, x, m), bitwise-exact for
# every value including -0.0 and signaling NaNs), reading and writing the
# input once with the same footprint as ATen. See ne_raw.xpu / not_equal.py
# for the original recipe this mirrors.
#
# In-place only: the raw payload writes into A (out == in). The out-of-place
# `nan_to_num` would need a full-size clone first (3 memory passes total)
# against the pointwise kernel's single out-of-place pass, so it keeps the
# pointwise path.
try:
    import triton.experimental.tle as tle

    _TLE_OK = True
except ImportError:
    tle = None
    _TLE_OK = False

_HERE = os.path.dirname(os.path.abspath(__file__))
_NCLUSTER = 12  # P800 (xpu3): one Triton program == one cluster of 64 cores
# Payload scalars are i32 (do_not_specialize); guard the byte range.
_RAW_MAX_ELEMS = 2**31 - 1
# Must match CHUNK_BYTES in nan_to_num_raw.xpu (the chunk-grid partition
# contract). 1792B is the largest 64B-aligned chunk that keeps 2 input + 2
# output buffers under the compiler's 8000B local-memory budget.
_RAW_CHUNK_BYTES = 1792

_RAW_TYPE_CODE = {
    torch.float32: 0,
    torch.float16: 1,
    torch.bfloat16: 2,
}

if _TLE_OK:

    @tle.raw.dialect("xpu3", file=os.path.join(_HERE, "nan_to_num_raw.xpu"))
    def nan_to_num_raw_(in_, out, numel, esz, type_code, nan_bits, pinf_bits,
                        ninf_bits, chunk_start, chunk_count):
        ...

    @triton.jit(do_not_specialize=[
        "numel", "esz", "type_code", "nan_bits", "pinf_bits", "ninf_bits",
        "chunk_count",
    ])
    def nan_to_num_raw_kernel(In, Out, numel, esz, type_code, nan_bits,
                              pinf_bits, ninf_bits, chunk_count):
        pid = tl.program_id(0)
        tle.raw.call(nan_to_num_raw_, (In, Out, numel, esz, type_code,
                                       nan_bits, pinf_bits, ninf_bits,
                                       pid * chunk_count, chunk_count))


@functools.lru_cache(maxsize=1024)
def _replacement_bits(v, dtype):
    """A replacement scalar converted to `dtype`, as a sign-extended i32 bit
    pattern (fp16/bf16 in the low 16 bits) -- exactly torch's promotion of a
    python float to the tensor's dtype."""
    if dtype == torch.float32:
        return int(torch.tensor(v, dtype=torch.float32).view(torch.int32).item())
    return int(torch.tensor(v, dtype=dtype).view(torch.int16).item())


def _view_u8(t):
    """Byte view of a tensor."""
    return t.view(torch.uint8)


def _raw_nan_to_num_(A, nan, posinf, neginf):
    """In-place nan_to_num_ via the raw payload, or None when it does not
    apply (non-contiguous / unsupported dtype / empty / too large)."""
    if not _TLE_OK or not A.is_contiguous():
        return None
    type_code = _RAW_TYPE_CODE.get(A.dtype)
    if type_code is None:
        return None
    M = A.numel()
    if M == 0 or M > _RAW_MAX_ELEMS:
        return None
    esz = A.element_size()
    nb, pb, mb = (_replacement_bits(nan, A.dtype),
                  _replacement_bits(posinf, A.dtype),
                  _replacement_bits(neginf, A.dtype))
    # partition by payload chunks (CHUNK_BYTES/esz elements each): every
    # program and core gets whole chunks so all GM2LM/LM2GM transfers are
    # CHUNK_BYTES-aligned in global memory.
    chunk_elems = _RAW_CHUNK_BYTES // esz
    total_chunks = (M + chunk_elems - 1) // chunk_elems
    per = (total_chunks + _NCLUSTER - 1) // _NCLUSTER
    with torch_device_fn.device(A.device):
        nan_to_num_raw_kernel[(_NCLUSTER,)](
            _view_u8(A), _view_u8(A), M, esz, type_code, nb, pb, mb, per)
    return A

# nan_to_num is an elementwise select (isnan / ±inf checks + tl.where): a pure
# memory-bound select/copy. Two independent findings drive this implementation:
#
# 1. Config: the old kunlunxin override used a BARE pointwise_dynamic with NO
#    CodeGenConfig, so on XPU it fell to the default path (buffer_size_limit
#    2048, no kunlunAutoGrid, no unroll) -> BLOCK=512 1d tile, underutilized
#    bandwidth. This reuses the proven memory-bound select/copy recipe shared
#    by neg / view_copy / masked_fill (autoGrid + unroll8 + buffer 4096).
#    Config sweep confirmed unroll16/buffer8192 and isCloseVectorization=True
#    give no further gain on this kernel.
#
# 2. NaN detection: `_isnan(x.to(tl.float32))` (extern libdevice call) is the
#    dominant cost on XPU — extern_elementwise lowers to a scalar/throughput-
#    limited path (~10x slower than a pure select, ~63 GB/s at [4096,4096]
#    fp16). Replaced it with an integer bit trick on the fp32 bits:
#      NaN    := (bits & 0x7FFFFFFF) > 0x7F800000   (exponent all-ones, mantissa != 0)
#      +inf   := bits == 0x7F800000
#      -inf   := bits == 0xFF800000
#    This is exact IEEE-754 semantics (bit identities for NaN/inf are unique),
#    uses only cheap integer ALU ops and removes the extern call. fp32 dtype
#    path needs no conversion at all; fp16/bf16 pay the same single fp32
#    upcast as before but skip the extern. Bit-identical output.
#    Self-compare variants (x != x / x > x) crash the XPU llir pass
#    (PassManager::run failed) and int16 bitcast is unsupported — both dead
#    ends; the fp32 bitmask is the fastest verified body.
config_ = CodeGenConfig(
    512,
    (65536, 65536, 65536),
    32,
    True,
    prefer_1d_tile=True,
    buffer_size_limit=4096,
    isCloseVectorization=True,
    kunlunAutoGrid=True,
    unroll_num=8,
)


@pointwise_dynamic(
    is_tensor=[True, False, False, False],
    promotion_methods=[(0, "DEFAULT")],
    config=config_,
)
@triton.jit
def nan_to_num_func(x, nan, posinf, neginf):
    # IEEE-754 bit patterns (fp32): |inf| = 0x7F800000, any NaN has exponent
    # all-ones and nonzero mantissa, so (bits & 0x7FFFFFFF) > 0x7F800000 is
    # exactly isnan; and ==0x7F800000 / ==0xFF800000 are +/-inf.
    x_bits = x.to(tl.float32).to(tl.int32, bitcast=True)
    x_nan = (x_bits & 0x7FFFFFFF) > 0x7F800000
    x_posinf = x_bits == 0x7F800000
    x_neginf = x_bits == 0xFF800000
    x = tl.where(x_nan, nan, x)
    x = tl.where(x_posinf, posinf, x)
    x = tl.where(x_neginf, neginf, x)
    return x


# At this element count the payload launch overhead (12 programs, chunk-grid
# arithmetic host side) is amortized and it beats the pointwise kernel (whose
# compile-time select is scalarized on XPU); small shapes keep the pointwise
# path, which is already bit-exact.
_RAW_MIN_ELEMS = 65536


# nan_to_num(Tensor self, float? nan=None, float? posinf=None, float? neginf=None) -> Tensor
def nan_to_num(A, nan=None, posinf=None, neginf=None):
    logger.debug("GEMS_KUNLUNXIN NAN_TO_NUM")
    if posinf is None:
        posinf = torch.finfo(A.dtype).max
    if neginf is None:
        neginf = torch.finfo(A.dtype).min
    if nan is None:
        nan = 0.0
    return nan_to_num_func(A, nan, posinf, neginf)


# nan_to_num_(Tensor self, float? nan=None, float? posinf=None, float? neginf=None) -> Tensor
# In-place variant: same fast kernel, writing back into A (out0=A). Large
# contiguous float inputs take the tle.raw payload (nan_to_num_raw.xpu): the
# payload compares with the hardware vector ne/eq intrinsics and does the
# per-lane select with a single masked-HOLD bitwise and (bit-exact, incl.
# -0.0 and signaling NaNs), in-place with a single input+output pass.
def nan_to_num_(A, nan=None, posinf=None, neginf=None):
    logger.debug("GEMS_KUNLUNXIN NAN_TO_NUM_")
    if posinf is None:
        posinf = torch.finfo(A.dtype).max
    if neginf is None:
        neginf = torch.finfo(A.dtype).min
    if nan is None:
        nan = 0.0
    if A.numel() >= _RAW_MIN_ELEMS:
        raw_out = _raw_nan_to_num_(A, nan, posinf, neginf)
        if raw_out is not None:
            return raw_out
    # Fallback (small shapes, unsupported dtypes, non-contiguous inputs).
    return nan_to_num_func(A, nan, posinf, neginf, out0=A)
