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
"""Kunlunxin (TritonXPU) specialization of the dense paged ``flash_mla`` decode.

The generic ``flag_gems/fused/flash_mla.py`` fuses inside a single Triton kernel
(a) two ``tl.dot`` calls and (b) ``tl.exp``/``tl.max`` softmax reductions. On this
backend that combination compiles but returns silently wrong values (the generic
kernel emits NaN/inf on XPU3). This mirrors the failure already documented in the
sibling ``flashmla_sparse.py``, so the op is split into dot-only and
reduction-only kernels and the proven ``_qk_logits`` / ``_softmax_stats`` /
``_pv_matmul`` kernels are reused verbatim from that module.
"""

import logging
import math
import sys

import torch
import triton
import triton.language as tl

from flag_gems.runtime import device, torch_device_fn

from .flashmla_sparse import (
    _BDV,
    _BH,
    _BT,
    _pv_matmul,
    _qk_logits,
    _softmax_stats,
)

device = device.name
logger = logging.getLogger(__name__)

_BD = 64  # d tile used by the gather kernels


@triton.jit
def _load_locs(block_table, cache_seqlens, i_sq, offs_t, stride_bt, PAGE_SIZE):
    seqlen = tl.load(cache_seqlens + i_sq)
    in_range = offs_t < seqlen
    t_safe = tl.where(in_range, offs_t, 0)
    page = tl.load(block_table + i_sq * stride_bt + t_safe // PAGE_SIZE)
    kv_loc = page.to(tl.int64) * PAGE_SIZE + (t_safe % PAGE_SIZE)
    kv_loc = tl.where(in_range, kv_loc, 0).to(tl.int64)
    return kv_loc, in_range


@triton.jit
def _valid_mask_paged(
    cache_seqlens,
    valid,  # [SQ, TP] float32
    TP: tl.constexpr,
    BT: tl.constexpr,
):
    i_sq = tl.program_id(0).to(tl.int64)
    i_t = tl.program_id(1)
    offs_t = i_t * BT + tl.arange(0, BT)
    seqlen = tl.load(cache_seqlens + i_sq)
    m = offs_t < seqlen
    tl.store(valid + i_sq * TP + offs_t, m.to(tl.float32))


@triton.jit
def _gather_paged_dt(
    kv,
    block_table,
    cache_seqlens,
    gkv_dt,  # [SQ, DQK, TP], d-major (token contiguous)
    stride_kvn,
    stride_bt,
    PAGE_SIZE,
    TP: tl.constexpr,
    DQK: tl.constexpr,
    BT: tl.constexpr,
    BD: tl.constexpr,
):
    i_sq = tl.program_id(0).to(tl.int64)
    i_t = tl.program_id(1)
    i_d = tl.program_id(2)
    offs_t = i_t * BT + tl.arange(0, BT)
    offs_d = i_d * BD + tl.arange(0, BD)
    kv_loc, _ = _load_locs(
        block_table, cache_seqlens, i_sq, offs_t, stride_bt, PAGE_SIZE
    )
    # [BD, BT] tile: outer stride 1 (d contiguous in kv), inner stride stride_kvn
    v = tl.load(kv + offs_d[:, None] + kv_loc[None, :] * stride_kvn)
    tl.store(gkv_dt + i_sq * DQK * TP + offs_d[:, None] * TP + offs_t[None, :], v)


@triton.jit
def _gather_paged_td(
    kv,
    block_table,
    cache_seqlens,
    gkv_td,  # [SQ, TP, DQK], token-major (d contiguous)
    stride_kvn,
    stride_bt,
    PAGE_SIZE,
    TP: tl.constexpr,
    DQK: tl.constexpr,
    BT: tl.constexpr,
    BD: tl.constexpr,
):
    i_sq = tl.program_id(0).to(tl.int64)
    i_t = tl.program_id(1)
    i_d = tl.program_id(2)
    offs_t = i_t * BT + tl.arange(0, BT)
    offs_d = i_d * BD + tl.arange(0, BD)
    kv_loc, _ = _load_locs(
        block_table, cache_seqlens, i_sq, offs_t, stride_bt, PAGE_SIZE
    )
    # [BT, BD] tile: rows are gathered kv rows, d contiguous -> no tl.trans needed
    v = tl.load(kv + kv_loc[:, None] * stride_kvn + offs_d[None, :])
    tl.store(gkv_td + i_sq * TP * DQK + offs_t[:, None] * DQK + offs_d[None, :], v)


def flash_mla(
    q,
    block_table,
    blocked_k,
    max_seqlen_pad,
    block_size,
    b,
    s_q,
    cache_seqlens,
    h_q,
    h_kv,
    d,
    dv,
    causal,
):
    logger.debug("GEMS_KUNLUNXIN FLASH_MLA")
    assert causal, "causal False not supported"
    assert d > dv, "mla with rope dim should be larger than no rope dim"
    assert s_q == 1, "kunlunxin flash_mla decode supports s_q == 1 only"

    batch_size, s_q, head_num, d = list(q.shape)
    q = q.view([-1, head_num, d]).contiguous()
    blocked_k = blocked_k.view([-1, d]).contiguous()
    block_table = block_table.contiguous()
    cache_seqlens = cache_seqlens.contiguous().to(torch.int32)

    sm_scale = 1 / math.sqrt(d)

    SQ = batch_size  # s_q == 1 -> one attention problem per batch
    DP = dv  # nope dim (512)
    TD = d - dv  # rope dim (64)
    DV = dv

    # over-allocate the token axis to a whole tile so every store is unmasked
    TP = triton.cdiv(max_seqlen_pad, _BT) * _BT
    NT = TP // _BT

    dev = q.device
    gkv_dt = torch.empty((SQ, d, TP), dtype=q.dtype, device=dev)
    gkv_td = torch.empty((SQ, TP, d), dtype=q.dtype, device=dev)
    valid = torch.empty((SQ, TP), dtype=torch.float32, device=dev)
    logits = torch.empty((SQ, head_num, TP), dtype=torch.float32, device=dev)
    probs = torch.empty((SQ, head_num, TP), dtype=q.dtype, device=dev)
    max_logits = torch.empty((SQ, head_num), dtype=torch.float32, device=dev)
    lse = torch.empty((SQ, head_num), dtype=torch.float32, device=dev)
    o = torch.empty((SQ, head_num, dv), dtype=q.dtype, device=dev)

    stride_kvn = blocked_k.stride(0)
    stride_bt = block_table.stride(0)

    with torch_device_fn.device(device):
        _valid_mask_paged[(SQ, NT)](cache_seqlens, valid, TP, _BT)
        _gather_paged_dt[(SQ, NT, d // _BD)](
            blocked_k,
            block_table,
            cache_seqlens,
            gkv_dt,
            stride_kvn,
            stride_bt,
            block_size,
            TP,
            d,
            _BT,
            _BD,
        )
        _gather_paged_td[(SQ, NT, d // _BD)](
            blocked_k,
            block_table,
            cache_seqlens,
            gkv_td,
            stride_kvn,
            stride_bt,
            block_size,
            TP,
            d,
            _BT,
            _BD,
        )
        _qk_logits[(SQ, head_num // _BH, NT)](
            q,
            gkv_dt,
            logits,
            q.stride(0),
            q.stride(1),
            TP,
            head_num,
            d,
            DP,
            TD,
            _BH,
            _BT,
        )
        _softmax_stats[(SQ, head_num)](
            logits,
            valid,
            probs,
            max_logits,
            lse,
            None,
            sm_scale,
            max_logits.stride(0),
            lse.stride(0),
            TP,
            NT,
            head_num,
            False,
            _BT,
        )
        _pv_matmul[(SQ, head_num // _BH, DV // _BDV)](
            probs,
            gkv_td,
            o,
            o.stride(0),
            o.stride(1),
            TP,
            NT,
            head_num,
            d,
            DV,
            _BH,
            _BT,
            _BDV,
        )

    return o.view([b, s_q, h_q, dv])


def _install():
    """Route ``flag_gems.flash_mla`` to the XPU3 split-kernel implementation.

    ``flash_mla`` is consumed by direct attribute access (the test calls
    ``flag_gems.flash_mla(...)``), bound at ``flag_gems`` import time via
    ``from flag_gems.fused import *``. The submodule attribute, the
    ``flag_gems.fused`` re-export and the top-level ``flag_gems`` re-export are
    all patched so a later lookup binds the XPU version.
    """
    for name in ("flag_gems.fused.flash_mla", "flag_gems.fused", "flag_gems"):
        mod = sys.modules.get(name)
        if mod is not None and hasattr(mod, "flash_mla"):
            mod.flash_mla = flash_mla


_install()
