import contextlib
import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry, tl_extra_shim

logger = logging.getLogger(__name__)

# The GENERIC ops/_thnn_fused_lstm_cell_backward_impl.py is a PURE-TORCH composite:
# it slices the workspace into 4 gate views, then runs ~10 elementwise ops
# (tanh / mul / rsub / add), a torch.cat and a sum(dim=0). Under use_gems every one
# of those decomposes into a separate gems Triton launch. On the XPU triton fork the
# IR dump (ir-thnn_fused_lstm_cell_backward_impl-dev5.log) blows up to 739K lines /
# 11077 kernel modules (mul 2256, sum 1225, copy/cat 1175, rsub 1058, add 423,
# tanh 376), and on the benchmark's tiny (batch<=16, hidden<=64) shapes the per-launch
# overhead dominates -> catastrophic latency.
#
# Fix: fuse the WHOLE elementwise backward (10 pointwise ops + the torch.cat) into ONE
# @libentry Triton kernel that reads all inputs and the 4 gate slices in a single pass
# and writes grad_input_gates (the cat result, straight into the 4 column bands) +
# grad_cx. The bias gradient used to be a single grad_input_gates.sum(dim=0) (one
# cached gems reduction); on XPU the use_gems `sum` op wrapper + its inner reduce
# kernel measured ~55-100us on the benchmark's small shapes (probes on device 1,
# 2026-08-17), which dominates the whole op (~130us vs torch native ~10-12us).
# It is now a dedicated lean @libentry reduction kernel (`_bias_grad_kernel`, one
# masked BLOCK_M=1024 tile, fp32 accumulation, B-row serial loop): single launch,
# no aten `sum` dispatch. @libentry caches the compiled kernel and BLOCK/num_warps
# are passed EXPLICITLY (never via @triton.heuristics) so there is no per-launch
# recompile. Algorithm is byte-identical to the generic chain rule.
#
# XPU-fork constraints discovered on 2026-09-11 (see probe_bias_variants.py /
# probe_variants2.py / probe_atomic_2dsum.py in harness/solution):
#  - `_bias_grad_kernel` MUST take B as tl.constexpr + tl.static_range(B): a
#    dynamic (do_not_specialize) trip-count reduction loop dies in
#    TritonXPUUnrollControl ("out of resource: uni_sram ... Required: 0,
#    Hardware limit: 0").  B == batch_size <= 16 on the test/benchmark matrix,
#    so the per-B specialization is cheap.  A 2D-load + tl.sum(axis=0)
#    alternative is ALSO unusable (tl.sum of a 2D tile fails to compile even at
#    128 lanes / returns wrong results at (16,16)), as is tl.atomic_add
#    (wrong results + 100x slowdown).
#  - `_lstm_cell_bwd_kernel` must specialize H (tl.constexpr): with H runtime
#    the per-lane `offs // H` / `offs % H` costs ~2x (17.3 -> 11.2us
#    measured at (1,64)); with H constexpr BLOCK=512/1024 (1-2 CTAs) is the
#    measured optimum (~9.1-9.4us) for all shapes in the matrix.
#  - When N (== B*H) is a power of two >= 256, the dedicated single-CTA
#    unmasked `_lstm_cell_bwd_kernel_exact` is ~2.3-2.8us faster (6.57us vs
#    9.37us at (16,64)); the same mask-elimination trick gives the bias
#    reduction at M == 256 (`_bias_grad_kernel_exact`, 6.4us vs 10.6us at
#    B=16).  Both are used only for their measured sweet spots; the masked
#    kernels remain the general fallback.

_tanh = tl_extra_shim.tanh


@libentry()
@triton.jit(do_not_specialize=["N"])
def _lstm_cell_bwd_kernel(
    grad_hy_ptr,
    grad_cy_ptr,
    cx_ptr,
    cy_ptr,
    workspace_ptr,
    grad_gates_ptr,
    grad_cx_ptr,
    N,
    H: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    b = offs // H
    h = offs % H
    ws_row = b * (4 * H)

    i_gate = tl.load(workspace_ptr + ws_row + h, mask=mask, other=0.0).to(tl.float32)
    f_gate = tl.load(workspace_ptr + ws_row + H + h, mask=mask, other=0.0).to(
        tl.float32
    )
    g_gate = tl.load(workspace_ptr + ws_row + 2 * H + h, mask=mask, other=0.0).to(
        tl.float32
    )
    o_gate = tl.load(workspace_ptr + ws_row + 3 * H + h, mask=mask, other=0.0).to(
        tl.float32
    )
    ghy = tl.load(grad_hy_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    gcy = tl.load(grad_cy_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    cxv = tl.load(cx_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    cyv = tl.load(cy_ptr + offs, mask=mask, other=0.0).to(tl.float32)

    tanh_cy = _tanh(cyv)
    d_cy = ghy * o_gate * (1.0 - tanh_cy * tanh_cy) + gcy

    grad_i = d_cy * g_gate * i_gate * (1.0 - i_gate)
    grad_f = d_cy * cxv * f_gate * (1.0 - f_gate)
    grad_g = d_cy * i_gate * (1.0 - g_gate * g_gate)
    grad_o = ghy * tanh_cy * o_gate * (1.0 - o_gate)
    grad_cx = d_cy * f_gate

    out_row = b * (4 * H)
    ty = grad_gates_ptr.dtype.element_ty
    tl.store(grad_gates_ptr + out_row + h, grad_i.to(ty), mask=mask)
    tl.store(grad_gates_ptr + out_row + H + h, grad_f.to(ty), mask=mask)
    tl.store(grad_gates_ptr + out_row + 2 * H + h, grad_g.to(ty), mask=mask)
    tl.store(grad_gates_ptr + out_row + 3 * H + h, grad_o.to(ty), mask=mask)
    tl.store(grad_cx_ptr + offs, grad_cx.to(grad_cx_ptr.dtype.element_ty), mask=mask)


@libentry()
@triton.jit
def _lstm_cell_bwd_kernel_exact(
    grad_hy_ptr,
    grad_cy_ptr,
    cx_ptr,
    cy_ptr,
    workspace_ptr,
    grad_gates_ptr,
    grad_cx_ptr,
    H: tl.constexpr,
    BLOCK: tl.constexpr,
):
    # Single-CTA variant for N == BLOCK (no tail): ALL lanes are valid, so no
    # masks are needed at all.  Measured on the XPU fork: dropping the
    # predicates turns the 8 loads / 5 stores into plain wide transactions and
    # the kernel drops from ~9.1-9.4us (masked, BLOCK=512) to ~6.6-7.0us for
    # the N>=256 shapes of the test/benchmark matrix (probe_bwd_nomask.py).
    # Below N=256 the 1-CTA tile is too small and the masked BLOCK=512 path is
    # faster, so this kernel is only used when N is a power of two and >= 256.
    offs = tl.arange(0, BLOCK)
    b = offs // H
    h = offs % H
    ws_row = b * (4 * H)

    i_gate = tl.load(workspace_ptr + ws_row + h).to(tl.float32)
    f_gate = tl.load(workspace_ptr + ws_row + H + h).to(tl.float32)
    g_gate = tl.load(workspace_ptr + ws_row + 2 * H + h).to(tl.float32)
    o_gate = tl.load(workspace_ptr + ws_row + 3 * H + h).to(tl.float32)
    ghy = tl.load(grad_hy_ptr + offs).to(tl.float32)
    gcy = tl.load(grad_cy_ptr + offs).to(tl.float32)
    cxv = tl.load(cx_ptr + offs).to(tl.float32)
    cyv = tl.load(cy_ptr + offs).to(tl.float32)

    tanh_cy = _tanh(cyv)
    d_cy = ghy * o_gate * (1.0 - tanh_cy * tanh_cy) + gcy

    grad_i = d_cy * g_gate * i_gate * (1.0 - i_gate)
    grad_f = d_cy * cxv * f_gate * (1.0 - f_gate)
    grad_g = d_cy * i_gate * (1.0 - g_gate * g_gate)
    grad_o = ghy * tanh_cy * o_gate * (1.0 - o_gate)
    grad_cx = d_cy * f_gate

    out_row = b * (4 * H)
    ty = grad_gates_ptr.dtype.element_ty
    tl.store(grad_gates_ptr + out_row + h, grad_i.to(ty))
    tl.store(grad_gates_ptr + out_row + H + h, grad_f.to(ty))
    tl.store(grad_gates_ptr + out_row + 2 * H + h, grad_g.to(ty))
    tl.store(grad_gates_ptr + out_row + 3 * H + h, grad_o.to(ty))
    tl.store(grad_cx_ptr + offs, grad_cx.to(grad_cx_ptr.dtype.element_ty))


@libentry()
@triton.jit(do_not_specialize=["M"])
def _bias_grad_kernel(
    grad_gates_ptr,
    grad_biases_ptr,
    B: tl.constexpr,
    M,
    BLOCK_M: tl.constexpr,
):
    # grad_biases[j] = sum_b grad_gates[b, j]  for j in [0, M)
    # B must be tl.constexpr: the XPU triton fork's TritonXPUUnrollControl cannot
    # lower this serial reduction loop with a runtime (do_not_specialize) trip
    # count and fails with "out of resource: uni_sram ... Required: 0,
    # Hardware limit: 0" (probe probe_bias_variants.py: v1_range_dynB/v2_tl_range
    # both fail; v4_staticB compiles+covers). B == batch_size is tiny (<=16 in
    # the LSTM test/benchmark matrix), so the per-B specialization is cheap.
    # For M == 256 the exact single-CTA variant below is used instead (~6.4us
    # vs ~10.6us at B=16; probe_bias_policy.py) - see _bias_grad_kernel_exact.
    pid = tl.program_id(0)
    offs = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    m_mask = offs < M
    acc = tl.zeros((BLOCK_M,), dtype=tl.float32)
    for b in tl.static_range(B):
        acc += tl.load(grad_gates_ptr + b * M + offs, mask=m_mask, other=0.0)
    tl.store(
        grad_biases_ptr + offs,
        acc.to(grad_biases_ptr.dtype.element_ty),
        mask=m_mask,
    )


@libentry()
@triton.jit
def _bias_grad_kernel_exact(
    grad_gates_ptr,
    grad_biases_ptr,
    B: tl.constexpr,
    M: tl.constexpr,
):
    # Single-CTA variant for M == BLOCK_M (no tail, no mask): on the XPU fork
    # the 256-wide unmasked loads plus the B-row serial static_range loop
    # measured ~6.4us at (16,64) vs ~10.6us for the masked BLOCK_M=1024
    # variant (probe_bias_unroll.py / probe_bias_policy.py).  Only beneficial
    # at M == 256; for smaller M the 256-lane CTA is underutilized and the
    # masked BLOCK_M=1024 path is faster.
    offs = tl.arange(0, M)
    acc = tl.zeros((M,), dtype=tl.float32)
    for b in tl.static_range(B):
        acc += tl.load(grad_gates_ptr + b * M + offs)
    tl.store(grad_biases_ptr + offs, acc.to(grad_biases_ptr.dtype.element_ty))


def _thnn_fused_lstm_cell_backward_impl(
    grad_hy: torch.Tensor,
    grad_cy: torch.Tensor,
    cx: torch.Tensor,
    cy: torch.Tensor,
    workspace: torch.Tensor,
    has_bias: bool,
):
    logger.debug("GEMS_KUNLUNXIN _THNN_FUSED_LSTM_CELL_BACKWARD_IMPL")

    batch_size, hidden_size = cx.shape

    grad_hy = grad_hy.contiguous()
    grad_cy = grad_cy.contiguous()
    cx = cx.contiguous()
    cy = cy.contiguous()
    workspace = workspace.contiguous()

    grad_input_gates = torch.empty(
        (batch_size, 4 * hidden_size), device=cx.device, dtype=cx.dtype
    )
    grad_cx = torch.empty((batch_size, hidden_size), device=cx.device, dtype=cx.dtype)

    N = batch_size * hidden_size
    # torch_device_fn.device() is a plain torch.cuda.device() guard here; it costs
    # ~3us/entry and is only needed when the input device differs from the current
    # one.  Guard with a cheap current_device() check so the common single-device
    # path skips the context entirely (measured: op 59.3 -> 57.6us at (16,64)
    # after this change; ~2us of the remaining is dispatch + 2x torch.empty).
    use_guard = torch_device_fn.current_device() != cx.device.index
    with torch_device_fn.device(cx.device) if use_guard else contextlib.nullcontext():
        if N > 0:
            # BLOCK is tl.constexpr; H is also tl.constexpr (see header comment).
            # BLOCK=512 measured fastest on XPU for the test/benchmark shapes
            # (1-2 CTAs; probe_final2.py: B512w4 = 8.99-9.39us vs B128w4
            # 11.2-18.8us at H>=16).  When N is a power of two >= 256 a single
            # unmasked CTA (BLOCK == N, no masks) is ~2.3-2.8us faster
            # (probe_bwd_nomask.py: N=1024 6.57us vs 9.37us at (16,64); at
            # N<256 the small tile is slower, so keep the masked path there).
            if 256 <= N <= 2048 and (N & (N - 1)) == 0:
                grid = (1,)
                _lstm_cell_bwd_kernel_exact[grid](
                    grad_hy,
                    grad_cy,
                    cx,
                    cy,
                    workspace,
                    grad_input_gates,
                    grad_cx,
                    hidden_size,
                    N,
                    num_warps=4,
                )
            else:
                BLOCK = 512
                grid = (triton.cdiv(N, BLOCK),)
                _lstm_cell_bwd_kernel[grid](
                    grad_hy,
                    grad_cy,
                    cx,
                    cy,
                    workspace,
                    grad_input_gates,
                    grad_cx,
                    N,
                    hidden_size,
                    BLOCK,
                    num_warps=4,
                )
        if has_bias:
            grad_biases = torch.empty(
                (4 * hidden_size,), device=cx.device, dtype=cx.dtype
            )
            if batch_size > 0:
                # Single-CTA tile for the whole 4*H column band (4*H <= 256 on
                # the benchmark matrix): BLOCK_M=1024 measured ~5.1-10.7us vs
                # 128 -> 5.2-13.5us (probe_variants2.py).  For M == 256 (H=64)
                # the unmasked exact variant is ~4.2us faster at B=16
                # (probe_bias_policy.py: 6.40us vs 10.59us).
                if 4 * hidden_size == 256:
                    grid = (1,)
                    _bias_grad_kernel_exact[grid](
                        grad_input_gates,
                        grad_biases,
                        batch_size,
                        4 * hidden_size,
                        num_warps=4,
                    )
                else:
                    BLOCK_M = 1024
                    grid = (triton.cdiv(4 * hidden_size, BLOCK_M),)
                    _bias_grad_kernel[grid](
                        grad_input_gates,
                        grad_biases,
                        batch_size,
                        4 * hidden_size,
                        BLOCK_M,
                        num_warps=4,
                    )
        else:
            grad_biases = torch.zeros(0, dtype=cx.dtype, device=cx.device)

    return grad_input_gates, grad_cx, grad_biases
