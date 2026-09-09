import math

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn


# Kunlunxin (XPU) override of aten::replication_pad3d_backward.
#
# The generic implementation (src/flag_gems/ops/replication_pad3d_backward.py)
# scatters grad_output with `tl.atomic_add` into the clamped input voxels.  On
# TritonXPU that is unreliable: several *programs* can touch the same input
# address concurrently (multiple output positions clamp to one input cell) and
# `tl.atomic_add` silently loses updates (verified on (2,4,32,64,512) pad
# (2,3,1,1,0,2): maxdiff flips 2e-6 <-> 1.5-1.8 across identical runs).
#
# Fix (same shape of fix as the vendor replication_pad2d_backward): the
# backward of replication pad is a separable box-fold
#
#     gi[d,h,w] = sum_{a<len(Gd(d))} sum_{b<len(Gh(h))} sum_{c<len(Gw(w))}
#                     go[Gd(d).lo+a, Gh(h).lo+b, Gw(w).lo+c]
#
# with per-axis preimage groups (for input size n, output size no, pad p0):
#     i==0:     G = [0, p0+1)          (i==n-1==0: [0, no))
#     i==n-1:   G = [p0+n-1, no)       (empty when p0+n-1 >= no)
#     interior: G = [p0+i, p0+i+1)     (empty when outside [0, no))
# This implementation is ATOMIC-FREE: every grad_input cell is written by
# exactly one program, reading a disjoint, complete partition of grad_output.
# Two passes (separable):
#   1. W-fold:        cf[d_out, h_out, w] = sum over Gw(w) of go[d_out, h_out, :]
#   2. D-H-fold:      gi[d, h, w] = sum over Gd(d) x Gh(h) of cf[:, :, w]
# Axes of size 1 are pre-folded once with a vendor `sum` (a size-1 axis makes
# its group span the whole padded dimension, which would blow the
# `tl.static_range` trip count).
#
# Correctness rules carried over from the vendor 2D fix:
# - Loop loads always use clamped in-bounds offsets plus a register-level
#   `tl.where(sel, v, 0.0)` select; a masked-load result never feeds a sum.
# - All accumulation is fp32 in registers; the final store auto-casts to the
#   output dtype (fp16/bf16 round once from fp32), so there is no fp32
#   intermediate buffer and no extra cast pass.
# - The tail mask is only applied on the (clamped) load/store pair.


@triton.jit
def _replication_pad3d_backward_wfold_kernel(
    go_ptr,
    cf_ptr,
    D_eff,  # 1 if the D axis was pre-folded, else D_out
    H_eff,  # 1 if the H axis was pre-folded, else H_out
    W_in,
    W_out,
    pad_left,
    pad_right,
    total,  # NC * D_eff * H_eff * W_in
    MAXG: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    o = pid * BLOCK + tl.arange(0, BLOCK)
    mask = o < total

    iw = o % W_in
    rest = o // W_in
    oh = rest % H_eff
    rest2 = rest // H_eff
    od = rest2 % D_eff
    nc = rest2 // D_eff

    lo_raw = tl.where(
        iw == 0, 0, tl.where(iw == W_in - 1, pad_left + W_in - 1, pad_left + iw)
    )
    lo = tl.minimum(tl.maximum(lo_raw, 0), W_out - 1)
    cnt = tl.where(
        iw == 0,
        tl.where(W_in == 1, W_out, tl.maximum(pad_left + 1, 0)),
        tl.where(
            iw == W_in - 1,
            tl.where(lo_raw >= W_out, 0, W_out - tl.maximum(lo_raw, 0)),
            tl.where((lo_raw >= 0) & (lo_raw < W_out), 1, 0),
        ),
    )

    out_base = ((nc * D_eff + od) * H_eff + oh) * W_out
    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    for c in tl.static_range(MAXG):
        cc = tl.minimum(c, tl.maximum(cnt - 1, 0))
        v = tl.load(go_ptr + out_base + lo + cc, mask=mask, other=0.0).to(tl.float32)
        acc += tl.where(c < cnt, v, 0.0)
    tl.store(cf_ptr + o, acc, mask=mask)


@triton.jit
def _replication_pad3d_backward_dhfold_kernel(
    cf_ptr,
    gi_ptr,
    D_in,
    H_in,
    W_in,
    D_out,
    H_out,
    W_out,
    pad_left,
    pad_top,
    pad_front,
    folded_d,  # 1: the D axis was pre-folded (single group [0,1))
    folded_h,  # 1: the H axis was pre-folded (single group [0,1))
    total,  # NC * D_in * H_in * W_in
    MAXD: tl.constexpr,
    MAXH: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    o = pid * BLOCK + tl.arange(0, BLOCK)
    mask = o < total

    iw = o % W_in
    rest = o // W_in
    ih = rest % H_in
    rest2 = rest // H_in
    id_ = rest2 % D_in
    nc = rest2 // D_in

    lo_raw_d = tl.where(
        id_ == 0,
        0,
        tl.where(id_ == D_in - 1, pad_front + D_in - 1, pad_front + id_),
    )
    lo_d = tl.minimum(tl.maximum(lo_raw_d, 0), D_out - 1)
    cnt_d = tl.where(
        id_ == 0,
        tl.where(D_in == 1, D_out, tl.maximum(pad_front + 1, 0)),
        tl.where(
            id_ == D_in - 1,
            tl.where(lo_raw_d >= D_out, 0, D_out - tl.maximum(lo_raw_d, 0)),
            tl.where((lo_raw_d >= 0) & (lo_raw_d < D_out), 1, 0),
        ),
    )
    lo_d = tl.where(folded_d == 1, 0, lo_d)
    cnt_d = tl.where(folded_d == 1, 1, cnt_d)

    lo_raw_h = tl.where(
        ih == 0, 0, tl.where(ih == H_in - 1, pad_top + H_in - 1, pad_top + ih)
    )
    lo_h = tl.minimum(tl.maximum(lo_raw_h, 0), H_out - 1)
    cnt_h = tl.where(
        ih == 0,
        tl.where(H_in == 1, H_out, tl.maximum(pad_top + 1, 0)),
        tl.where(
            ih == H_in - 1,
            tl.where(lo_raw_h >= H_out, 0, H_out - tl.maximum(lo_raw_h, 0)),
            tl.where((lo_raw_h >= 0) & (lo_raw_h < H_out), 1, 0),
        ),
    )
    lo_h = tl.where(folded_h == 1, 0, lo_h)
    cnt_h = tl.where(folded_h == 1, 1, cnt_h)

    d_eff = tl.where(folded_d == 1, 1, D_out)
    h_eff = tl.where(folded_h == 1, 1, H_out)
    cf_base = nc * (d_eff * h_eff * W_in)
    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    for a in tl.static_range(MAXD):
        dd = tl.minimum(a, tl.maximum(cnt_d - 1, 0))
        for b in tl.static_range(MAXH):
            hh = tl.minimum(b, tl.maximum(cnt_h - 1, 0))
            cf_off = (
                cf_base
                + (lo_d + dd) * (h_eff * W_in)
                + (lo_h + hh) * W_in
                + iw
            )
            v = tl.load(cf_ptr + cf_off, mask=mask, other=0.0).to(tl.float32)
            acc += tl.where((a < cnt_d) & (b < cnt_h), v, 0.0)

    in_base = nc * (D_in * H_in * W_in) + (id_ * H_in + ih) * W_in
    tl.store(gi_ptr + in_base + iw, acc, mask=mask)


def _replication_pad3d_backward_launch(
    grad_output: torch.Tensor, self: torch.Tensor, padding
) -> torch.Tensor:
    if not isinstance(padding, (list, tuple)) or len(padding) != 6:
        raise ValueError("padding must contain six values")
    if self.dim() < 3:
        raise ValueError("self must have at least three dimensions")
    if grad_output.device != self.device or grad_output.dtype != self.dtype:
        raise ValueError("grad_output and self must have the same device and dtype")
    if self.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise ValueError("replication_pad3d_backward supports floating point dtypes")

    pad_left, pad_right, pad_top, pad_bottom, pad_front, pad_back = map(int, padding)
    x = self.contiguous()
    grad_output = grad_output.contiguous()
    d_in, h_in, w_in = (int(x.shape[-3]), int(x.shape[-2]), int(x.shape[-1]))
    d_out = d_in + pad_front + pad_back
    h_out = h_in + pad_top + pad_bottom
    w_out = w_in + pad_left + pad_right
    if d_out <= 0 or h_out <= 0 or w_out <= 0:
        raise ValueError("padding results in a non-positive output dimension")
    expected_spatial = (d_out, h_out, w_out)
    if tuple(grad_output.shape[-3:]) != expected_spatial:
        raise ValueError(
            "grad_output spatial shape "
            f"{tuple(grad_output.shape[-3:])} does not match {expected_spatial}"
        )
    if tuple(grad_output.shape[:-3]) != tuple(x.shape[:-3]):
        raise ValueError("grad_output and self must have matching leading dimensions")

    if all(
        value == 0
        for value in (pad_left, pad_right, pad_top, pad_bottom, pad_front, pad_back)
    ):
        return grad_output.reshape(x.shape)

    batch = math.prod(x.shape[:-3]) if x.dim() > 3 else 1
    x_flat = x.reshape(batch, d_in, h_in, w_in)
    go_flat = grad_output.reshape(batch, d_out, h_out, w_out)

    # Pre-fold (with a vendor reduction) every input axis of size 1: for such
    # an axis the preimage group spans the whole padded dimension, which
    # cannot be represented as a bounded static_range fold.  The fold runs in
    # fp32: the vendor reduction accumulates in the input dtype, which loses
    # ~2-4 ulp on f16/bf16 sums and exceeds the reference tolerance.
    data = go_flat.float()
    if d_in == 1:
        data = data.sum(dim=-3, keepdim=True)
    if h_in == 1:
        data = data.sum(dim=-2, keepdim=True)
    if w_in == 1:
        data = data.sum(dim=-1, keepdim=True)
    data = data.contiguous()
    d_eff = 1 if d_in == 1 else d_out
    h_eff = 1 if h_in == 1 else h_out

    gi = torch.empty((batch, d_in, h_in, w_in), device=x.device, dtype=x.dtype)
    with torch_device_fn.device(x.device):
        if w_in > 1:
            cf = torch.empty(
                (batch, d_eff, h_eff, w_in), device=x.device, dtype=torch.float32
            )
            total1 = batch * d_eff * h_eff * w_in
            _replication_pad3d_backward_wfold_kernel[(triton.cdiv(total1, 256),)](
                data,
                cf,
                d_eff,
                h_eff,
                w_in,
                w_out,
                pad_left,
                pad_right,
                total1,
                MAXG=max(pad_left + 1, pad_right + 1, 1),
                BLOCK=256,
            )
        else:
            cf = data
        total2 = batch * d_in * h_in * w_in
        _replication_pad3d_backward_dhfold_kernel[(triton.cdiv(total2, 256),)](
            cf,
            gi,
            d_in,
            h_in,
            w_in,
            d_out,
            h_out,
            w_out,
            pad_left,
            pad_top,
            pad_front,
            1 if d_in == 1 else 0,
            1 if h_in == 1 else 0,
            total2,
            MAXD=(1 if d_in == 1 else max(pad_front + 1, pad_back + 1, 1)),
            MAXH=(1 if h_in == 1 else max(pad_top + 1, pad_bottom + 1, 1)),
            BLOCK=256,
        )
    return gi.reshape(x.shape)


def replication_pad3d_backward(
    grad_output: torch.Tensor, self: torch.Tensor, padding
) -> torch.Tensor:
    return _replication_pad3d_backward_launch(grad_output, self, padding)