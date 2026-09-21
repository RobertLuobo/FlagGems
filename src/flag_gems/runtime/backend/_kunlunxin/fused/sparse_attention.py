import torch
import triton
import triton.language as tl


@triton.jit
def _sparse_attn_expand_src_kernel(
    IDX,
    SRC,
    stride_idxb,
    stride_idxm,
    topk,
    kv_len,
    M,
    TP: tl.constexpr,
    BLOCK: tl.constexpr,
):
    i_pid = tl.program_id(0)
    offs = i_pid * BLOCK + tl.arange(0, BLOCK)
    row = offs // TP
    i_t = offs % TP
    i_b = row // M
    i_m = row % M
    valid = i_t < topk
    i_tc = tl.minimum(i_t, topk - 1)
    ids = tl.load(IDX + i_b * stride_idxb + i_m * stride_idxm + i_tc)
    ids = tl.where(valid & (ids >= 0), ids, 0)
    tl.store(SRC + offs, i_b * kv_len + ids)


@triton.jit
def _sparse_attn_gather_flat_kernel(
    KV,
    GKV,
    SRC,
    D: tl.constexpr,
    BLOCK: tl.constexpr,
):
    i_pid = tl.program_id(0)
    offs = i_pid * BLOCK + tl.arange(0, BLOCK)
    row = offs // D
    col = offs % D
    src = tl.load(SRC + row)
    val = tl.load(KV + src * D + col)
    tl.store(GKV + offs, val)


@triton.jit
def _sparse_attn_qk_kernel(
    Q,
    GKV,
    LOGITS,
    stride_qb,
    scale,
    D: tl.constexpr,
    TP: tl.constexpr,
    BT: tl.constexpr,
    HP: tl.constexpr,
):
    i_bm = tl.program_id(0)
    i_t = tl.program_id(1)
    offs_h = tl.arange(0, HP)
    offs_t = i_t * BT + tl.arange(0, BT)
    offs_d = tl.arange(0, D)
    q = tl.load(Q + i_bm * stride_qb + offs_h[:, None] * D + offs_d[None, :])
    kv = tl.load(GKV + i_bm * (TP * D) + offs_t[:, None] * D + offs_d[None, :])
    acc = tl.dot(q, tl.trans(kv), out_dtype=tl.float32) * scale
    tl.store(LOGITS + (i_bm * HP + offs_h[:, None]) * TP + offs_t[None, :], acc)


@triton.jit
def _sparse_attn_softmax_kernel(
    LOGITS,
    PROBS,
    attn_sink,
    topk,
    H_ACTUAL,
    TP: tl.constexpr,
    HP: tl.constexpr,
):
    i_bm = tl.program_id(0)
    offs_h = tl.arange(0, HP)
    offs_t = tl.arange(0, TP)
    base = i_bm * HP * TP
    x = tl.load(LOGITS + base + offs_h[:, None] * TP + offs_t[None, :])
    x = tl.where(offs_t[None, :] < topk, x, float("-inf"))
    rmax = tl.max(x, axis=1)
    sink_val = tl.load(attn_sink + tl.minimum(offs_h, H_ACTUAL - 1))
    rsum = tl.exp(sink_val - rmax) + tl.sum(tl.exp(x - rmax[:, None]), axis=1)
    lse = rmax + tl.math.log(rsum)
    p = tl.exp(x - lse[:, None])
    p = tl.where(offs_t[None, :] < topk, p, 0.0)
    tl.store(PROBS + base + offs_h[:, None] * TP + offs_t[None, :], p.to(tl.bfloat16))


@triton.jit
def _sparse_attn_pv_kernel(
    PROBS,
    GKV,
    O,
    D: tl.constexpr,
    TP: tl.constexpr,
    BT: tl.constexpr,
    BD: tl.constexpr,
    HP: tl.constexpr,
):
    i_bm = tl.program_id(0)
    i_d = tl.program_id(1)
    offs_h = tl.arange(0, HP)
    offs_v = i_d * BD + tl.arange(0, BD)
    acc = tl.zeros([HP, BD], dtype=tl.float32)
    for i_t in range(TP // BT):
        offs_t = i_t * BT + tl.arange(0, BT)
        pb = tl.load(PROBS + (i_bm * HP + offs_h[:, None]) * TP + offs_t[None, :])
        vb = tl.load(GKV + i_bm * (TP * D) + offs_t[:, None] * D + offs_v[None, :])
        acc = tl.dot(pb, vb, acc, out_dtype=tl.float32)
    tl.store(O + (i_bm * HP + offs_h[:, None]) * D + offs_v[None, :], acc.to(tl.bfloat16))


def sparse_attn_triton(
    q: torch.Tensor,
    kv: torch.Tensor,
    attn_sink: torch.Tensor,
    topk_idxs: torch.Tensor,
    softmax_scale: float,
) -> torch.Tensor:
    b, m, h, d = q.shape
    topk = topk_idxs.shape[-1]
    kv_len = kv.shape[1]

    BT = 64
    bm = b * m
    HP = triton.next_power_of_2(h)
    BDP = triton.next_power_of_2(d)
    TP = max(BT, triton.next_power_of_2(topk))
    BD = min(512, BDP)
    SBLK = min(1024, TP)
    BLK = min(16384, TP * BDP)
    NM = bm * TP
    NG = NM * BDP

    if BDP == d:
        kv_pad = kv
    else:
        kv_pad = torch.zeros((b, kv_len, BDP), device=kv.device, dtype=kv.dtype)
        kv_pad[:, :, :d] = kv
    q_pad = torch.zeros((bm, HP, BDP), device=q.device, dtype=q.dtype)
    q_pad[:, :h, :d] = q.reshape(bm, h, d)

    src = torch.empty((NM,), device=q.device, dtype=torch.int32)
    gkv = torch.empty((NG,), device=q.device, dtype=torch.bfloat16)
    logits = torch.empty((bm * HP, TP), device=q.device, dtype=torch.float32)
    probs = torch.empty((bm * HP, TP), device=q.device, dtype=torch.bfloat16)
    o_pad = torch.empty((bm * HP, BDP), device=q.device, dtype=q.dtype)

    _sparse_attn_expand_src_kernel[(NM // SBLK,)](
        topk_idxs,
        src,
        topk_idxs.stride(0),
        topk_idxs.stride(1),
        topk,
        kv_len,
        m,
        TP=TP,
        BLOCK=SBLK,
    )

    _sparse_attn_gather_flat_kernel[(NG // BLK,)](kv_pad, gkv, src, D=BDP, BLOCK=BLK)

    _sparse_attn_qk_kernel[(bm, TP // BT)](
        q_pad, gkv, logits, q_pad.stride(0), softmax_scale,
        D=BDP, TP=TP, BT=BT, HP=HP,
    )

    _sparse_attn_softmax_kernel[(bm,)](
        logits, probs, attn_sink, topk, h, TP=TP, HP=HP,
    )

    _sparse_attn_pv_kernel[(bm, BDP // BD)](
        probs, gkv, o_pad, D=BDP, TP=TP, BT=BT, BD=BD, HP=HP,
    )

    return o_pad.reshape(bm, HP, BDP)[:, :h, :d].reshape(b, m, h, d).contiguous()
