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

import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as ext

logger = logging.getLogger(__name__)


def _dests_per_program(out_numel: int) -> int:
    """每个 program 负责多少个输出元素。

    原实现是 `grid = (input.numel(),)`，即**一个输出元素一个 program**：
      * program 派发在该后端实测 0.113 us/program（HARNESS_SUMMARY §4.1），
        `numel=4096` 光派发就 0.46 ms，而 torch 参考侧整体只有 0.193 ms；
      * 每个 program 都要把 mask/index0..2/values 五个数组整读一遍（32 B/source），
        所以访存量是 `numel * mask_numel * 32 B`。
    让一个 program 负责 DESTS 个输出元素后，源数组只读一次即可复用给 DESTS 个目标，
    派发数与访存量同时降 DESTS 倍。DESTS 上限取 32 是为了压住 `tl.static_range` 的展开体积。
    """
    if out_numel >= 512:
        return 32
    return max(1, min(32, triton.next_power_of_2(out_numel) // 16))


@libentry()
@triton.jit(do_not_specialize=["mask_numel", "out_numel"])
def _unsafe_masked_index_put_accumulate_kernel(
    out_ptr,
    inp_ptr,
    mask,
    index0,
    index1,
    index2,
    values,
    mask_numel,
    out_numel,
    SHAPE0: tl.constexpr,
    SHAPE1: tl.constexpr,
    SHAPE2: tl.constexpr,
    STRIDE0: tl.constexpr,
    STRIDE1: tl.constexpr,
    STRIDE2: tl.constexpr,
    RANK: tl.constexpr,
    DESTS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    dest_base = ext.program_id(0) * DESTS

    offsets = tl.arange(0, BLOCK_SIZE)
    active = offsets < mask_numel
    keep = tl.load(mask + offsets, mask=active, other=0) != 0

    # 一次性算出每个 source 的**扁平目的地偏移**，而不是每个目标各比一遍三个坐标。
    # 越界/负下标先 clamp 再用（HARNESS_SUMMARY §3.2：该后端在 mask 生效前就把地址交给
    # gm2lm，靠 mask 屏蔽非法下标会真读越界并打挂整卡）；clamp 到 [0, SHAPE-1] 与
    # torch 的 `index.clamp(min=-size, max=size-1)` + index_put 在非负下标上等价。
    i0 = tl.load(index0 + offsets, mask=active, other=0).to(tl.int32)
    i0 = tl.minimum(tl.maximum(i0, 0), SHAPE0 - 1)
    dest = i0 * STRIDE0
    if RANK >= 2:
        i1 = tl.load(index1 + offsets, mask=active, other=0).to(tl.int32)
        i1 = tl.minimum(tl.maximum(i1, 0), SHAPE1 - 1)
        dest += i1 * STRIDE1
    if RANK >= 3:
        i2 = tl.load(index2 + offsets, mask=active, other=0).to(tl.int32)
        i2 = tl.minimum(tl.maximum(i2, 0), SHAPE2 - 1)
        dest += i2 * STRIDE2

    # mask 掉的 source 与 tile 尾部空 lane 一律把 update 置 0，
    # 于是下面的匹配不需要再带 mask，`dest` 撞上任何目标都贡献 0。
    update = tl.load(values + offsets, mask=active, other=0.0).to(tl.float32)
    update = tl.where(keep & active, update, 0.0)

    for c in tl.static_range(DESTS):
        out_off = dest_base + c
        # out 缓冲区尾部多留了 DESTS 个元素做哨兵，所以这里的 store 不需要 mask
        # （§3.2：离散/标量 store 的 mask 在地址碰撞时不可依赖，宁可写进合法的填充区）。
        acc = tl.sum(tl.where(dest == out_off, update, 0.0), axis=0)
        in_off = tl.minimum(out_off, out_numel - 1)
        base = tl.load(inp_ptr + in_off).to(tl.float32)
        tl.store(out_ptr + out_off, base + acc)


# ---------------------------------------------------------------------------
# 多轮「胜者循环」路径（大规模）
#
# match kernel 是 O(out_numel * mask_numel) 的：131072 实测 1253 ms，而 torch 参考只有
# 5.01 ms —— 结构性不可达，必须换成 O(mask_numel) 的算法。
# 该后端上两条路都被实测堵死：
#   * `tl.atomic_add` 21.5 ms / 131072（151 ns/elem），**且 mask 全 false 也照收全价**
#     （probe_round.log Q4：all_false / sparse_16 / all_true 三者都是 21.5 ms）；
#   * 排序 / 前缀和需要 4+ kernel 且踩 TritonXPU 的多处崩溃点。
# 于是用「无 atomic 的胜者循环」：靠**离散 store 的天然单胜者语义**每轮从每个目标里
# 挑出恰好一个源，R 轮后覆盖到最大重数为止（probe_round.log Q1 实测：
# touched==nonzero_tag、bad_winner=0、missing=0，语义成立）。
#
# 关键性能约束：**离散访存的代价只取决于「有多少 lane 打在同一个地址上」**，
# 与地址是直接 load 还是 tl.where 算出来的无关（probe_round4.log，单变量只改重复度，
# kernel 体一字不动，M=N=131072 / BLOCK=2048 / grid=64）：
#   dup=0.00（地址在 N 个槽上随机，Poisson(1) 重数）  scatter 0.126 ms（ 0.96 ns/elem）
#   dup=0.25（1/4 的 lane 打同一个槽）                scatter 6.330 ms（48.3 ns/elem）
#   dup=0.50                                          scatter 12.61 ms（96.2 ns/elem）
#   dup=1.00（全部 lane 打同一个槽）                  scatter 25.21 ms（192  ns/elem）
# 即同址碰撞被完全串行化，单价 ~192 ns/lane —— 与 `tl.atomic_add` 的 151~192 ns/elem
# 恰好同一个量级，可以认为离散 scatter 的碰撞消解走的就是 atomic 那条串行通路。
# 离散 gather 对碰撞敏感得多但绝对值低（0.16 -> 4.5 ns/elem）。
#
# 这条结论**推翻了本文件上一版的归因**：probe_scatter_variants.log 里
# 「`addr = tl.where(cond, d, N)` 比 `addr = d` 慢 98x」并不是「算出来的地址掉标量路径」，
# 而是那个 tl.where 把一半 lane 全部折到同一个哨兵槽 N 上（dup=0.5，实测 96.2 ns/elem，
# 与 v2 的 96 ns/elem 逐位吻合）。probe_round3.log 是判决性证据：把完整 round kernel
# 原样喂**随机地址**只要 0.134 ms（w6/w9），同一个 kernel 在 op 里却 12.63 ms，
# 差别只在输入数据的重复度。
#
# 因此本实现的核心设计是**绝不让任何两个 lane 共享一个哨兵槽**：
#   * 源 i 退休/被 mask 掉时，它的地址是**它私有的槽** `out_numel + i`，不是公共的 out_numel；
#   * 目标 d「本轮无人获胜」的标记也是**它私有的值** `d`，不是公共的 0，
#     再借一张 `val_lookup`（前 out_numel 个元素恒为 0，后面才是 values）把它翻译成 0。
# 于是 prep / round / finish / combine 里所有离散访存的重复度都只剩输入自身的
# 自然重数（benchmark 里 Poisson(1)，最大 4~9），全部落在 ~1 ns/elem 的快路径上。
#
# tag 的地址空间（每轮一行，行长 row = out_numel + pad）：
#   [0, out_numel)              目标槽；值 < out_numel 表示「本轮该目标无人获胜」
#   [out_numel, out_numel+pad)  源私有槽；源 i 的 marker 就是 out_numel + i
# 所以 tag 只要**全 0 初始化**即可（0 < out_numel，恒等于「无人」，也永不等于任何 marker），
# 不需要 `torch.arange`——实测该后端 `torch.arange(n, dtype=torch.int32, device='cuda')`
# 会打 `[ASSERT-FAIL](2==SUCCESS) .../xdnn_pytorch_wrapper/arange.cpp:54` + `cast.cpp:59`
# 并返回垃圾值（probe_stage.log 首次运行，垃圾下标直接把 combine 打成
# `illegal memory access`）。**int32 arange 在这个后端不可用。**
# val_lookup 与 tag 同长：[0, out_numel) 恒 0；[out_numel, out_numel+mask_numel) = values。
# 于是 combine 里把「无人」的下标改写成 `offs`（每个目标互不相同、且 val_lookup[offs]==0）
# 就能一次离散 gather 同时完成「有没有胜者」和「胜者的 value 是多少」，且零碰撞。
# ---------------------------------------------------------------------------


@libentry()
@triton.jit(do_not_specialize=["mask_numel", "out_numel"])
def _umipa_prep_kernel(
    dest_buf,
    val_lookup,
    mask,
    index0,
    index1,
    index2,
    values,
    mask_numel,
    out_numel,
    SHAPE0: tl.constexpr,
    SHAPE1: tl.constexpr,
    SHAPE2: tl.constexpr,
    STRIDE0: tl.constexpr,
    STRIDE1: tl.constexpr,
    STRIDE2: tl.constexpr,
    RANK: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """把 (index0..2, mask) 压成一个扁平的 int32 目的地数组，并把 values 搬到 val_lookup 尾部。

    * 越界/负下标先 clamp 再用（§3.2：该后端在 mask 生效前就把地址交给 gm2lm）；
      clamp 到 [0, SHAPE-1] 与 torch decomposition 的 `index.clamp(-size, size-1)` 在
      非负下标上等价。
    * 被 mask 掉的源直接写**它私有的槽** `out_numel + i`（不是公共哨兵，见文件头的碰撞
      实测），此后所有 round kernel 都不必再看 mask。
    * values 搬到 `val_lookup[out_numel + i]`，于是「源 i 的 marker」`out_numel + i`
      既是 tag 里的私有槽下标，又是 val_lookup 里取值的下标，combine 一次 gather 到位。
    """
    offs = ext.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    inb = offs < mask_numel

    i0 = tl.load(index0 + offs, mask=inb, other=0).to(tl.int32)
    i0 = tl.minimum(tl.maximum(i0, 0), SHAPE0 - 1)
    dest = i0 * STRIDE0
    if RANK >= 2:
        i1 = tl.load(index1 + offs, mask=inb, other=0).to(tl.int32)
        i1 = tl.minimum(tl.maximum(i1, 0), SHAPE1 - 1)
        dest += i1 * STRIDE1
    if RANK >= 3:
        i2 = tl.load(index2 + offs, mask=inb, other=0).to(tl.int32)
        i2 = tl.minimum(tl.maximum(i2, 0), SHAPE2 - 1)
        dest += i2 * STRIDE2

    keep = tl.load(mask + offs, mask=inb, other=0) != 0
    marker = (out_numel + offs).to(tl.int32)
    tl.store(dest_buf + offs, tl.where(inb & keep, dest, marker))
    v = tl.load(values + offs, mask=inb, other=0.0)
    tl.store(val_lookup + out_numel + offs, v)


@libentry()
@triton.jit(do_not_specialize=["out_numel"])
def _umipa_round_kernel(dest_buf, tag_prev, tag_cur, out_numel, BLOCK: tl.constexpr):
    """一轮胜者循环。

    上一轮谁的 marker 留在 tag_prev[dest] 上，谁就是上一轮的胜者：把它从 dest_buf 里摘掉
    （地址改成它自己的私有槽 marker），并往 tag_cur[dest] 写一个 < out_numel 的值
    （就写 `dest` 自己）表示「本轮该目标暂时无人」；还活着的 lane 则往 tag_cur[dest]
    写自己的 marker 申领本轮。离散 store 的单胜者语义保证每个目标最多留下一个 marker。

    **已知次优（本轮实测，未修）**：退休的胜者也往 `tag_cur[dest]` 写，会把同目标其它
    活跃 lane 的申领冲掉，于是每一级重数平均要烧掉约两轮 —— probe_stage2.log 里单批
    8 轮跑不完 131072（单批 3.92 ms，整函数 8.11 ms = 两批）。
    正确的修法是让胜者**只写自己的私有槽**，但**不能**写成
    `addr = tl.where(win, marker, d); tl.store(tag_cur + addr, marker)`：
    该写法在本后端稳定触发 `kl3ChannelCheckErrors ... error set to 721`（illegal address）
    并 wedge 整卡，两张卡（gate5 = XPU 2、gate6 = XPU 5）都复现，
    而 probe_bounds.log 已在 host 侧逐 kernel 证明所有中间值都在 [0,row) 内
    （`dest_buf after prep min=2 max=4094 allowed=[0,4095] OK`，round 0 之后即挂）。
    与本版唯一的差别就是离散 store 的**地址表达式**从 `d`（load 出来的向量）换成
    `tl.where(win, marker, d)`（一支是 arange 派生的仿射向量）——怀疑后端的地址仿射分析
    把它误判成连续 store。可行的替代是**拆成两个 kernel**（A 只做退休 + 连续 store，
    B 只做 `tag_cur + d` 的申领；此时 B 里的 `d` 已经是退休后的私有槽，天然不冲突），
    探针见 `harness/perf_ir/unsafe_masked_index_put_accumulate/probe_addr.py`（变体 r5），
    因全机 8 卡在收尾阶段全部 wedge/占用而**未能上机验证**，故本文件保留已验证的这一版。
    """
    offs = ext.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    d = tl.load(dest_buf + offs)
    w = tl.load(tag_prev + d)
    marker = (out_numel + offs).to(tl.int32)
    win = w == marker
    tl.store(dest_buf + offs, tl.where(win, marker, d))
    tl.store(tag_cur + d, tl.where(win, d, marker))


@libentry()
@triton.jit(do_not_specialize=["out_numel"])
def _umipa_retire_kernel(dest_buf, tag_prev, out_numel, BLOCK: tl.constexpr):
    """拆分臂（`mask_numel >= _ROUND_SPLIT_MIN_MASK_NUMEL`）的前半：只做「上一轮的胜者退休」，只写连续的 dest_buf。

    与 `_umipa_round_kernel` 的区别是**不碰 tag**，于是「胜者不冲刷同目标其它 lane 的申领」
    这件事不再需要一个由 `tl.where` 算出来的 store 地址（那个写法打 721，见 round kernel 的
    docstring）。代价是每轮多一次 launch（该后端 ~0.045 ms/launch）。
    r=0 时 tag_prev 是全 0 行，`w == marker` 恒假，等于不退休。
    """
    offs = ext.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    d = tl.load(dest_buf + offs)
    w = tl.load(tag_prev + d)
    marker = (out_numel + offs).to(tl.int32)
    tl.store(dest_buf + offs, tl.where(w == marker, marker, d))


@libentry()
@triton.jit(do_not_specialize=["out_numel"])
def _umipa_claim_kernel(dest_buf, tag_cur, out_numel, BLOCK: tl.constexpr):
    """拆分臂（`mask_numel >= _ROUND_SPLIT_MIN_MASK_NUMEL`）的后半：只做「还活着的 lane 申领本轮」。

    此时 `d` 已经是退休之后的值：已退休的源手里是自己的私有槽，所以它这一次 store 落在
    `tag_cur[out_numel + i]` 上，**天然不触碰任何目标**，也就不会冲刷别人的申领。
    store 地址是 load 出来的 `d`，是已验证可用的地址形式。
    """
    offs = ext.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    d = tl.load(dest_buf + offs)
    marker = (out_numel + offs).to(tl.int32)
    tl.store(tag_cur + d, marker)


@libentry()
@triton.jit(do_not_specialize=["out_numel"])
def _umipa_finish_kernel(dest_buf, tag_last, alive, out_numel, BLOCK: tl.constexpr):
    """收掉最后一轮的胜者，并按 program 统计还剩多少活跃源。

    最后一轮的胜者是在「下一轮的 round kernel」里才被摘掉的，所以必须补这一步，
    否则 alive 会恒大于 0、白跑一整批。`d < out_numel` 就是「还活着」。
    """
    pid = ext.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    d = tl.load(dest_buf + offs)
    w = tl.load(tag_last + d)
    marker = (out_numel + offs).to(tl.int32)
    d = tl.where(w == marker, marker, d)
    tl.store(dest_buf + offs, d)
    tl.store(alive + pid, tl.sum((d < out_numel).to(tl.int32), axis=0))


@libentry()
@triton.jit(do_not_specialize=["out_numel", "row"])
def _umipa_combine_kernel(
    out_ptr,
    src_ptr,
    tag,
    val_lookup,
    out_numel,
    row,
    ROUNDS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """目的地域求和：out[d] = src[d] + sum_r val_lookup[tag[r][d]]。

    `tag[r][d] >= out_numel` 才表示「第 r 轮 d 上有胜者」，此时该值就是胜者的 marker，
    `val_lookup[marker]` 直接是它的 value；否则改用 `offs` 当下标，`val_lookup[offs] == 0`
    （offs < out_numel）。关键是「无人」这件事对每个 d 用的是**互不相同**的下标，
    不会像「统一写 0 再 gather val[0]」那样制造 out_numel 路同址碰撞（见文件头 dup 实测）。
    tag 的 gather 故意不带 mask：tail lane 读到的是私有槽区，地址合法且互不相同，
    结果被最后的 store mask 丢掉。
    """
    offs = ext.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    inb = offs < out_numel
    acc = tl.load(src_ptr + offs, mask=inb, other=0.0).to(tl.float32)
    limit = out_numel.to(tl.int32)
    self_idx = offs.to(tl.int32)
    for r in tl.static_range(ROUNDS):
        tid = tl.load(tag + (r + 1) * row + offs)
        tid = tl.where(tid >= limit, tid, self_idx)
        acc += tl.load(val_lookup + tid).to(tl.float32)
    tl.store(out_ptr + offs, acc, mask=inb)


# 一批跑多少轮。Poisson(1) 重数（benchmark 里 mask_numel == out_numel）下最大重数
# 通常 4~9（torch_probe.log 实测 4/6/7/9）。但**当前 round kernel 的胜者会冲刷同目标其它
# lane 的申领**（见 `_umipa_round_kernel` 的 docstring），实测每级重数要烧掉约两轮
# ⇒ 131072 在 8 轮一批下需要**两批**（probe_stage2.log：单批 3.92 ms，整函数 8.11 ms）。
# 调大到 16 并不划算：一批的 round 成本是线性的，而两批只多一次 D2H + 一次 tag.zero_()。
# 修掉冲刷（probe_addr.py 的 r5 拆两个 kernel）之后需要的轮数才等于最大重数，此时 8 足够。
_ROUNDS_PER_BATCH = 8
# 多轮路径的四个 kernel 全部**不带** `isCloseVectorization` / `buffer_size_limit`。
# 注意：曾经以为「带上 isCloseVectorization 会掉标量路径」，实测已证伪
# （post_bench_shared_v2.log 带、v3.log 不带，131072 都是 386 ms，一模一样）——
# 真正的瓶颈是同址碰撞，见文件头。这里保留成一行常量只是为了参数型 A/B 好切臂。
_ROUND_LAUNCH_KW = {}
# match 路径的代价 ~ out_numel * mask_numel * 74 ps；多轮路径有 ~11 次 launch 的地板
# （0.014~0.018 ms/launch）。两者在 ~4e6 对上交叉。
_MULTI_ROUND_MIN_WORK = 4_000_000
# 每轮拆成 retire + claim 两个 kernel 的规模门槛（`mask_numel >=` 本值才拆）。
#
# 拆分做什么：融合版的 `tl.store(tag_cur + d, tl.where(win, d, marker))` 让**退休的胜者也往
# 目标槽写**，会冲掉同目标其它活跃 lane 的申领 ⇒ 每一级重数平均烧掉两轮。拆开之后
# retire 只写连续的 dest_buf、claim 只写 `tag_cur + d`（此时 d 已是退休后的私有槽），
# 谁也不冲谁 ⇒ **轮数 = 最大重数**，离散访存总量减半。
# 前置缺陷规避已上机验证：`probe_addr.py` r5 → `r5 split OK rounds=8 active=1004
# recorded=1004 dup=0 left=0`，无 721、无 wedge（融合版那个 `tl.where` 地址表达式会打 721）。
#
# 为什么要门槛，而不是无条件拆：`_ROUNDS_PER_BATCH` 是**固定的 8**，只有批数自适应。
# 于是拆分每批多付 `rounds` 次 launch（该后端经 gems 分派 ~45–55 us/次），
# 只有当「省下的离散访存」超过「多付的 launch」才回本：
#     省 = rounds × mask_numel × ~3.2 ns/elem（一轮一次离散 gather + 一次离散 scatter，
#          benchmark 的自然重数下实测单价，见文件头 dup 表）
#     付 = rounds × ~45 us
#   ⇒ 交叉点 mask_numel ≈ 45e-6 / 3.2e-9 ≈ **14000**，取 2 的幂 16384。
#
# 对上 benchmark 的四个 shape（`mask = randint(0, 2)` ⇒ 约一半源被 mask 掉 ⇒ 目的地重数是
# Poisson(0.5)，最大重数 N=4096 时 ~4–5、N=131072 时 ~6–7，两者都 ≤ 8 ⇒ 拆分后都是单批）：
#   `[64]` 4096 对、`[8,128]` 1.05e6 对 —— 连 `_MULTI_ROUND_MIN_WORK` 都没到，走 match 快路，
#       **本常量对它们完全惰性**（VERDICT.md §3.1 曾误判「它们也会多 8 次 launch」）。
#   `[4096]`     mask_numel=4096   < 16384 ⇒ 保持融合。融合版实测 0.61 ms ≈ 11 次 launch，
#       已经是单批；拆了只会变成 19 次 ⇒ 必须留在融合臂。
#   `[2,1024,64]` mask_numel=131072 ≥ 16384 ⇒ 拆。融合版 8.11 ms = 21 次 launch(1.16 ms)
#       + 16 轮 × 0.434 ms；拆分后 19 次 launch(1.05 ms) + 8 轮 × 0.434 ⇒ 投影 **~4.5 ms**。
#
# 正确性不依赖本常量取值：`alive` 的收敛循环在轮数不够时会自动再跑一批，
# 门槛只影响性能。设成 0 = 恒拆，设成极大值 = 恒融合（参数型 A/B 仍是一行）。
_ROUND_SPLIT_MIN_MASK_NUMEL = 16384


def _unsafe_masked_index_put_accumulate_multi_round(
    inp, mask_c, idx_c, values_c, rank, shape, strides
):
    out_numel = inp.numel()
    mask_numel = mask_c.numel()
    rounds = _ROUNDS_PER_BATCH
    round_split = mask_numel >= _ROUND_SPLIT_MIN_MASK_NUMEL

    block = max(64, min(2048, triton.next_power_of_2(mask_numel)))
    grid_m = (triton.cdiv(mask_numel, block),)
    m_pad = grid_m[0] * block
    block_n = max(64, min(2048, triton.next_power_of_2(out_numel)))
    grid_n = (triton.cdiv(out_numel, block_n),)

    # tag / val_lookup 的行长：out_numel 个目标槽 + 每个源一个私有槽；
    # 再保证 combine 那次**不带 mask** 的 tag gather（offs 最大到 grid_n*block_n-1）在界内。
    pad = max(m_pad, grid_n[0] * block_n - out_numel)
    row = out_numel + pad
    dev = inp.device

    # 每行全 0 即可：0 < out_numel 恒表示「该目标无人获胜」，且永不等于任何 marker。
    # （不能用 torch.arange(int32)，该后端会 ASSERT-FAIL 并返回垃圾，见文件头。）
    tag = torch.zeros((rounds + 1) * row, dtype=torch.int32, device=dev)

    dest_buf = torch.empty(m_pad, dtype=torch.int32, device=dev)
    # 前 out_numel 个必须恒为 0（= 该目标本轮无贡献），尾部由 prep 填入 values。
    val_lookup = torch.zeros(row, dtype=inp.dtype, device=dev)
    alive = torch.empty(grid_m[0], dtype=torch.int32, device=dev)
    out = torch.empty_like(inp)

    with torch_device_fn.device(dev):
        _umipa_prep_kernel[grid_m](
            dest_buf,
            val_lookup,
            mask_c,
            idx_c[0],
            idx_c[1],
            idx_c[2],
            values_c,
            mask_numel,
            out_numel,
            SHAPE0=shape[0],
            SHAPE1=shape[1],
            SHAPE2=shape[2],
            STRIDE0=strides[0],
            STRIDE1=strides[1],
            STRIDE2=strides[2],
            RANK=rank,
            BLOCK=block,
            **_ROUND_LAUNCH_KW,
        )
        src = inp
        for batch in range(64):
            if batch:
                tag.zero_()
            for r in range(rounds):
                if round_split:
                    _umipa_retire_kernel[grid_m](
                        dest_buf,
                        tag[r * row :],
                        out_numel,
                        BLOCK=block,
                        **_ROUND_LAUNCH_KW,
                    )
                    _umipa_claim_kernel[grid_m](
                        dest_buf,
                        tag[(r + 1) * row :],
                        out_numel,
                        BLOCK=block,
                        **_ROUND_LAUNCH_KW,
                    )
                else:
                    _umipa_round_kernel[grid_m](
                        dest_buf,
                        tag[r * row :],
                        tag[(r + 1) * row :],
                        out_numel,
                        BLOCK=block,
                        **_ROUND_LAUNCH_KW,
                    )
            _umipa_finish_kernel[grid_m](
                dest_buf,
                tag[rounds * row :],
                alive,
                out_numel,
                BLOCK=block,
                **_ROUND_LAUNCH_KW,
            )
            _umipa_combine_kernel[grid_n](
                out,
                src,
                tag,
                val_lookup,
                out_numel,
                row,
                ROUNDS=rounds,
                BLOCK=block_n,
                **_ROUND_LAUNCH_KW,
            )
            src = out
            # 唯一的 device->host 同步：只在一批 rounds 结束后读一次 alive。
            # 典型输入（最大重数 <= 8）一次就退出，不产生第二次同步。
            if int(alive.sum().item()) == 0:
                break
        else:
            raise RuntimeError(
                "Kunlunxin _unsafe_masked_index_put_accumulate did not converge"
            )
    return out


def _unsafe_masked_index_put_accumulate(input, mask, indices, values):
    logger.debug("GEMS_KUNLUNXIN _UNSAFE_MASKED_INDEX_PUT_ACCUMULATE")
    rank = input.ndim
    if rank < 1 or rank > 3 or len(indices) != rank:
        raise RuntimeError(
            "Kunlunxin _unsafe_masked_index_put_accumulate supports ranks 1 to 3"
        )
    # aten::_unsafe_masked_index_put_accumulate 的 schema 是函数式（self 不可变），
    # 参考实现是 `x.clone()` 后 index_put_(accumulate=True)，所以必须返回新张量。
    if input.numel() == 0 or mask.numel() == 0:
        return input.clone()

    inp = input if input.is_contiguous() else input.contiguous()
    mask_contiguous = mask.contiguous()
    values_contiguous = values.contiguous()
    contiguous_indices = [index.contiguous() for index in indices]
    while len(contiguous_indices) < 3:
        contiguous_indices.append(contiguous_indices[0])

    shape = list(inp.shape) + [1] * (3 - rank)
    strides = list(inp.stride()) + [0] * (3 - rank)
    out_numel = inp.numel()

    # 规模分流：小规模下 O(N*M) 的 match kernel 只要一次 launch，比多轮路径的
    # ~13 次 launch 地板便宜；大规模下 match 是结构性不可达的（131072 实测 1253 ms）。
    if out_numel * mask.numel() >= _MULTI_ROUND_MIN_WORK:
        return _unsafe_masked_index_put_accumulate_multi_round(
            inp,
            mask_contiguous,
            contiguous_indices,
            values_contiguous,
            rank,
            shape,
            strides,
        ).view(inp.shape)

    dests = _dests_per_program(out_numel)
    grid = (triton.cdiv(out_numel, dests),)
    # 尾部哨兵：grid * dests 可能大于 out_numel，多出来的 lane 写进填充区而不是被 mask 掉。
    out_buf = torch.empty(grid[0] * dests, dtype=inp.dtype, device=inp.device)
    out = out_buf[:out_numel].view(inp.shape)
    block_size = triton.next_power_of_2(mask.numel())

    with torch_device_fn.device(input.device):
        _unsafe_masked_index_put_accumulate_kernel[grid](
            out_buf,
            inp,
            mask_contiguous,
            contiguous_indices[0],
            contiguous_indices[1],
            contiguous_indices[2],
            values_contiguous,
            mask.numel(),
            out_numel,
            SHAPE0=shape[0],
            SHAPE1=shape[1],
            SHAPE2=shape[2],
            STRIDE0=strides[0],
            STRIDE1=strides[1],
            STRIDE2=strides[2],
            RANK=rank,
            DESTS=dests,
            BLOCK_SIZE=block_size,
            isCloseVectorization=True,
            buffer_size_limit=2048,
        )
    return out
