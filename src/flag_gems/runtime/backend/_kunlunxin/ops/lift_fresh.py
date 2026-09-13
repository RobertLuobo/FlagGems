import logging

import torch

logger = logging.getLogger("flag_gems").getChild(__name__.lstrip("."))


def lift_fresh(x: torch.Tensor):
    # ATen 语义核对（CPU 探针）：`torch.ops.aten.lift_fresh(x)` 在 eager 下直接返回输入本身
    # （`y is x` 为 True，data_ptr 相同、零拷贝，延迟恒定 ~3us 与形状无关，O(1) 元数据操作）。
    # 该 op 仅用于把自动求导图中的视图"提升"为独立张量，本身不搬运任何数据，
    # 因此无需 Triton 内核：拷贝内核会引入 GB 级数据搬移（大 shape speedup 低至 0.001-0.2）
    # 且破坏"与输入共享存储"的语义。返回 x 即与 ATen eager 行为完全一致。
    logger.debug("GEMS_KUNLUNXIN LIFT_FRESH")
    return x
