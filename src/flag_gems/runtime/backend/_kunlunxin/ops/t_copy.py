import logging

import torch

import flag_gems

logger = logging.getLogger("flag_gems").getChild(__name__.lstrip("."))


def _launch_t_copy_kernel(inp: torch.Tensor, out: torch.Tensor):
    """Transpose-and-copy delivered by the vendor native strided-copy engine.

    `t_copy` is a pure data-movement op (out[i, j] = inp[j, i] for 2D input, a
    plain copy for 0-D/1-D input).  We express it as a view of `inp`
    (`transpose(0, 1)` / the tensor itself) + `torch.ops.aten._copy_from` into
    the pre-allocated `out`.  Gems never registers `_copy_from`, so the call
    reaches the vendor native strided-copy kernel (RegisterCUDA.cpp) instead of
    a Triton transpose kernel: on XPU a Triton transpose has one inherently
    discrete (stride != 1) side and measured ~8.7ms for 4096^2, while the
    native strided-copy engine does the same transpose in ~0.05-0.1ms
    (~100x+).  Same pattern as the accepted `sum_dim._compress` /
    `slice_backward` / `resize` / `constant_pad_nd` / `block_diag` fixes
    ("同一把钥匙"); not a CPU/native-composite fallback -- the copy executes
    on-device in the vendor engine and the output is a device tensor.
    """
    if inp.device.type != flag_gems.device or out.device.type != flag_gems.device:
        raise ValueError(f"t_copy kernels require {flag_gems.device} tensors")
    assert inp.dtype == out.dtype, "dtype mismatch between input and output"

    dim = inp.dim()
    if dim > 2:
        raise RuntimeError("t_copy expects a tensor with <= 2 dims")
    if inp.numel() == 0:
        return

    if dim == 2:
        M, N = inp.shape  # input dims
        # out should be (N, M)
        assert (
            out.dim() == 2 and out.shape[0] == N and out.shape[1] == M
        ), "Output shape must be (input.size(1), input.size(0)) for t_copy"
        src = inp.transpose(0, 1)  # shape (N, M), arbitrary strides
    else:
        # 0-D / 1-D t_copy is an identity copy.
        assert out.numel() == inp.numel(), "Output size mismatch for t_copy"
        src = inp

    torch.ops.aten._copy_from(src, out, False)


def t_copy_out(
    input: torch.Tensor,
    out: torch.Tensor,
    memory_format: torch.memory_format | None = None,
):
    logger.debug("GEMS_KUNLUNXIN T_COPY_OUT")
    _launch_t_copy_kernel(input, out)
    return out


def t_copy(input: torch.Tensor, memory_format: torch.memory_format | None = None):
    logger.debug("GEMS_KUNLUNXIN T_COPY")
    dim = input.dim()
    if dim == 0:
        out = torch.empty((), dtype=input.dtype, device=input.device)
    elif dim == 1:
        out = torch.empty_like(input, memory_format=torch.contiguous_format)
    elif dim == 2:
        M, N = input.shape
        out = torch.empty(
            (N, M),
            dtype=input.dtype,
            device=input.device,
            memory_format=torch.contiguous_format,
        )
    else:
        raise RuntimeError("t_copy expects a tensor with <= 2 dims")
    _launch_t_copy_kernel(input, out)
    return out
