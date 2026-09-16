import torch


def _linalg_eigvals(inp):
    handle = torch.ops.aten._linalg_eigvals.default._handle
    keyset = torch._C.DispatchKeySet(torch._C.DispatchKey.XPU)
    return handle.redispatch_boxed(keyset, inp)
