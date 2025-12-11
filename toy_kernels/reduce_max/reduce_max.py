import os
import torch
from torch.utils import cpp_extension
from toy_kernels.utils import get_src_files, get_cxx_flags, get_nvcc_flags

__all__ = ["backend"]

CSRC_DIR = os.path.join(os.path.dirname(__file__), "csrc")

backend = cpp_extension.load(
    "reduce_max",
    get_src_files(CSRC_DIR, "*.cpp") + get_src_files(CSRC_DIR, "*.cu"),
    extra_cflags=get_cxx_flags(),
    extra_cuda_cflags=get_nvcc_flags(),
)


def reduce_max(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    if dim < 0:
        dim = x.dim() + dim
    return backend.reduce_max(x, dim)
