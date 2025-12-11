import torch
import pytest
from toy_kernels.reduce_1d_sum import backend

device = torch.device("cuda")


def test_shfl():
    x = torch.tensor([1, 2, 3, 4, 5], device=device, dtype=torch.float32)
    print(backend.test_shfl_down_sync(x))
