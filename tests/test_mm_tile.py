import torch
import pytest
from toy_kernels.matmul_tile import backend

device = torch.device("cuda")


@pytest.mark.parametrize(
    ("a", "b"),
    [
        (torch.ones(2, 3).to(device=device, dtype=torch.float32), torch.ones(3, 2).to(device=device, dtype=torch.float32)),
    ],
)
def test_mm_tile(a: torch.Tensor, b: torch.Tensor):
    c_exp = a @ b
    c = backend.matmul_tile(a, b)
    print(c_exp)
    print(c)


if __name__ == "__main__":
    pytest.main([__file__, "-s", "-v"])
