import torch
import pytest
from toy_kernels.reduce_max.reduce_max import reduce_max


@pytest.mark.parametrize(
    ("shape", "dim"),
    [
        ([2, 3, 4], 1),
        ([2, 3, 4], -1),
        ([2, 3, 4], 2),
        ([5, 6, 7, 8], 2),
        ([10, 10, 10, 10, 10], 1),
        ([10, 10, 10, 10, 10], 2),
        ([10, 10, 10, 10, 10], 3),
        ([10, 10, 10, 10, 10], -1),
    ],
)
def test_reduce_max(shape: list[int], dim: int):
    x = torch.randn(*shape, device="cuda", dtype=torch.float32)
    result = reduce_max(x, dim=dim)
    expected = torch.max(x, dim=dim).values
    assert torch.allclose(result, expected), f"x: {x} \n result: {result}, \n expected: {expected}"
