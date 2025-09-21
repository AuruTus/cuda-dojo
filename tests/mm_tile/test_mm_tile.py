import torch
import unittest
from dataclasses import dataclass
from toy_kernels.matmul_tile import backend

device = torch.device("cuda")


class TestMatMulTile(unittest.TestCase):
    def test_add(self):
        @dataclass
        class _TestCase:
            a: torch.Tensor
            b: torch.Tensor

        cases = [
            _TestCase(
                a=torch.ones(2, 3).to(device=device, dtype=torch.float32),
                b=torch.ones(3, 2).to(device=device, dtype=torch.float32),
            )
        ]

        for case in cases:
            c_exp = case.a @ case.b
            c = backend.matmul_tile(case.a, case.b)
            print(c_exp)
            print(c)


if __name__ == "__main__":
    unittest.main()
