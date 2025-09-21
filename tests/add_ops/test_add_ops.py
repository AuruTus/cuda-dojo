import torch
from toy_kernels.add_ops import add_ops
import unittest

backend = add_ops.backend

TEST_TIME = 100
device = torch.device("cuda")


class TestAddOps(unittest.TestCase):
    def test_add(self):
        for i in range(TEST_TIME):
            shape = torch.randint(1, 10, (3,), device=device).tolist()
            a = torch.randn(shape, dtype=torch.float32, device=device)
            b = torch.randn(shape, dtype=torch.float32, device=device)
            c_cuda = backend.add_cuda(a, b)
            c_cpu = backend.add_cpu(a.cpu(), b.cpu())
            self.assertTrue((c_cpu == c_cuda.cpu()).all().item())


if __name__ == "__main__":
    unittest.main()
