#include <ATen/ATen.h>
#include <ATen/Operators.h>
#include <Python.h>
#include <torch/all.h>
#include <torch/library.h>

#include <cuda.h>
#include <cuda_runtime.h>


namespace add_ops {

    template<typename T>
    __global__ void _add_cuda_kernel(
        const T* a,
        const T* b,
        T* c,
        const int64_t numel
    ) {
        auto idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < numel) {
            c[idx] = a[idx] + b[idx];
        }
    }


    auto add_cuda(const at::Tensor& a, const at::Tensor& b) -> at::Tensor {
        TORCH_CHECK(a.sizes() == b.sizes());
        TORCH_CHECK(a.is_cuda());
        TORCH_CHECK(b.is_cuda());
        TORCH_CHECK(a.dtype() == at::kFloat);
        TORCH_CHECK(b.dtype() == at::kFloat);

        auto a_contig = a.contiguous();
        auto b_contig = b.contiguous();
        auto c = torch::empty_like(a_contig, a_contig.options());

        auto a_ptr = a_contig.data_ptr<float>();
        auto b_ptr = b_contig.data_ptr<float>();
        auto c_ptr = c.data_ptr<float>();

        auto numel = a_contig.numel();
        _add_cuda_kernel << <(numel + 31) / 32, 32 >> > (a_ptr, b_ptr, c_ptr, numel);
        return c;
    }

    TORCH_LIBRARY_IMPL(TORCH_EXTENSION_NAME, CUDA, m) {
        m.impl("add", &add_cuda);
    }
}
