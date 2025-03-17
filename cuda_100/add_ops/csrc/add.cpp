// #include "ATen/ops/empty.h"

#include <ATen/Operators.h>
#include <Python.h>
#include <torch/all.h>
#include <torch/library.h>


namespace add_ops {

    auto add_cpu(const at::Tensor& a, const at::Tensor& b) -> at::Tensor {
        TORCH_CHECK(a.is_cpu());
        TORCH_CHECK(b.is_cpu());
        TORCH_CHECK(a.dtype() == at::kFloat);
        TORCH_CHECK(b.dtype() == at::kFloat);
        TORCH_CHECK(a.sizes() == b.sizes());

        auto a_contig = a.contiguous();
        auto b_contig = b.contiguous();

        auto c = torch::empty(
            a_contig.sizes(),
            a_contig.options()
        );

        auto a_ptr = a_contig.data_ptr<float>();
        auto b_ptr = b_contig.data_ptr<float>();
        auto c_ptr = c.data_ptr<float>();
        for (int i = 0; i < a.numel(); ++i) {
            c_ptr[i] = a_ptr[i] + b_ptr[i];
        }

        return c;
    }



    TORCH_LIBRARY_IMPL(TORCH_EXTENSION_NAME, CPU, m) {
        m.impl("add", &add_cpu);
    }

} // namespace add_ops
