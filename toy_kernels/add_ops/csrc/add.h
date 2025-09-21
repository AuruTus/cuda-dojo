#ifndef ADD_H
#define ADD_H

#include <torch/all.h>
namespace add_ops {
    auto add_cpu(const at::Tensor& a, const at::Tensor& b) -> at::Tensor;
    auto add_cuda(const at::Tensor& a, const at::Tensor& b) -> at::Tensor;
}

#endif
