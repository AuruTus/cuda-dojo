#ifndef ADD_H
#define ADD_H

#include <torch/all.h>
namespace day_01 {
    auto add_cpu(const at::Tensor& a, const at::Tensor& b) -> at::Tensor;
    auto add_cuda(const at::Tensor& a, const at::Tensor& b) -> at::Tensor;
}

#endif
