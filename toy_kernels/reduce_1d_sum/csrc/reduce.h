#ifndef REDUCE_H
#define REDUCE_H

#include <torch/all.h>

namespace reduce {
    auto reduce_cuda(const at::Tensor& A, const at::Tensor& B) -> at::Tensor;
    auto test_shfl_down_sync(const at::Tensor& A) -> at::Tensor;
}

#endif