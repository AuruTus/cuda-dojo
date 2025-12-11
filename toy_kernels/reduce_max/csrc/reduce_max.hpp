#ifndef REDUCE_MAX_H
#define REDUCE_MAX_H

#include <torch/all.h>

namespace reduce_max {
    auto reduce_max_cuda(const at::Tensor& A, int dim) -> at::Tensor;
}

#endif
