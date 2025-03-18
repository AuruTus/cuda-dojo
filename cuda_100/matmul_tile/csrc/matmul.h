#ifndef MATMUL_TILE_H
#define MATMUL_TILE_H

#include <torch/all.h>

namespace matmul_tile {
	auto matmul_tile_cuda(const at::Tensor& A, const at::Tensor& B) -> at::Tensor;
}

#endif