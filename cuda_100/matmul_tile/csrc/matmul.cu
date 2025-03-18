#include <ATen/ATen.h>
#include <ATen/Operators.h>
#include <Python.h>
#include <torch/all.h>
#include <torch/library.h>

#include <cuda.h>
#include <cuda_runtime.h>


namespace matmul_tile {

	constexpr int TILE_WIDTH = 2;

#define ROUND_UP(dividend, divisor) ((int)((dividend + divisor - 1) / divisor))
#define DEBUG_SCOPE(BODY) \
	if constexpr(DEBUG_KERNEL) { BODY; }

	template<const bool DEBUG_KERNEL = false, typename T = float>
	__global__ void matmul_cuda_kernel(
		const T* A,
		const T* B,
		T* C,
		const int row_len,
		const int col_len,
		const int width
	) {

		__shared__ T Asd[TILE_WIDTH][TILE_WIDTH];
		__shared__ T Bsd[TILE_WIDTH][TILE_WIDTH];

		auto bx = blockIdx.x;
		auto by = blockIdx.y;
		auto tx = threadIdx.x;
		auto ty = threadIdx.y;

		// global row and col
		auto row = by * TILE_WIDTH + ty;
		auto col = bx * TILE_WIDTH + tx;

		DEBUG_SCOPE(
			if (tx == 0 && ty == 0) {
				printf("row_len: %d, col_len: %d, width: %d\n", row_len, col_len, width);
			};
		);

		T c_value = 0;

		auto phases = ROUND_UP(width, TILE_WIDTH);
		for (auto ph = 0; ph < phases; ++ph) {
			Asd[ty][tx] = row < row_len && (ph * TILE_WIDTH + tx) < width ?
				A[row * width + ph * TILE_WIDTH + tx] : 0.0f;
			Bsd[ty][tx] = col < col_len && (ph * TILE_WIDTH + ty) < width ?
				B[(ph * TILE_WIDTH + ty) * col_len + col] : 0.0f;
			DEBUG_SCOPE(
				printf(
					"bx: %d, by: %d, tx: %d, ty: %d, col: %d, row: %d, phase: %d, Acol: %d, Brow: %f\n",
					bx, by, tx, ty, col, row, ph, ph * TILE_WIDTH + ty, ph * TILE_WIDTH + tx
				);
			);

			__syncthreads();

			for (auto k = 0; k < TILE_WIDTH; ++k) {
				DEBUG_SCOPE(
					printf(
						"bx: %d, by: %d, tx: %d, ty: %d, col: %d, row: %d, phase: %d, k: %d, Ads: %f, Bds: %f\n",
						bx, by, tx, ty, col, row, ph, k, Asd[ty][k], Bsd[k][tx]
					);
					);
				c_value += Asd[ty][k] * Bsd[k][tx];
			}

			__syncthreads();
		}

		if (row < row_len && col < col_len) {
			DEBUG_SCOPE(
				printf("tx: %d, ty: %d, cid: %d\n", tx, ty, row * col_len + col);
				);
			C[row * col_len + col] = c_value;
		}
	}


	auto matmul_tile_cuda(const at::Tensor& A, const at::Tensor& B) -> at::Tensor {
		TORCH_CHECK(A.is_cuda());
		TORCH_CHECK(B.is_cuda());
		TORCH_CHECK(A.is_floating_point());
		TORCH_CHECK(B.is_floating_point());
		TORCH_CHECK(A.size(1) == B.size(0));
		auto A_contig = A.contiguous();
		auto B_contig = B.contiguous();
		auto C = torch::empty(at::makeArrayRef({ A.size(0), B.size(1) }), A_contig.options());

		auto row_len = A.size(0);
		auto col_len = B.size(1);
		auto width = A.size(1);

		auto grid_dim = dim3(ROUND_UP(row_len, TILE_WIDTH), ROUND_UP(col_len, TILE_WIDTH));
		auto block_dim = dim3(TILE_WIDTH, TILE_WIDTH);
		matmul_cuda_kernel<false> << <grid_dim, block_dim >> > (
			A_contig.data_ptr<float>(),
			B_contig.data_ptr<float>(),
			C.data_ptr<float>(),
			row_len,
			col_len,
			width
			);
		return C;
	}
}
