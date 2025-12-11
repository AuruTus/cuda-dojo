#include <torch/all.h>
#include <cuda_runtime.h>


constexpr int THREADS_PER_BLOCK = 256;

template<typename T>
__host__ __device__ __forceinline__ auto div_ciel(T divident, T divisor) -> T {
    return (divident + divisor - 1) / divisor;
}

__global__ void reduce_max_kernel(
    const float* __restrict__ A,
    float* __restrict__ B,
    int dim_size,
    int inner_size,
    int outer_size
) {
    auto inner_thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (inner_thread_id >= inner_size) {
        return;
    }
    float max_val = -FLT_MAX;
    const float* A_ptr = A + blockIdx.y * dim_size * inner_size + inner_thread_id;
    for (int dim_id = 0; dim_id < dim_size; ++dim_id) {
        float val = *A_ptr;
        if (val > max_val) {
            max_val = val;
        }
        A_ptr += inner_size;
    }
    int out_index = blockIdx.y * inner_size + inner_thread_id;
    B[out_index] = max_val;
}

namespace reduce_max {

    auto reduce_max_cuda(const at::Tensor& A, int dim) -> at::Tensor {
        auto shape = A.sizes().vec();
        shape.erase(shape.begin() + dim);
        auto B = at::empty(shape, A.options());

        auto inner_size = 1;
        for (int i = dim; i < shape.size(); ++i) {
            inner_size *= shape[i];
        }
        auto dim_size = A.size(dim);
        auto outer_size = A.numel() / (dim_size * inner_size);
        // printf("inner_size: %d, dim_size: %d, outer_size: %d\n", inner_size, dim_size, outer_size);

        auto num_blocks = static_cast<size_t>(div_ciel(inner_size, THREADS_PER_BLOCK));
        // printf("num_blocks: %d\n", num_blocks);
        dim3 grid_size{ num_blocks, outer_size, 1 };

        // printf("grid_size: (%d, %d, %d)\n", grid_size.x, grid_size.y, grid_size.z);

        reduce_max_kernel << <grid_size, THREADS_PER_BLOCK >> > (
            A.data_ptr<float>(),
            B.data_ptr<float>(),
            dim_size,
            inner_size,
            outer_size
            );

        return B;
    }

}