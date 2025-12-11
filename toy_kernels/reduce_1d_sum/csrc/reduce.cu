#include <ATen/ATen.h>
#include <ATen/Operators.h>
#include <Python.h>
#include <torch/all.h>
#include <torch/library.h>

#include <cuda.h>

#include <cuda_runtime.h>
#include <stdio.h>


namespace reduce {
    constexpr int WARP_SIZE_LIT = 32;

    template<const int WARP_SIZE = WARP_SIZE_LIT>
    __device__ __forceinline__ float warpReduceSum(float val) {
        for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
            val += __shfl_down_sync(0xffffffff, val, offset);
        }
        return val;
    }


    template<const int WARP_SIZE = WARP_SIZE_LIT>
    __global__ void test_shfl_down_sync_f16_kernel(const float* __restrict__ A, float* __restrict__ B, int64_t N) {
        float sum = 0.0f;
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        int stride = blockDim.x * gridDim.x;

        for (int i = idx; i < N; i += stride) {
            sum += A[i];
        }

        sum = warpReduceSum<WARP_SIZE>(sum);

        if ((threadIdx.x & (WARP_SIZE - 1)) == 0) {
            atomicAdd(B, sum);
        }

    }

    at::Tensor reduce_cuda(const at::Tensor& A, const at::Tensor& B) {
        // TODO
        return A;
    }


    at::Tensor test_shfl_down_sync(const at::Tensor& A) {
        auto B = at::zeros({ 1 }, A.options());
        const int threads = 256;
        const int blocks = (A.numel() + threads - 1) / threads;
        AT_DISPATCH_FLOATING_TYPES(A.scalar_type(), "test_shfl_down_sync", ([&] {
            test_shfl_down_sync_f16_kernel << <blocks, threads >> > (
                A.data_ptr<float>(), B.data_ptr<float>(), A.numel());
                                                                            }));
        return B;
    }
}