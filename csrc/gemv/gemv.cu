#include <cuda_runtime.h>
#include <cuda_runtime.h>
#include <iostream>
#include <chrono>
#include <vector>
#include <cassert>


// CPU reference implementation
void gemv_cpu(
    const float* input_mat,
    const float* input_vec,
    float* output,
    const int N,
    const int V
) {
    for (int row = 0; row < N; row++) {
        float sum = 0.0f;
        for (int col = 0; col < V; col++) {
            sum += input_mat[row * V + col] * input_vec[col];
        }
        output[row] = sum;
    }
}

// Compare two arrays with tolerance
bool compare_arrays(
    const float* arr1,
    const float* arr2,
    int size,
    float tolerance = 1e-4f,
    bool verbose = false
) {
    bool match = true;
    float max_error = 0.0f;
    int max_error_idx = -1;

    for (int i = 0; i < size; i++) {
        float error = std::abs(arr1[i] - arr2[i]);
        if (error > max_error) {
            max_error = error;
            max_error_idx = i;
        }
        if (error > tolerance) {
            if (verbose) {
                std::cout << "Mismatch at index " << i
                    << ": CPU=" << arr1[i]
                    << ", GPU=" << arr2[i]
                    << ", error=" << error << std::endl;
            }
            match = false;
        }
    }

    if (!match) {
        std::cout << "Maximum error: " << max_error
            << " at index " << max_error_idx << std::endl;
    }

    return match;
}

constexpr size_t ARR_LEN = 256;
__constant__ float CONST_VECTOR[ARR_LEN];


__global__ void
gemv_constant(
    float* __restrict__ input,
    float* __restrict__ output,
    const int N, const int V
) {
    __shared__ float warp_gather[32];
    auto row = blockIdx.x;
    auto lane_idx = threadIdx.x % 32;
    auto warp_idx = threadIdx.x / 32;

    float pvalue = 0.0f;
    for (auto i = threadIdx.x; i < V; i += blockDim.x) {
        pvalue += input[row * V + i] * CONST_VECTOR[i];
    }
    for (auto offset = 16; offset > 0; offset >>= 1) {
        pvalue += __shfl_down_sync(0xffffffff, pvalue, offset);
    }
    if (lane_idx == 0) {
        warp_gather[warp_idx] = pvalue;
    }
    __syncthreads();

    if (warp_idx == 0) {
        pvalue = (lane_idx < 32) ? warp_gather[lane_idx] : 0.0f;
        for (auto offset = 16; offset > 0; offset >>= 1) {
            pvalue += __shfl_down_sync(0xffffffff, pvalue, offset);
        }

        if (lane_idx == 0) {
            output[row] = pvalue;
        }
    }
}


__global__ void
gemv_shmem(
    float* __restrict__ input_mat,
    float* __restrict__ input_vec,
    float* __restrict__ output,
    const int N, const int V
) {
    __shared__ float warp_gather[32];
    auto row = blockIdx.x;
    auto lane_idx = threadIdx.x % 32;
    auto warp_idx = threadIdx.x / 32;

    float pvalue = 0.0f;
    for (auto i = threadIdx.x; i < V; i += blockDim.x) {
        pvalue += input_mat[row * V + i] * input_vec[i];
    }
    for (auto offset = 16; offset > 0; offset >>= 1) {
        pvalue += __shfl_down_sync(0xffffffff, pvalue, offset);
    }
    if (lane_idx == 0) {
        warp_gather[warp_idx] = pvalue;
    }
    __syncthreads();

    if (warp_idx == 0) {
        pvalue = (lane_idx < 32) ? warp_gather[lane_idx] : 0.0f;
        for (auto offset = 16; offset > 0; offset >>= 1) {
            pvalue += __shfl_down_sync(0xffffffff, pvalue, offset);
        }

        if (lane_idx == 0) {
            output[row] = pvalue;
        }
    }
}


// Benchmarking function
void benchmark_gemv(int N, int V, int iterations = 1000) {
    // Validate parameters
    assert(V <= ARR_LEN);

    // Allocate host memory
    std::vector<float> h_input_mat(N * V);
    std::vector<float> h_input_vec(V);
    std::vector<float> h_output_constant(N);
    std::vector<float> h_output_shmem(N);

    // Initialize with test data
    for (int i = 0; i < N * V; i++) {
        h_input_mat[i] = static_cast<float>(i % 100) * 0.01f;
    }
    for (int i = 0; i < V; i++) {
        h_input_vec[i] = static_cast<float>(i % 50) * 0.02f;
    }

    // Allocate device memory
    float* d_input_mat, * d_input_vec, * d_output_constant, * d_output_shmem;
    cudaMalloc(&d_input_mat, N * V * sizeof(float));
    cudaMalloc(&d_input_vec, V * sizeof(float));
    cudaMalloc(&d_output_constant, N * sizeof(float));
    cudaMalloc(&d_output_shmem, N * sizeof(float));

    // Copy data to device
    cudaMemcpy(d_input_mat, h_input_mat.data(), N * V * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_input_vec, h_input_vec.data(), V * sizeof(float), cudaMemcpyHostToDevice);

    // Copy vector to constant memory
    cudaMemcpyToSymbol(CONST_VECTOR, h_input_vec.data(), V * sizeof(float));

    // Choose block size (multiple of 32 for warp operations)
    const int block_size = 256;
    const int grid_size = N;

    // Create CUDA events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    float constant_time = 0.0f, shmem_time = 0.0f;

    // Warm-up runs
    for (int i = 0; i < 10; i++) {
        gemv_constant << <grid_size, block_size >> > (d_input_mat, d_output_constant, N, V);
        gemv_shmem << <grid_size, block_size >> > (d_input_mat, d_input_vec, d_output_shmem, N, V);
    }
    cudaDeviceSynchronize();

    // Benchmark constant memory version
    cudaEventRecord(start);
    for (int i = 0; i < iterations; i++) {
        gemv_constant << <grid_size, block_size >> > (d_input_mat, d_output_constant, N, V);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&constant_time, start, stop);

    // Benchmark shared memory version  
    cudaEventRecord(start);
    for (int i = 0; i < iterations; i++) {
        gemv_shmem << <grid_size, block_size >> > (d_input_mat, d_input_vec, d_output_shmem, N, V);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&shmem_time, start, stop);

    // Copy results back for verification
    cudaMemcpy(h_output_constant.data(), d_output_constant, N * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_output_shmem.data(), d_output_shmem, N * sizeof(float), cudaMemcpyDeviceToHost);

    // Verify results match (within tolerance)
    bool results_match = true;
    float tolerance = 1e-4f;
    for (int i = 0; i < N; i++) {
        if (std::abs(h_output_constant[i] - h_output_shmem[i]) > tolerance) {
            results_match = false;
            break;
        }
    }

    // Print results
    std::cout << "=== GEMV Benchmark Results ===" << std::endl;
    std::cout << "Matrix: " << N << " x " << V << std::endl;
    std::cout << "Iterations: " << iterations << std::endl;
    std::cout << "Block size: " << block_size << std::endl;
    std::cout << std::endl;
    std::cout << "Constant Memory: " << constant_time << " ms" << std::endl;
    std::cout << "Shared Memory:   " << shmem_time << " ms" << std::endl;
    std::cout << std::endl;
    std::cout << "Results match: " << (results_match ? "YES" : "NO") << std::endl;

    if (results_match) {
        std::cout << "Performance ratio (shared/constant): "
            << (shmem_time / constant_time) << std::endl;
    }

    // Calculate GFLOPs
    double flops_per_iteration = 2.0 * N * V; // N*V multiplications + N*V additions
    double total_flops = flops_per_iteration * iterations;

    std::cout << std::endl;
    std::cout << "Constant Memory Throughput: "
        << (total_flops / (constant_time * 1e6)) << " GFLOP/s" << std::endl;
    std::cout << "Shared Memory Throughput:   "
        << (total_flops / (shmem_time * 1e6)) << " GFLOP/s" << std::endl;

    // Cleanup
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_input_mat);
    cudaFree(d_input_vec);
    cudaFree(d_output_constant);
    cudaFree(d_output_shmem);
}

void validate_gemv(
    const int N, const int V
) {
    std::cout << "assert the GEMV implements" << std::endl;

    std::vector<float> h_input_mat(N * V);
    std::vector<float> h_input_vec(V);
    std::vector<float> h_output_cpu(N);
    std::vector<float> h_output_constant(N);
    std::vector<float> h_output_shmem(N);

    for (int i = 0; i < N * V; i++) {
        h_input_mat[i] = static_cast<float>(i % 100) * 0.01f;
    }
    for (int i = 0; i < V; i++) {
        h_input_vec[i] = static_cast<float>(i % 50) * 0.02f;
    }

    float* d_input_mat;
    float* d_input_vec;
    float* d_output_constant;
    float* d_output_shmem;
    cudaMalloc(&d_input_mat, N * V * sizeof(float));
    cudaMalloc(&d_input_vec, V * sizeof(float));
    cudaMalloc(&d_output_constant, N * sizeof(float));
    cudaMalloc(&d_output_shmem, N * sizeof(float));

    cudaMemcpy(d_input_mat, h_input_mat.data(), N * V * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_input_vec, h_input_vec.data(), V * sizeof(float), cudaMemcpyHostToDevice);


    const int block_size = 256;
    const int grid_size = N;

    cudaMemcpyToSymbol(CONST_VECTOR, h_input_vec.data(), V * sizeof(float));
    gemv_constant << <grid_size, block_size >> > (
        d_input_mat, d_output_constant, N, V
        );
    gemv_shmem << <grid_size, block_size >> > (
        d_input_mat, d_input_vec, d_output_shmem, N, V
        );

    gemv_cpu(
        h_input_mat.data(), h_input_vec.data(), h_output_cpu.data(), N, V
    );

    cudaMemcpy(h_output_constant.data(), d_output_constant, N * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_output_shmem.data(), d_output_shmem, N * sizeof(float), cudaMemcpyDeviceToHost);

    constexpr float tolerance = 1e-4f;
    for (auto i = 0; i < N; ++i) {
        // std::cout << "cpu: " << h_output_cpu[i]
        //     << ", shmem: " << h_output_shmem[i]
        //     << ", constant: " << h_output_constant[i] << std::endl;
        assert(std::abs(h_output_cpu[i] - h_output_shmem[i]) <= tolerance);
        assert(std::abs(h_output_cpu[i] - h_output_constant[i]) <= tolerance);
    }

    std::cout << "assertion for "
        << "N: " << N
        << ", V: " << V
        << ", GEMV passed " << std::endl;

}

int main() {
    // Test with different sizes
    std::cout << "Testing GEMV kernels..." << std::endl;

    // Small vector (good for constant memory)
    std::cout << "\n--- Small Vector (V=64) ---" << std::endl;
    validate_gemv(1000, 64);
    benchmark_gemv(1000, 64, 1000);

    // Medium vector
    std::cout << "\n--- Medium Vector (V=128) ---" << std::endl;
    validate_gemv(1000, 128);
    benchmark_gemv(1000, 128, 1000);

    // Large vector (approaching constant memory limits)
    std::cout << "\n--- Large Vector (V=256) ---" << std::endl;
    validate_gemv(1000, 256);
    benchmark_gemv(1000, 256, 1000);

    return 0;
}
