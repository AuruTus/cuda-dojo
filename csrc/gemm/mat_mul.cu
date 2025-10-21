#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <cassert>

constexpr size_t TILE_WIDTH = 16;

__global__ void matrix_multiplication_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K
) {
    extern __shared__ float smem[];

    auto* smem_a = smem;
    auto* smem_b = smem + TILE_WIDTH * TILE_WIDTH;

    auto col = blockIdx.x * TILE_WIDTH;
    auto row = blockIdx.y * TILE_WIDTH;
    auto tile_row = threadIdx.y * TILE_WIDTH;


    auto pvalue = 0.0f;
    size_t N_phase = (N + TILE_WIDTH - 1) / TILE_WIDTH;
    for (auto ph = 0; ph < N_phase; ++ph) {
        if (row + threadIdx.y < M && ph * TILE_WIDTH + threadIdx.x < N) {
            smem_a[tile_row + threadIdx.x] = A[(row + threadIdx.y) * N + ph * TILE_WIDTH + threadIdx.x];
        } else {
            smem_a[tile_row + threadIdx.x] = 0.0f;
        }
        if (ph * TILE_WIDTH + threadIdx.y < N && col + threadIdx.x < K) {
            smem_b[tile_row + threadIdx.x] = B[(ph * TILE_WIDTH + threadIdx.y) * K + col + threadIdx.x];
        } else {
            smem_b[tile_row + threadIdx.x] = 0.0f;
        }
        __syncthreads();

        for (auto i = 0; i < TILE_WIDTH; ++i) {
            pvalue += smem_a[tile_row + i] * smem_b[i * TILE_WIDTH + threadIdx.x];
        }
        __syncthreads();
    }

    if (row + threadIdx.y < M && col + threadIdx.x < K) {
        C[(row + threadIdx.y) * K + col + threadIdx.x] = pvalue;
    }
}


// A, B, C are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* A, const float* B, float* C, int M, int N, int K) {
    dim3 threadsPerBlock(TILE_WIDTH, TILE_WIDTH);
    dim3 blocksPerGrid(
        (K + threadsPerBlock.x - 1) / threadsPerBlock.x,
        (M + threadsPerBlock.y - 1) / threadsPerBlock.y
    );
    const auto smem_size = 2 * TILE_WIDTH * TILE_WIDTH * sizeof(float);

    matrix_multiplication_kernel << <blocksPerGrid, threadsPerBlock, smem_size >> > (
        A, B, C, M, N, K
        );
    cudaDeviceSynchronize();
}


// CPU reference implementation for verification
void matrix_multiply_cpu(const std::vector<float>& A, const std::vector<float>& B,
                         std::vector<float>& C, int M, int N, int K) {
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < K; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < N; ++k) {
                sum += A[i * N + k] * B[k * K + j];
            }
            C[i * K + j] = sum;
        }
    }
}


// Generate random matrix
std::vector<float> generate_random_matrix(int rows, int cols) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(0.0f, 1.0f);

    std::vector<float> matrix(rows * cols);
    for (int i = 0; i < rows * cols; ++i) {
        matrix[i] = dis(gen);
    }
    return matrix;
}


// Compare two matrices with tolerance
bool compare_matrices(const std::vector<float>& A, const std::vector<float>& B,
                      int rows, int cols, float tolerance = 1e-4f) {
    for (int i = 0; i < rows * cols; ++i) {
        if (std::abs(A[i] - B[i]) > tolerance) {
            std::cout << "Mismatch at element " << i << ": " << A[i] << " vs " << B[i]
                << " (diff: " << std::abs(A[i] - B[i]) << ")" << std::endl;
            return false;
        }
    }
    return true;
}


int main() {
    // Test dimensions
    // constexpr int M = 16;  // Rows of A and C
    // constexpr int N = 8;  // Columns of A, Rows of B
    // constexpr int K = 8;  // Columns of B and C

    constexpr int M = 64;  // Rows of A and C
    constexpr int N = 32;  // Columns of A, Rows of B
    constexpr int K = 48;  // Columns of B and C

    std::cout << "Testing matrix multiplication: " << M << " x " << N << " * "
        << N << " x " << K << " = " << M << " x " << K << std::endl;

    // Generate test matrices
    auto h_A = generate_random_matrix(M, N);
    auto h_B = generate_random_matrix(N, K);
    std::vector<float> h_C_gpu(M * K, 0.0f);
    std::vector<float> h_C_cpu(M * K, 0.0f);

    // Allocate device memory
    float* d_A, * d_B, * d_C;
    cudaMalloc(&d_A, M * N * sizeof(float));
    cudaMalloc(&d_B, N * K * sizeof(float));
    cudaMalloc(&d_C, M * K * sizeof(float));

    // Copy data to device
    cudaMemcpy(d_A, h_A.data(), M * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B.data(), N * K * sizeof(float), cudaMemcpyHostToDevice);

    // Run CUDA kernel
    std::cout << "Running CUDA kernel..." << std::endl;
    solve(d_A, d_B, d_C, M, N, K);

    // Copy result back to host
    cudaMemcpy(h_C_gpu.data(), d_C, M * K * sizeof(float), cudaMemcpyDeviceToHost);

    // Run CPU reference
    std::cout << "Running CPU reference..." << std::endl;
    matrix_multiply_cpu(h_A, h_B, h_C_cpu, M, N, K);

    // Compare results
    std::cout << "Comparing results..." << std::endl;
    bool success = compare_matrices(h_C_gpu, h_C_cpu, M, K);

    if (success) {
        std::cout << "✓ Test PASSED: GPU and CPU results match!" << std::endl;

        // Print a small sample for visual verification
        std::cout << "\nSample results (first 3x3 elements):" << std::endl;
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                std::cout << "C[" << i << "][" << j << "]: GPU=" << h_C_gpu[i * K + j]
                    << ", CPU=" << h_C_cpu[i * K + j] << std::endl;
            }
        }
    } else {
        std::cout << "✗ Test FAILED: GPU and CPU results don't match!" << std::endl;
    }

    // Cleanup
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return success ? 0 : 1;
}
