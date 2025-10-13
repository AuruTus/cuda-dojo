#include <cstdio>
#include <cstdlib>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = (call); \
        if (err != cudaSuccess) { \
            printf("CUDA error at %s:%d - %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(1); \
        } \
    } while(0)

__global__ void vectorAdd(int* a, int* b, int* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

int main() {
    const int N = 10000;
    const int block_size = 256;
    const int grid_size = (N + block_size - 1) / block_size;

    // Host arrays
    int* h_a = new int[N];
    int* h_b = new int[N];
    int* h_c = new int[N];

    // Device arrays
    int* d_a, * d_b, * d_c;

    // Initialize host data
    for (int i = 0; i < N; i++) {
        h_a[i] = i;
        h_b[i] = i * 2;
    }

    // Allocate device memory with error checking
    CUDA_CHECK(cudaMalloc(&d_a, N * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_b, N * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_c, N * sizeof(int)));

    // Copy to device
    CUDA_CHECK(cudaMemcpy(d_a, h_a, N * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b, N * sizeof(int), cudaMemcpyHostToDevice));

    // Launch kernel
    printf("Launching kernel with %d blocks, %d threads per block\n",
           grid_size, block_size);
    vectorAdd << <grid_size, block_size >> > (d_a, d_b, d_c, N);

    // Check kernel execution
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy back results
    CUDA_CHECK(cudaMemcpy(h_c, d_c, N * sizeof(int), cudaMemcpyDeviceToHost));

    // Verify results
    bool success = true;
    for (int i = 0; i < N; i++) {
        if (h_c[i] != h_a[i] + h_b[i]) {
            printf("Error at index %d: %d != %d + %d\n",
                   i, h_c[i], h_a[i], h_b[i]);
            success = false;
            break;
        }
    }

    if (success) {
        printf("Kernel executed successfully!\n");
        printf("Sample result: %d + %d = %d\n", h_a[100], h_b[100], h_c[100]);
    }

    // Cleanup
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));
    delete[] h_a;
    delete[] h_b;
    delete[] h_c;

    return 0;
}