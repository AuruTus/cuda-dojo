__global__ void reduce_cuda(
    const float* input,
    float* output,
    int N
) {
    // for every first thread of each warps
    extern __shared__ float smem[];

    auto tid = blockIdx.x * blockDim.x + threadIdx.x;
    auto warp_size = (blockDim.x + 31) / 32;
    auto lane_id = threadIdx.x % 32;
    auto wid = threadIdx.x / 32;
    auto grid_stride = gridDim.x * blockDim.x;

    float value = 0;
    for (auto i = tid; i < N; i += grid_stride) {
        value += input[i];
    }

    for (int offset = 16; offset > 0; offset /= 2) {
        value += __shfl_down_sync(0xFFFFFFFF, value, offset);
    }

    if (lane_id == 0) {
        smem[wid] = value;
    }
    __syncthreads();

    if (wid == 0) {
        value = (lane_id < warp_size) ? smem[lane_id] : 0;
        for (int offset = 16; offset > 0; offset /= 2) {
            value += __shfl_down_sync(0xFFFFFFFF, value, offset);
        }

        if (lane_id == 0) {
            atomicAdd(output, value);
        }
    }
}

// input, output are device pointers
extern "C" void solve(const float* input, float* output, int N) {
    constexpr size_t BLOCK_SIZE = 256;
    constexpr size_t NUM_WAPRS = (BLOCK_SIZE + 31) / 32;
    constexpr size_t SMEM_SIZE = NUM_WAPRS * sizeof(float);

    const size_t NUM_BLOCKS = min(65535, static_cast<int>((N + BLOCK_SIZE - 1) / BLOCK_SIZE));

    reduce_cuda << <NUM_BLOCKS, BLOCK_SIZE, SMEM_SIZE >> > (
        input, output, N
        );
    cudaDeviceSynchronize();
}