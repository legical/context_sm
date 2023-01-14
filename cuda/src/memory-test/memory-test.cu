#include "myutil.hpp"
#include "util.cuh"

__global__ void read_random_arr(int *arr_gpu, const int ARR_SIZE/*, const int inter_cycle*/)
{
    uint32_t threadid = getThreadIdInBlock();
// #pragma unroll
    for (int j = 0; j < 3; j++)
    {
        int i = threadid;
#pragma unroll
        while (i < ARR_SIZE)
        {
            i = arr_gpu[i] + 31;
        }
        // for (int i = threadid; i < ARR_SIZE; i += 32)
        // {
        //     arr_gpu[i] |= i & 1;
        // }
    }
}

int main(int argc, char *argv[])
{
    // Default: array size = 1GB
    int ARR_SIZE = 1024 * 1024 * 256/*, inter_cycle = 8*/;

    int *arr, *arr_gpu;

    // allocate pinned memory in system memory
    gpuErrAssert(cudaHostAlloc((void **)&arr,
                               ARR_SIZE * sizeof(int),
                               cudaHostAllocDefault));
    init_chase_arr<int>(arr, ARR_SIZE, 1);

    gpuErrAssert(cudaMalloc((void **)&arr_gpu, ARR_SIZE * sizeof(int)));

    // copy random memory from host to gpu
    gpuErrAssert(cudaMemcpy(arr_gpu, arr, ARR_SIZE * sizeof(int), cudaMemcpyHostToDevice));

    // run kernel for random GPU memory access
    read_random_arr<<<1, 32>>>(arr_gpu, ARR_SIZE/*, inter_cycle*/);

    // copy back random memory from gpu to host
    gpuErrAssert(cudaMemcpy(arr, arr_gpu, ARR_SIZE * sizeof(int), cudaMemcpyDeviceToHost));

    gpuErrAssert(cudaFree(arr_gpu));
    gpuErrAssert(cudaFreeHost(arr));

    return 0;
}