#ifndef __UTIL_CUH__
#define __UTIL_CUH__

#include <stdio.h>
#include <stdint.h>

__device__ __inline__ void busySleep(clock_t clock_count) {
    clock_t start_clock = clock();
    clock_t clock_offset = 0;
    while (clock_offset < clock_count) {
        clock_offset = clock() - start_clock;
    }
}
// 获取当前时间
__device__ __inline__ float nowTime(int clockRate) {
    return (float)clock() / clockRate;
}

// 延时
__device__ __inline__ void yesleep(float t, int clockRate) {
    float Start_time = nowTime(clockRate);
    float clock_offset = 0.0;
    while (clock_offset < t)
        clock_offset = nowTime(clockRate) - Start_time;
}

static __device__ void MySleep(long num) {
    long count = 0;
    for (long i = 0; i <= num; i++) {
        for (long j = 0; j <= num; j++) {
            count = num;
            while (count--) {
                printf("");
            }
        }
    }
}

static __device__ __inline__ uint32_t getThreadNumPerBlock(void) {
    return blockDim.x * blockDim.y * blockDim.z;
}

static __device__ __inline__ uint32_t getBlockNumInGrid(void) {
    return gridDim.x * gridDim.y * gridDim.z;
}

static __device__ __inline__ uint32_t getThreadIdInBlock(void) {
    return blockDim.x * blockDim.y * threadIdx.z + blockDim.x * threadIdx.y + threadIdx.x;
}

static __device__ __inline__ uint32_t getGlobalThreadId(void) {
    return (gridDim.x * gridDim.y * blockIdx.z + gridDim.x * blockIdx.y + blockIdx.x) * (blockDim.x * blockDim.y * blockDim.z) + blockDim.x * blockDim.y * threadIdx.z + blockDim.x * threadIdx.y + threadIdx.x;
}

static __device__ __inline__ uint32_t getBlockIDInGrid(void) {
    return gridDim.x * gridDim.y * blockIdx.z + gridDim.x * blockIdx.y + blockIdx.x;
}

static __device__ __inline__ uint32_t getSMID(void) {
    uint32_t smid;
    asm volatile("mov.u32 %0, %%smid;"
                 : "=r"(smid));
    return smid;
}

static __device__ __inline__ uint32_t getWarpID(void) {
    uint32_t smid;
    asm volatile("mov.u32 %0, %%warpid;"
                 : "=r"(smid));
    return smid;
}

#endif