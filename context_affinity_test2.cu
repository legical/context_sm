#include <cuda.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#include <iostream>
#include <thread>
#include <utility>

#include "util.cu"

using namespace std;

// __device__ void yesleep(float t) {
//     cudaDeviceProp prop;
//     cudaGetDeviceProperties(&prop, 0);
//     clock_t CLOCK_RATE = prop.clockRate;
//     clock_t t0 = clock64();
//     clock_t t1 = t0;
//     while ((t1 - t0)/(CLOCK_RATE*1000.0f) < t)
//         t1 = clock64();
// }

__global__ void MyKernel(int numSms, int numBlocks, int clockRate) {
    clock_t  start_clock = clock();
    float    Start_time = (float)start_clock / clockRate;
    uint32_t smid = getSMID();
    uint32_t blockid = getBlockIDInGrid();
    uint32_t threadid = getThreadIdInBlock();
    MySleep(35); // about 56ms

    clock_t end_clock = clock();
    float   End_time = (float)end_clock / clockRate;
    // if(i == 3)//只输出其中一个CPU线程的Kernel
    // printf("CPU thread %d:\t
    // SMID:%d,\tBlockID:%d,\tThreadID:%d,\tS_time:%f,\tE_time:%f\tSm_num:%d,\tBlock_num:%d!\n",i,smid,blockid,threadid,Start_time,End_time,numSms,numBlocks);
    if (blockid % 8 == 0) {
        printf(
            "Kernel need %d sms***\tSMID:%d,\tBlockID:%d,\tThreadID:%d,\tS_time:%f,\tE_time:%f\n",
            numSms, smid, blockid, threadid, Start_time, End_time);
    }

    return;
}

__global__ void Test_Kernel(int numBlocks, int numSms, int kernelID,
                            int clockRate) {
    // shared memory : 32 KB
    const uint32_t   SM_size = 32 * 1024 / sizeof(float);
    int              i = 0;
    __shared__ float s_tvalue[SM_size];
    clock_t          start_clock = clock();
    float            Start_time = (float)start_clock / clockRate;
    uint32_t         smid = getSMID();
    uint32_t         blockid = getBlockIDInGrid();
    uint32_t         threadid = getThreadIdInBlock();
    // MySleep(600);
    // clock_t end_clock = clock();
    // float   End_time = (float)end_clock / clockRate;

    for (i = 0; i < SM_size; i++) {
        s_tvalue[i] = i + 8;
    }
    while (i < SM_size) {
        i = s_tvalue[i];
    }
    clock_t end_clock = clock();
    float   End_time = (float)end_clock / clockRate;

    printf("%d\t%d\t%d\t%.6f\t%.6f\n", kernelID, blockid,
           smid, Start_time, End_time);
    return;
}

const char* MyGetRuntimeError(cudaError_t error) {
    if (error != cudaSuccess) {
        return cudaGetErrorString(error);
    } else
        return NULL;
}

char* MyGetdeviceError(CUresult error) {
    if (error != CUDA_SUCCESS) {
        char* charerr = (char*)malloc(100);
        cuGetErrorString(error, (const char**)&charerr);
        return charerr;
    } else
        return NULL;
}

int main(void) {
    //初始化
    // cuInit(0);
    int            device = 0;
    cudaDeviceProp prop;
    const int      CONTEXT_POOL_SIZE = 4;
    CUcontext      contextPool[CONTEXT_POOL_SIZE];
    int            smCounts[CONTEXT_POOL_SIZE];
    cudaSetDevice(device);
    // cudaGetDevice(&device);
    // printf("device:%d\n",device);
    cudaGetDeviceProperties(&prop, device);
    int clockRate = prop.clockRate;
    int sm_number = prop.multiProcessorCount;
    printf("*********   This GPU has %d SMs   *********\n", sm_number);

    smCounts[0] = 1;
    smCounts[1] = 2;
    smCounts[2] = (sm_number - 3) / 3;
    smCounts[3] = (sm_number - 3) / 3 * 2;

    //创建Context
    for (int i = 0; i < CONTEXT_POOL_SIZE; i++) {
        CUexecAffinityParam affinity;
        affinity.type = CU_EXEC_AFFINITY_TYPE_SM_COUNT;
        affinity.param.smCount.val = smCounts[i];

        CUresult err2;
        err2 = cuCtxCreate_v3(&contextPool[i], &affinity, 1, 0, device);

        if (MyGetdeviceError(err2) != NULL) {
            printf("The %d cuCtxCreate_v3 Error:%s\n", i, MyGetdeviceError(err2));
        }
        // cuCtxCreate_v3
        // 创建带有affinity的上下文，并且CU_EXEC_AFFINITY_TYPE_SM_COUNT属性仅在Volta及更新的架构上以及MPS下可用
        //链接：https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__CTX.html#group__CUDA__CTX_1g2a5b565b1fb067f319c98787ddfa4016
        // cuCtxCreate_v3(&contextPool[i], &affinity, 1, 0, deviceOrdinal);
    }

    std::thread mythread[CONTEXT_POOL_SIZE];
    int         step = 0;
    for (step = 0; step < CONTEXT_POOL_SIZE; step++)
        mythread[step] = std::thread([=]() {
            // printf("thread %d start!\n",i);
            int                 numSms = 0;
            int                 numBlocks = 0;
            int                 numBlocksPerSm = 0;
            int                 numThreads = 1; //每个Block中的Thread数目
            CUexecAffinityParam affinity;

            CUresult err1;
            //将指定的CUDA上下文绑定到调用CPU线程
            err1 = cuCtxSetCurrent(contextPool[step]);
            if (err1 != CUDA_SUCCESS) {
                printf("thread cuCtxSetCurrent Error:%s\n", MyGetdeviceError(err1));
            }

            CUresult err2;
            // Returns the execution affinity setting for the current context
            err2 = cuCtxGetExecAffinity(&affinity, CU_EXEC_AFFINITY_TYPE_SM_COUNT);
            if (err2 != CUDA_SUCCESS) {
                printf("thread cuCtxGetExecAffinity Error:%s\n",
                       MyGetdeviceError(err2));
            }

            //获取当前context对应的线程数目
            numSms = affinity.param.smCount.val;
            if (numSms != smCounts[step]) {
                printf("Context %d parititioning SM error!\tPlan:%d\tactual:%d\n", step,
                       smCounts[step], numSms);
                // cout<< "Context "<< step << " parititioning SM error!\tPlan:" <<
                // smCounts[step] << "\tactual:" << numSms << endl;
            } else {
                printf("Context %d parititioning SM successed!\tPlan:%d\tactual:%d\n", step,
                       smCounts[step], numSms);
            }
            // printf("numSms:%d\n",numSms);

            // cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocksPerSm, kernel,
            // numThreads, 0);
            //返回 Kernel的占用率
            cudaError_t error1;
            error1 = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                &numBlocksPerSm, MyKernel, numThreads, 0);
            if (error1 != cudaSuccess) {
                printf(
                    "thread cudaOccupancyMaxActiveBlocksPerMultiprocessor Error:%s\n",
                    MyGetRuntimeError(error1));
            }
            // printf("thread %d  numBlocksPerSm:%d\n",step, numBlocksPerSm);

            numBlocks = 1 + smCounts[step] * 17; // 2个SM，最多同时执行32个Block
            printf("KernelID\t%d\tSMnum\t%d\tBlocknum\t%d\n", step, numSms,
                   numBlocks);
            // cout << "KernelID\t"<< step << "\tSMnum\t" << numSms << "\tBlocknum\t"
            // << numBlocks << endl;
            // printf("Block nums:%d\n",numBlocks);
            dim3 dimBlock(numThreads, 1, 1); //每个Block中thread数目：numThreads
            dim3 dimGrid(numBlocks, 1, 1);   //每个Grid中Block数目

            printf("kernelID\tBlockID\tSMID\tStart_time\tEnd_time\n");

            Test_Kernel<<<dimGrid, dimBlock>>>(numBlocks, numSms, step, clockRate);
        });

    for (step = 0; step < CONTEXT_POOL_SIZE; step++)
        mythread[step].join();

    cudaDeviceReset();
    return 0;
}
