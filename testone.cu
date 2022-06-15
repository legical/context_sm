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

__device__ void yesleep(float t, int clockRate) {
    clock_t t0 = clock64();
    clock_t t1 = t0;
    while ((t1 - t0) / (clockRate * 1000.0f) < t)
        t1 = clock64();
}

__global__ void Test_Kernel(int numBlocks, int numSms, int kernelID,
                            int clockRate) {
    clock_t  start_clock = clock();
    float    Start_time = (float)start_clock / clockRate;
    uint32_t smid = getSMID();
    uint32_t blockid = getBlockIDInGrid();
    uint32_t threadid = getThreadIdInBlock();
    yesleep(600.0, clockRate);
    clock_t end_clock = clock();
    float   End_time = (float)end_clock / clockRate;

    for (int i = 0; i < kernelID; i++) printf("\t");
    printf("BlockID\t%d\tSMID\t%d\tStart_time\t%.6f\tEnd_time\t%.6f\n", blockid,
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

int main_test(int threads, int numBlocks, int numSms, int clockRate) {
    Test_Kernel<<<numBlocks, threads>>>(numBlocks, numSms, 0, clockRate);
    cudaDeviceSynchronize();
    return 0;
}

int main(void) {
    int            device = 0;
    cudaDeviceProp prop;
    CUcontext      contextPool;
    int            smCounts = 1;
    cudaGetDevice(&device);
    // printf("device:%d\n",device);
    cudaGetDeviceProperties(&prop, device);
    int clockRate = prop.clockRate;
    int sm_number = prop.multiProcessorCount;
    printf("*********   This GPU has %d SMs   *********\n", sm_number);
    // cout << "*********   This GPU has " << sm_number << " SMs   *********"
    // <<endl;
    // output GPU prop

    CUexecAffinityParam affinity;
    affinity.type = CU_EXEC_AFFINITY_TYPE_SM_COUNT;
    affinity.param.smCount.val = smCounts;

    CUresult err2;
    err2 = cuCtxCreate_v3(&contextPool, &affinity, 1, 0, device);

    if (MyGetdeviceError(err2) != NULL) {
        printf("cuCtxCreate_v3 Error:%s\n", MyGetdeviceError(err2));
    }

    int      numSms = 0;
    int      numBlocks = 0;
    int      numBlocksPerSm = 0;
    int      numThreads = 1; //每个Block中的Thread数目
    CUresult err1;
    //将指定的CUDA上下文绑定到调用CPU线程
    err1 = cuCtxSetCurrent(contextPool);
    if (err1 != CUDA_SUCCESS) {
        printf("thread cuCtxSetCurrent Error:%s\n", MyGetdeviceError(err1));
    }

    CUresult err3;
    // Returns the execution affinity setting for the current context
    err3 = cuCtxGetExecAffinity(&affinity, CU_EXEC_AFFINITY_TYPE_SM_COUNT);
    if (err3 != CUDA_SUCCESS) {
        printf("thread cuCtxGetExecAffinity Error:%s\n",
               MyGetdeviceError(err3));
    }

    //获取当前context对应的线程数目
    numSms = affinity.param.smCount.val;
    if (numSms != smCounts) {
        printf("Context parititioning SM error!\tPlan:%d\tactual:%d\n", smCounts, numSms);
        // cout<< "Context "<< step << " parititioning SM error!\tPlan:" <<
        // smCounts[step] << "\tactual:" << numSms << endl;
    }
    // printf("numSms:%d\n",numSms);
    // printf("thread %d  numBlocksPerSm:%d\n",step, numBlocksPerSm);
    // cout << "KernelID\t"<< step << "\tSMnum\t" << numSms << "\tBlocknum\t"
    // << numBlocks << endl;
    // printf("Block nums:%d\n",numBlocks);
    // dim3 dimBlock(numThreads, 1, 1); //每个Block中thread数目：numThreads
    // dim3 dimGrid(numBlocks, 1, 1);   //每个Grid中Block数目

    for (int i = 1; i < 10; i++) {
        numBlocks += 16 * i;
        printf("KernelID\t%d\tSMnum\t%d\tBlocknum\t%d\n", i, numSms,
               numBlocks);
        main_test(numThreads, numBlocks, numSms, clockRate);
    }

    cudaDeviceReset();

    return 0;
}