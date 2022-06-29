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
#define DATATYPE     float
#define SMEMSIZE     1024
#define DATA_OUT_NUM 4

__device__ void yesleep(float t, int clockRate) {
    clock_t t0 = clock64();
    clock_t t1 = t0;
    while ((t1 - t0) / (clockRate * 1000.0f) < t)
        t1 = clock64();
}

__device__ void killtime(DATATYPE* a, int n) {
    for (int i = 0; i < n; i++) {
        a[i] = 0.000;
    }
}

//初始化数组，a[i]=0
void init_order(DATATYPE* a, int n) {
    for (int i = 0; i < n; i++) {
        a[i] = 0.000;
    }
}

__global__ void Test_Kernel(int numBlocks, int numSms, int kernelID,
                            int clockRate, DATATYPE* d_out) {
    clock_t  start_clock = clock();
    float    Start_time = (float)start_clock / clockRate;
    uint32_t smid = getSMID();
    uint32_t blockid = getBlockIDInGrid();
    uint32_t threadid = getThreadIdInBlock();
    yesleep(50.0, clockRate);
    clock_t end_clock = clock();
    float   End_time = (float)end_clock / clockRate;

    __syncthreads();

    //用d_out数组存储输出的数据
    int index = blockid * DATA_OUT_NUM;
    d_out[index] = blockid + 0.000;
    d_out[index + 1] = smid + 0.000;
    d_out[index + 2] = Start_time;
    d_out[index + 3] = End_time;
    // for (int i = 0; i < kernelID; i++) printf("\t");
    printf("%d\t%d\t%.6f\t%.6f\n", blockid,
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

int main_test(int threads, int numBlocks, int numSms, int clockRate, DATATYPE* h_in1) {
    //在device上创建一个数据存储用的数组，通过copy host的数组进行初始化
    DATATYPE* d_out;
    cudaMalloc((void**)&d_out, sizeof(DATATYPE) * DATA_OUT_NUM * numBlocks);
    cudaMemcpy(d_out, h_in1, sizeof(DATATYPE) * DATA_OUT_NUM * numBlocks, cudaMemcpyHostToDevice);

    printf("BlockID\tSMID\tStart_time\tEnd_time\n");
    Test_Kernel<<<numBlocks, threads>>>(numBlocks, numSms, 0, clockRate, d_out);
    //等待kernel执行完毕
    cudaDeviceSynchronize();

    //保存输出数据
    cudaMemcpy(h_in1, d_out, sizeof(DATATYPE) * DATA_OUT_NUM * numBlocks, cudaMemcpyDeviceToHost);

    cudaFree(d_out);
    return 0;
}

int main(void) {
    int            device = 0;
    cudaDeviceProp prop;
    CUcontext      contextPool;
    int            smCounts = 1;
    cudaSetDevice(device);
    // cudaGetDevice(&device);
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

    CUexecAffinityParam affinity2;
    CUresult            err3;
    // Returns the execution affinity setting for the current context
    err3 = cuCtxGetExecAffinity(&affinity2, CU_EXEC_AFFINITY_TYPE_SM_COUNT);
    if (err3 != CUDA_SUCCESS) {
        printf("thread cuCtxGetExecAffinity Error:%s\n",
               MyGetdeviceError(err3));
    }

    //获取当前context对应的线程数目
    numSms = affinity2.param.smCount.val;
    if (numSms != smCounts) {
        printf("Context parititioning SM error!\tPlan:%d\tactual:%d\n", smCounts, numSms);
        // cout<< "Context "<< step << " parititioning SM error!\tPlan:" <<
        // smCounts[step] << "\tactual:" << numSms << endl;
    } else {
        printf("Context parititioning SM success!\tPlan:%d\tactual:%d\n", smCounts, numSms);
    }
    // printf("numSms:%d\n",numSms);
    // printf("thread %d  numBlocksPerSm:%d\n",step, numBlocksPerSm);
    // cout << "KernelID\t"<< step << "\tSMnum\t" << numSms << "\tBlocknum\t"
    // << numBlocks << endl;
    // printf("Block nums:%d\n",numBlocks);
    // dim3 dimBlock(numThreads, 1, 1); //每个Block中thread数目：numThreads
    // dim3 dimGrid(numBlocks, 1, 1);   //每个Grid中Block数目

    //读写文件。文件存在则被截断为零长度，不存在则创建一个新文件
    FILE* fp = NULL;
    fp = fopen("data.csv", "w+");
    if (fp == NULL) {
        fprintf(stderr, "fopen() failed.\n");
        exit(EXIT_FAILURE);
    }
    fprintf(fp, "KernelID,SMnum,Blocknum,BlockID,SMID,Start_time,End_time\n");
    fclose(fp);
    printf("write file title success! \n");

    for (int i = 1; i < 4; i++) {
        numBlocks += 16 * i;
        printf("\nKernelID\t%d\tSMnum\t%d\tBlocknum\t%d\n", i, numSms,
               numBlocks);

        //先在host上创建一个数据存储用的数组，并初始化
        DATATYPE* h_in1;
        h_in1 = (DATATYPE*)malloc(sizeof(DATATYPE) * DATA_OUT_NUM * numBlocks);
        init_order(h_in1, DATA_OUT_NUM * numBlocks);

        main_test(numThreads, numBlocks, numSms, clockRate, h_in1);
        cudaDeviceReset();

        //读写文件。文件不存在则创建新文件。读取会从文件的开头开始，写入则只能是追加模式
        fp = fopen("data.csv", "a+");
        if (fp == NULL) {
            fprintf(stderr, "fopen() failed.\n");
            exit(EXIT_FAILURE);
        }

        for (int j = 0; j < numBlocks; j++) {
            int index = j * DATA_OUT_NUM;
            fprintf(fp, "%d,%d,%d,%.0f,%.0f,%.6f,%.6f\n", i, numSms,
                    numBlocks, h_in1[index], h_in1[index + 1], h_in1[index + 2], h_in1[index + 3]);
        }
        fclose(fp);
        //释放h_in1
        free(h_in1);
    }

    // cudaDeviceReset();

    return 0;
}