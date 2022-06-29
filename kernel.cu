#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <unistd.h>
#include <cuda.h>
#include "util.cu"
#include <iostream>
#include <utility>
#include <thread>
using namespace std;

#define DATATYPE     float
#define SMEMSIZE     1024
#define DATA_OUT_NUM 7

__device__ void yesleep(float t, int clockRate) {
    clock_t t0 = clock64();
    clock_t t1 = t0;
    while ((t1 - t0) / (clockRate * 1000.0f) < t)
        t1 = clock64();
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
    d_out[index] = kernelID + 0.000;
    d_out[index + 1] = numSms + 0.000;
    d_out[index + 2] = numBlocks + 0.000;
    d_out[index + 3] = blockid + 0.000;
    d_out[index + 4] = smid + 0.000;
    d_out[index + 5] = Start_time;
    d_out[index + 6] = End_time;
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

int main_test(int kernelID, int threads, int* numBlock, int numSms, int clockRate, DATATYPE* h_in1) {
    int index = 0;
    for (int i = 0; i < kernelID; i++) {
        index += numBlock[i];
    }
    const float *h_in1_index = &h_in1[index];
    //在device上创建一个数据存储用的数组，通过copy host的数组进行初始化
    DATATYPE* d_out;
    int       numBlocks = numBlock[kernelID];
    cudaMalloc((void**)&d_out, sizeof(DATATYPE) * DATA_OUT_NUM * numBlocks);
    cudaMemcpy(d_out, h_in1_index, sizeof(DATATYPE) * DATA_OUT_NUM * numBlocks, cudaMemcpyHostToDevice);

    printf("BlockID\tSMID\tStart_time\tEnd_time\n");
    Test_Kernel<<<numBlocks, threads>>>(numBlocks, numSms, kernelID, clockRate, d_out);
    //等待kernel执行完毕
    cudaDeviceSynchronize();

    //保存输出数据
    cudaMemcpy(h_in1_index, d_out, sizeof(DATATYPE) * DATA_OUT_NUM * numBlocks, cudaMemcpyDeviceToHost);

    cudaFree(d_out);
    return 0;
}

int main(void) {
    //初始化
    // cuInit(0);
    int            device = 0;
    cudaDeviceProp prop;
    const int      CONTEXT_POOL_SIZE = 4;
    CUcontext      contextPool[CONTEXT_POOL_SIZE];
    int            smCounts[CONTEXT_POOL_SIZE];
    int            numBlocks[CONTEXT_POOL_SIZE];
    int            sizecsv = 0;
    int            allnumblocks = 0;
    cudaSetDevice(device);
    // printf("device:%d\n",device);
    cudaGetDeviceProperties(&prop, device);
    int clockRate = prop.clockRate;
    int sm_number = prop.multiProcessorCount;
    cout << "*********   This GPU has " << sm_number << " SMs   *********" << endl;
    // output GPU prop

    smCounts[0] = 6;
    smCounts[1] = 4;
    smCounts[2] = 4;
    smCounts[3] = 2;

    //获取当前设备的 COMPUTE_MODE
    CUresult err1;
    err1 = cuDeviceGetAttribute(&device, CU_DEVICE_ATTRIBUTE_COMPUTE_MODE, device);
    if (err1 != CUDA_SUCCESS) {
        printf("cuDeviceGetAttribute Error:%s\n", MyGetdeviceError(err1));
    }

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
        // cuCtxCreate_v3 创建带有affinity的上下文，并且CU_EXEC_AFFINITY_TYPE_SM_COUNT属性仅在Volta及更新的架构上以及MPS下可用
        //链接：https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__CTX.html#group__CUDA__CTX_1g2a5b565b1fb067f319c98787ddfa4016
        // cuCtxCreate_v3(&contextPool[i], &affinity, 1, 0, deviceOrdinal);

        numBlocks[i] = 1 + smCounts[i] * 17;
        sizecsv += numBlocks[i] * DATA_OUT_NUM;
        allnumblocks += numBlocks[i];
    }

    // save all output data
    DATATYPE* h_data;
    h_data = (DATATYPE*)malloc(sizeof(DATATYPE) * sizecsv);
    init_order(h_data, sizecsv);

    //读写文件。文件存在则被截断为零长度，不存在则创建一个新文件
    FILE* fp = NULL;
    fp = fopen("outdata.csv", "w+");
    if (fp == NULL) {
        fprintf(stderr, "fopen() failed.\n");
        exit(EXIT_FAILURE);
    }
    fprintf(fp, "KernelID,SMnum,Blocknum,BlockID,SMID,Start_time,End_time\n");
    fclose(fp);
    printf("write file title success! \n");

    std::thread mythread[CONTEXT_POOL_SIZE];
    int         step = 0;
    for (step = 0; step < CONTEXT_POOL_SIZE; step++)
        mythread[step] = std::thread([=]() {
            // printf("thread %d start!\n",i);
            int                 numSms = 0;
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
                printf("thread cuCtxGetExecAffinity Error:%s\n", MyGetdeviceError(err2));
            }

            //获取当前context对应的线程数目
            numSms = affinity.param.smCount.val;
            if (numSms != smCounts[step]) {
                printf("Context parititioning SM error!\tPlan:%d\tactual:%d\n", smCounts, numSms);
                // cout<< "Context "<< step << " parititioning SM error!\tPlan:" <<
                // smCounts[step] << "\tactual:" << numSms << endl;
            } else {
                printf("Context parititioning SM success!\tPlan:%d\tactual:%d\n", smCounts, numSms);
            }

            main_test(step, numThreads, numBlocks, numSms, clockRate, h_data);
        });

    for (step = 0; step < CONTEXT_POOL_SIZE; step++)
        mythread[step].join();

    cudaDeviceReset();

    //读写文件。文件不存在则创建新文件。读取会从文件的开头开始，写入则只能是追加模式
    fp = fopen("outdata.csv", "a+");
    if (fp == NULL) {
        fprintf(stderr, "fopen() failed.\n");
        exit(EXIT_FAILURE);
    }

    for (int j = 0; j < allnumblocks; j++) {
        int index = j * DATA_OUT_NUM;
        fprintf(fp, "%.0f,%.0f,%.0f,%.0f,%.0f,%.6f,%.6f\n", h_data[index], h_data[index + 1], h_data[index + 2], h_data[index + 3], h_data[index + 4], h_data[index + 5], h_data[index + 6]);
    }
    fclose(fp);

    free(h_data);
    return 0;
}

// int main(void)
// {
//     //初始化
//     // cuInit(0);
//     int device = 0;
//     cudaDeviceProp prop;
//     CUcontext contextPool;
//     int smCounts;
//     cudaGetDevice(&device);
//     // printf("device:%d\n",device);
//     cudaGetDeviceProperties(&prop, device);
//     int clockRate = prop.clockRate;
//     smCounts = 2;
//     std::thread mythread;
//     int pi = -1;

//     //获取当前设备的 COMPUTE_MODE
//     CUresult err1;
//     err1 = cuDeviceGetAttribute(&pi, CU_DEVICE_ATTRIBUTE_COMPUTE_MODE, device);
//     if (err1 != CUDA_SUCCESS)
//     {
//         printf("cuDeviceGetAttribute Error:%s\n", MyGetdeviceError(err1));
//     }
//     // printf("cuDeviceGetAttribute:%d\n",pi);

//     //创建Context
//     CUexecAffinityParam affinity;
//     affinity.type = CU_EXEC_AFFINITY_TYPE_SM_COUNT;
//     affinity.param.smCount.val = smCounts;
//     // cuCtxCreate_v3 创建带有affinity的上下文，并且CU_EXEC_AFFINITY_TYPE_SM_COUNT属性仅在Volta及更新的架构上以及MPS下可用
//     //链接：https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__CTX.html#group__CUDA__CTX_1g2a5b565b1fb067f319c98787ddfa4016
//     // cuCtxCreate_v3(&contextPool[i], &affinity, 1, 0, deviceOrdinal);

//     CUresult err2;
//     err2 = cuCtxCreate_v3(&contextPool, &affinity, 1, 0, device);
//     if (MyGetdeviceError(err2) != NULL)
//     {
//         printf("cuCtxCreate_v3 Error:%s\n", MyGetdeviceError(err2));
//     }

//     mythread = std::thread([contextPool, clockRate]()
//     {
//         // printf("thread %d start!\n",i);
//         int numSms = 0;
//         int numBlocks = 0;
//         int numBlocksPerSm = 0;
//         int numThreads = 1; //每个Block中的Thread数目
//         CUexecAffinityParam affinity;

//         CUresult err1;
//         //将指定的CUDA上下文绑定到调用CPU线程
//         err1 = cuCtxSetCurrent(contextPool);
//         if (err1 != CUDA_SUCCESS)
//         {
//             printf("thread cuCtxSetCurrent Error:%s\n", MyGetdeviceError(err1));
//         }

//         CUresult err2;
//         // Returns the execution affinity setting for the current context
//         err2 = cuCtxGetExecAffinity(&affinity, CU_EXEC_AFFINITY_TYPE_SM_COUNT);
//         if (err2 != CUDA_SUCCESS)
//         {
//             printf("thread cuCtxGetExecAffinity Error:%s\n", MyGetdeviceError(err2));
//         }

//         //获取当前context对应的线程数目
//         numSms = affinity.param.smCount.val;
//         printf("numSms:%d\n",numSms);

//         // cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocksPerSm, kernel, numThreads, 0);
//         //返回 Kernel的占用率
//         cudaError_t error1;
//         error1 = cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocksPerSm, MyKernel, numThreads, 0);
//         if (error1 != cudaSuccess)
//         {
//             printf("thread cudaOccupancyMaxActiveBlocksPerMultiprocessor Error:%s\n", MyGetRuntimeError(error1));
//         }
//         printf("numBlocksPerSm:%d\n",numBlocksPerSm);

//         numBlocks = 33;//2个SM，最多同时执行32个Block
//         printf("Block nums:%d\n",numBlocks);
//         dim3 dimBlock(numThreads, 1, 1);//每个Block中thread数目：numThreads
//         dim3 dimGrid(numBlocks, 1, 1);//每个Grid中Block数目
//         // void *kernelArgs[] = {(int *)&numSms, (int *)&numBlocks, (int *)&clockRate}; /* add kernel args */
//         // // Launches a device function where thread blocks can cooperate and synchronize as they execute
//         // // https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__EXECUTION.html#group__CUDART__EXECUTION_1g504b94170f83285c71031be6d5d15f73
//         // cudaError_t error2;
//         // error2 = cudaLaunchCooperativeKernel((void *)MyKernel, dimGrid, dimBlock, kernelArgs);
//         // if (error2 != cudaSuccess)
//         // {
//         //     printf("thread cudaLaunchCooperativeKernel Error:%s\n", MyGetRuntimeError(error2));
//         // }
//         MyKernel<<<dimGrid,dimBlock>>>(numSms,numBlocks,clockRate);
//     });
//     mythread.join();

//     cudaDeviceReset();
//     return 0;
// }
