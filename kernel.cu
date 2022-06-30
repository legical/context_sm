#include <stdio.h>
#include <stdlib.h>
#include <string.h>
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
template <class T>
void init_order(T* a, int n, T para) {
    for (int i = 0; i < n; i++) {
        a[i] = para;
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
    //在device上创建一个数据存储用的数组，通过copy host的数组进行初始化
    DATATYPE* d_out;
    int       numBlocks = numBlock[kernelID];
    cudaMalloc((void**)&d_out, sizeof(DATATYPE) * DATA_OUT_NUM * numBlocks);
    cudaMemcpy(d_out, h_in1, sizeof(DATATYPE) * DATA_OUT_NUM * numBlocks, cudaMemcpyHostToDevice);

    // printf("BlockID\tSMID\tStart_time\tEnd_time\n");
    Test_Kernel<<<numBlocks, threads>>>(numBlocks, numSms, kernelID, clockRate, d_out);
    //等待kernel执行完毕
    cudaDeviceSynchronize();

    //保存输出数据
    printf("kernel %d over, saving data...\n", kernelID);
    cudaMemcpy(h_in1, d_out, sizeof(DATATYPE) * DATA_OUT_NUM * numBlocks, cudaMemcpyDeviceToHost);
    printf("kernel %d saving data success!\n", kernelID);
    cudaFree(d_out);
    return 0;
}

int str_to_int(char buf[]) {
    int num = 0;
    for (int i = 0; i < strlen(buf); i++) {
        // 通过减去'0'可以将字符转换为int类型的数值
        num = num * 10 + buf[i] - '0';
    }
    return num;
}

//命令行传参
int init_para(int argc, char* argv[], int* smCounts, int device_sm_num) {
    init_order(smCounts, device_sm_num, 2);
    int  kernelnum = 0;
    char kernel_num[] = "-k";
    char sm_num[] = "-s";
    for (int i = 1; i < argc; i++) {
        //如果匹配到输入kernel数量的参数
        if (strcmp(argv[i], kernel_num) == 0) {
            kernelnum = str_to_int(argv[i + 1]);
            break;
        }
    }
    printf("kernel number:%d\t", kernelnum);

    for (int i = 1; i < argc; i++) {
        //如果匹配到每个kernel绑定的sm数量的参数
        if (strcmp(argv[i], sm_num) == 0) {
            int smnum = argc - 4;
            int step = smnum;
            if (smnum > kernelnum) {
                printf("sm_to_kernel number > kernel number, the overflow will be discarded!\n");
                step = kernelnum;
            }
            for (int j = 0; j < step; j++) {
                smCounts[j] = str_to_int(argv[i + 1 + j]);
            }
            break;
        }
    }

    printf("\teach sm_to_kernel: ");
    int allsm = 0;
    for (int j = 0; j < kernelnum; j++) {
        allsm += smCounts[j];
        printf("%d  ", smCounts[j]);
    }
    printf("\n");

    if (allsm > device_sm_num) {
        printf("allocate sm number > device total sm number, exit!\n");
        exit(-1);
    }

    return kernelnum;
}

int main(int argc, char* argv[]) {
    //初始化
    // cuInit(0);
    int            device = 0;
    cudaDeviceProp prop;
    int            sizecsv = 0;
    int            allnumblocks = 0;
    cudaSetDevice(device);
    // printf("device:%d\n",device);
    cudaGetDeviceProperties(&prop, device);
    int clockRate = prop.clockRate;
    int sm_number = prop.multiProcessorCount;
    printf("*********   This GPU has %d SMs   *********\n", sm_number);
    // output GPU prop

    int* smC;
    smC = (int*)malloc(sizeof(int) * sm_number);
    const int CONTEXT_POOL_SIZE = init_para(argc, argv, smC, sm_number);

    // const int      CONTEXT_POOL_SIZE = 4;
    CUcontext contextPool[CONTEXT_POOL_SIZE];
    int       smCounts[CONTEXT_POOL_SIZE];
    int       numBlocks[CONTEXT_POOL_SIZE];
    for (int i = 0; i < CONTEXT_POOL_SIZE; i++) {
        smCounts[i] = smC[i];
    }
    free(smC);
    // smCounts[0] = 6;
    // smCounts[1] = 4;
    // smCounts[2] = 4;
    // smCounts[3] = 2;

    //获取当前设备的 COMPUTE_MODE
    CUresult err1;
    err1 = cuDeviceGetAttribute(&device, CU_DEVICE_ATTRIBUTE_COMPUTE_MODE, device);
    if (err1 != CUDA_SUCCESS) {
        printf("cuDeviceGetAttribute Error:%s\n", MyGetdeviceError(err1));
    }

    // save all output data
    DATATYPE** h_data;
    h_data = (DATATYPE**)malloc(sizeof(DATATYPE*) * CONTEXT_POOL_SIZE);

    //创建Context
    for (int i = 0; i < CONTEXT_POOL_SIZE; i++) {
        CUexecAffinityParam affinity;
        affinity.type = CU_EXEC_AFFINITY_TYPE_SM_COUNT;
        affinity.param.smCount.val = smCounts[i];

        CUresult err2;
        err2 = cuCtxCreate_v3(&contextPool[i], &affinity, 1, 0, 0);

        if (MyGetdeviceError(err2) != NULL) {
            printf("The %d cuCtxCreate_v3 Error:%s\n", i, MyGetdeviceError(err2));
        }
        // cuCtxCreate_v3 创建带有affinity的上下文，并且CU_EXEC_AFFINITY_TYPE_SM_COUNT属性仅在Volta及更新的架构上以及MPS下可用
        //链接：https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__CTX.html#group__CUDA__CTX_1g2a5b565b1fb067f319c98787ddfa4016
        // cuCtxCreate_v3(&contextPool[i], &affinity, 1, 0, deviceOrdinal);

        numBlocks[i] = 1 + smCounts[i] * 17;
        sizecsv += numBlocks[i] * DATA_OUT_NUM;
        allnumblocks += numBlocks[i];
        //为每个线程分配data数组
        h_data[i] = (DATATYPE*)malloc(sizeof(DATATYPE) * numBlocks[i] * DATA_OUT_NUM);
    }

    //读写文件。文件存在则被截断为零长度，不存在则创建一个新文件
    FILE* fp = fopen("outdata.csv", "w+");
    if (fp == NULL) {
        fprintf(stderr, "fopen() failed.\n");
        exit(EXIT_FAILURE);
    }
    fprintf(fp, "KernelID,SMnum,Blocknum,BlockID,SMID,Start_time,End_time\n");
    fclose(fp);
    // printf("write file title success! \n");
    
    printf("BlockID\tSMID\tStart_time\tEnd_time\n");
    std::thread mythread[CONTEXT_POOL_SIZE];
    int         step = 0;
    for (step = 0; step < CONTEXT_POOL_SIZE; step++)
        mythread[step] = std::thread([&, step]() {
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
                printf("Context parititioning SM error!\tPlan:%d\tactual:%d\n", smCounts[step], numSms);
                // cout<< "Context "<< step << " parititioning SM error!\tPlan:" <<
                // smCounts[step] << "\tactual:" << numSms << endl;
            } else {
                printf("Context parititioning SM success!\tPlan:%d\tactual:%d\n", smCounts[step], numSms);
            }
            DATATYPE temp = 0;
            init_order(h_data[step], numBlocks[step], temp);

            main_test(step, numThreads, numBlocks, numSms, clockRate, h_data[step]);
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
    for (step = 0; step < CONTEXT_POOL_SIZE; step++) {
        for (int j = 0; j < numBlocks[step]; j++) {
            int index = j * DATA_OUT_NUM;
            fprintf(fp, "%.0f,%.0f,%.0f,%.0f,%.0f,%.6f,%.6f\n", h_data[step][index], h_data[step][index + 1], h_data[step][index + 2],
                    h_data[step][index + 3], h_data[step][index + 4], h_data[step][index + 5], h_data[step][index + 6]);
        }
    }

    fclose(fp);

    for (step = 0; step < CONTEXT_POOL_SIZE; step++) {
        free(h_data[step]);
    }
    free(h_data);
    return 0;
}