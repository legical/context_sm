/**
 * @file kernel.cu
 * @author lzh
 * @version 0.1
 * 读取共享内存时，每个sm最多同时运行3个block
 * 单纯sleep时，对于同一个kernel，每个sm最多同时运行其16个block
 * 但可以运行多个kernel的block，启动时间略有差异
 * @copyright Copyright (c) 2022
 * nvcc -arch sm_86 -lcuda -o test kernel.cu util.cu
 * ./test -k4 -p1 -b16 -s"6,2,4,2"
 * ./test -k4 -p0 -b16 -s"6,2,4,2"
 * ./test -k4 -p2 -b16 -s"6,2,4,2"
 */
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

#define DATATYPE float
// #define DATATYPE     uint32_t
#define SMEMSIZE     1024
#define DATA_OUT_NUM 7

// __device__ void yesleep(float t, int clockRate) {
//     clock_t  start_clock = clock();
//     float    Start_time = (float)start_clock / clockRate;
//     clock_t t0 = clock64();
//     clock_t t1 = t0;
//     while ((t1 - t0) / (clockRate * 1000.0f) < t)
//         t1 = clock64();
// }

//初始化数组，a[i]=0
template <class T>
void init_order(T* a, int n, T para) {
    for (int i = 0; i < n; i++) {
        a[i] = para;
    }
}

__global__ void Test_Kernel_sleep(int numBlocks, int numSms, int kernelID,
                                  int clockRate, DATATYPE* d_out) {
    clock_t  start_clock = clock();
    float    Start_time = (float)start_clock / clockRate;
    uint32_t smid = getSMID();
    uint32_t blockid = getBlockIDInGrid();
    uint32_t threadid = getThreadIdInBlock();
    yesleep(200.0, clockRate);
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
    printf("\t%d\t%d\t%d\t%.6f\t%.6f\t%.6f\n", kernelID, blockid,
           smid, Start_time, End_time, End_time - Start_time);

    return;
}

__global__ void Test_Kernel_global(int numBlocks, int numSms, int kernelID,
                                   int clockRate, DATATYPE* d_out, DATATYPE* d_array) {
    __shared__ DATATYPE time[2];
    time[0] = nowTime(clockRate);
    uint32_t smid = getSMID();
    uint32_t blockid = getBlockIDInGrid();
    uint32_t threadid = getThreadIdInBlock();
    // printf("here is global\n");
    const uint32_t d_array_num = sizeof(DATATYPE) * 1024 * 1024;

#pragma unroll
    for (uint32_t j = 0; j < 1024; j++) {
        for (uint32_t i = 0; i < d_array_num; i++) {
            d_array[i] = i + kernelID + j;
        }
    }

    time[1] = nowTime(clockRate);

    __syncthreads();
    // printf("here is global\n");
    //用d_out数组存储输出的数据
    int index = blockid * DATA_OUT_NUM;
    d_out[index] = kernelID + 0.000;
    d_out[index + 1] = numSms + 0.000;
    d_out[index + 2] = numBlocks + 0.000;
    d_out[index + 3] = blockid + 0.000;
    d_out[index + 4] = smid + 0.000;
    d_out[index + 5] = time[0];
    d_out[index + 6] = time[1];

    // for (int i = 0; i < kernelID; i++) printf("\t");
    printf("\t%d\t%d\t%d\t%.6f\t%.6f\t%.6f\n", kernelID, blockid,
           smid, time[0], time[1], time[1] - time[0]);

    return;
}

__global__ void Test_Kernel(int numBlocks, int numSms, int kernelID,
                            int clockRate, DATATYPE* d_out) {
    const uint32_t   SM_size = 32 * 1024 / sizeof(float);
    int              i = 0;
    __shared__ float s_tvalue[SM_size];

    clock_t  start_clock = clock();
    float    Start_time = (float)start_clock / clockRate;
    uint32_t smid = getSMID();
    uint32_t blockid = getBlockIDInGrid();
    uint32_t threadid = getThreadIdInBlock();
    // yesleep(50.0, clockRate);
    for (i = 0; i < SM_size; i++) {
        s_tvalue[i] = i + kernelID + 1;
    }
#pragma unroll
    for (int j = 0; j < SM_size; j++) {
        i = 0;
        while (i < SM_size) {
            i = s_tvalue[i];
        }
    }

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
    // d_out[index] = kernelID;
    // d_out[index + 1] = numSms;
    // d_out[index + 2] = numBlocks;
    // d_out[index + 3] = blockid;
    // d_out[index + 4] = smid;
    // d_out[index + 5] = (DATATYPE)(start_clock);
    // d_out[index + 6] = (DATATYPE)(end_clock);
    // for (int i = 0; i < kernelID; i++) printf("\t");
    printf("\t%d\t%d\t%d\t%.6f\t%.6f\t%.6f\n", kernelID, blockid,
           smid, Start_time, End_time, End_time - Start_time);

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

int main_test(int kernelID, int threads, int* numBlock, int numSms, int clockRate, DATATYPE* h_in1, int patt, DATATYPE* d_array) {
    //在device上创建一个数据存储用的数组，通过copy host的数组进行初始化
    DATATYPE* d_out;
    DATATYPE* d_global;
    int       numBlocks = numBlock[kernelID];
    cudaMalloc((void**)&d_out, sizeof(DATATYPE) * DATA_OUT_NUM * numBlocks);
    // 4MB
    cudaMalloc((void**)&d_global, sizeof(DATATYPE) * 1024 * 1024);
    cudaMemcpy(d_out, h_in1, sizeof(DATATYPE) * DATA_OUT_NUM * numBlocks, cudaMemcpyHostToDevice);
    cudaMemcpy(d_global, d_array, sizeof(DATATYPE) * 1024 * 1024, cudaMemcpyHostToDevice);
    
    // printf("BlockID\tSMID\tStart_time\tEnd_time\n");
    switch (patt) {
    case 0:
        // 0-使用sleep测试
        Test_Kernel_sleep<<<numBlocks, threads>>>(numBlocks, numSms, kernelID, clockRate, d_out);
        break;
    case 2:
        // 2-使用global memory测试
        Test_Kernel_global<<<numBlocks, threads>>>(numBlocks, numSms, kernelID, clockRate, d_out, d_global);
        break;
    default:
        //默认使用共享内存测试
        Test_Kernel<<<numBlocks, threads>>>(numBlocks, numSms, kernelID, clockRate, d_out);
        break;
    }
    //等待kernel执行完毕
    cudaDeviceSynchronize();

    //保存输出数据
    printf("kernel %d over, saving data...\n", kernelID);
    cudaMemcpy(h_in1, d_out, sizeof(DATATYPE) * DATA_OUT_NUM * numBlocks, cudaMemcpyDeviceToHost);
    printf("kernel %d saving data success!\n", kernelID);
    cudaFree(d_out);
    cudaFree(d_global);
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

char* int_to_str(int num, char* str) // 10进制
{
    int i = 0;   //指示填充str
    if (num < 0) //如果num为负数，将num变正
    {
        num = -num;
        str[i++] = '-';
    }
    //转换
    do {
        str[i++] = num % 10 + 48; //取num最低位 字符0~9的ASCII码是48~57；简单来说数字0+48=48，ASCII码对应字符'0'
        num /= 10;                //去掉最低位
    } while (num);                // num不为0继续循环

    str[i] = '\0';

    //确定开始调整的位置
    int j = 0;
    if (str[0] == '-') //如果有负号，负号不用调整
    {
        j = 1; //从第二位开始调整
        ++i;   //由于有负号，所以交换的对称轴也要后移1位
    }
    //对称交换
    for (; j < i / 2; j++) {
        //对称交换两端的值 其实就是省下中间变量交换a+b的值：a=a+b;b=a-b;a=a-b;
        str[j] = str[j] + str[i - 1 - j];
        str[i - 1 - j] = str[j] - str[i - 1 - j];
        str[j] = str[j] - str[i - 1 - j];
    }

    return str; //返回转换后的值
}

char* gene_filename(char* filename, int* smCounts, int block_per_sm, int CONTEXT_POOL_SIZE, int patt) {
    filename[0] = '\0';

    switch (patt) {
    case 0:
        strcat(filename, "./outdata/sleep-");
        printf("test pattern is : sleep\n");
        break;
    case 2:
        strcat(filename, "./outdata/globa-");
        printf("test pattern is : load global memory\n");
        break;
    default:
        strcat(filename, "./outdata/share-");
        printf("test pattern is : reading shared memory many times\n");
        break;
    }
    strcat(filename, "outdata-k");
    {
        char kernelnum[3];
        strcat(filename, int_to_str(CONTEXT_POOL_SIZE, kernelnum));
    }
    strcat(filename, "-s");
    for (int i = 0; i < CONTEXT_POOL_SIZE; i++) {
        char smC[2];
        strcat(filename, int_to_str(smCounts[i], smC));
        // free(smC);
    }
    strcat(filename, "-b");
    char block[2];
    strcat(filename, int_to_str(block_per_sm, block));
    // free(block);
    strcat(filename, ".csv");

    return filename;
}

//命令行传参
int init_para(int argc, char* argv[], int* smCounts, int device_sm_num, int* block_per_sm) {
    init_order(smCounts, device_sm_num, 2);
    int  kernelnum = 0;
    char kernel_num[] = "-k";
    char sm_num[] = "-s";
    //每个sm上启动的block数量
    char block_per[] = "-b";
    int  block_per_yes = 0;
    for (int i = 1; i < argc; i++) {
        //如果匹配到输入kernel数量的参数
        if (strcmp(argv[i], kernel_num) == 0) {
            kernelnum = str_to_int(argv[i + 1]);
            break;
        }
    }
    printf("kernel number:%d\t", kernelnum);

    for (int i = 1; i < argc; i++) {
        //如果匹配到输入kernel数量的参数
        if (strcmp(argv[i], block_per) == 0) {
            *block_per_sm = str_to_int(argv[i + 1]);
            block_per_yes = 1;
            break;
        }
    }

    for (int i = 1; i < argc; i++) {
        //如果匹配到每个kernel绑定的sm数量的参数
        if (strcmp(argv[i], sm_num) == 0) {
            int smnum = argc - 4 - block_per_yes * 2;
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
    printf("\tblocks_per_sm: %d\n", *block_per_sm);

    if (allsm > device_sm_num) {
        printf("allocate sm number > device total sm number, exit!\n");
        exit(-1);
    }

    return kernelnum;
}

//命令行传参
int init_getopt(int argc, char* argv[], int* smCounts, int device_sm_num, int* block_per_sm, int* patt) {
    init_order(smCounts, device_sm_num, 2);
    int kernelnum = 0;

    int para;
    /**
     * @brief 4个参数
     * -p 0-sleep 1-share 2-GPU global memory
     * -s "6,2,2,4" sm number of each kernel
     * -b int block number of every kernel
     * -k int kernel number
     * ./test -k4 -p1 -b3 -s"6,2,4,2"
     */
    const char* optstring = "k::p::b::s::";

    while ((para = getopt(argc, argv, optstring)) != -1) {
        switch (para) {
        case 'k':
            kernelnum = str_to_int(optarg);
            printf("kernelnum succeed!\n");
            break;
        case 'p':
            // if (optarg[0] == 't' || optarg[0] == 'y') {
            //     *patt = true;
            // } else {
            //     *patt = false;
            // }
            *patt = str_to_int(optarg);
            printf("patt succeed!\n");
            break;
        case 'b':
            *block_per_sm = str_to_int(optarg);
            printf("block_per_sm succeed!\n");
            break;
        case 's': {
            char* temp = strtok(optarg, ",");
            int   count = 0;
            while (temp) {
                smCounts[count++] = str_to_int(temp);
                temp = strtok(NULL, ",");
            }
            printf("smCounts succeed!\n");
        } break;
        default:
            printf("error optopt: %c\n", optopt);
            printf("error opterr: %d\n", opterr);
            break;
        }
    }

    printf("kernel number:%d\t", kernelnum);

    printf("\teach sm_to_kernel: ");
    int allsm = 0;
    for (int j = 0; j < kernelnum; j++) {
        allsm += smCounts[j];
        printf("%d  ", smCounts[j]);
    }
    printf("\tblocks_per_sm: %d\n", *block_per_sm);

    // if (allsm > device_sm_num) {
    //     printf("allocate sm number > device total sm number, exit!\n");
    //     exit(-1);
    // }

    return kernelnum;
}

int main(int argc, char* argv[]) {
    //初始化
    // cuInit(0);
    int            device = 0;
    cudaDeviceProp prop;
    int            sizecsv = 0;
    int            allnumblocks = 0;
    int            block_per_sm = 17;
    int*           smC;
    smC = (int*)malloc(sizeof(int) * block_per_sm);
    int patt = 1;
    cudaSetDevice(device);
    // printf("device:%d\n",device);
    cudaGetDeviceProperties(&prop, device);
    int clockRate = prop.clockRate;
    int sm_number = prop.multiProcessorCount;
    printf("*********   This GPU has %d SMs, clockRate is %d   *********\n", sm_number, clockRate);
    // output GPU prop
    const int CONTEXT_POOL_SIZE = init_getopt(argc, argv, smC, sm_number, &block_per_sm, &patt);

    // const int      CONTEXT_POOL_SIZE = 4;
    CUcontext contextPool[CONTEXT_POOL_SIZE];
    int       smCounts[CONTEXT_POOL_SIZE];
    int       numBlocks[CONTEXT_POOL_SIZE];
    for (int i = 0; i < CONTEXT_POOL_SIZE; i++) {
        smCounts[i] = smC[i];
    }
    free(smC);

    //获取当前设备的 COMPUTE_MODE
    CUresult err1;
    err1 = cuDeviceGetAttribute(&device, CU_DEVICE_ATTRIBUTE_COMPUTE_MODE, device);
    if (err1 != CUDA_SUCCESS) {
        printf("cuDeviceGetAttribute Error:%s\n", MyGetdeviceError(err1));
    }

    // save all output data
    DATATYPE** h_data;
    h_data = (DATATYPE**)malloc(sizeof(DATATYPE*) * CONTEXT_POOL_SIZE);

    DATATYPE** d_arr;
    d_arr = (DATATYPE**)malloc(sizeof(DATATYPE*) * CONTEXT_POOL_SIZE);

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

        // numBlocks[i] = 1 + smCounts[i] * block_per_sm;
        numBlocks[i] = smCounts[i] * block_per_sm;
        sizecsv += numBlocks[i] * DATA_OUT_NUM;
        allnumblocks += numBlocks[i];
        //为每个线程分配data数组
        h_data[i] = (DATATYPE*)malloc(sizeof(DATATYPE) * numBlocks[i] * DATA_OUT_NUM);
        d_arr[i] = (DATATYPE*)malloc(sizeof(DATATYPE) * 1024 * 1024);
    }

    char* filename;
    filename = (char*)malloc(sizeof(char) * (10 + 6 + 11 + 5 + 2 + 2 + sizeof(smCounts) + 2 + 2));
    gene_filename(filename, smCounts, block_per_sm, CONTEXT_POOL_SIZE, patt);
    // printf("\nfilename:%s\n", filename);

    //读写文件。文件存在则被截断为零长度，不存在则创建一个新文件
    FILE* fp = fopen(filename, "w+");
    if (fp == NULL) {
        printf("filename = %s \n", filename);
        fprintf(stderr, "fopen() failed.\n");
        exit(EXIT_FAILURE);
    }
    fprintf(fp, "KernelID,SMnum,Blocknum,BlockID,SMID,Start_time,End_time,Exec_time\n");
    fclose(fp);
    // printf("write file title success! \n");

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
                printf("Context %d parititioning SM error!\tPlan:%d\tactual:%d\n", step, smCounts[step], numSms);
                // cout<< "Context "<< step << " parititioning SM error!\tPlan:" <<
                // smCounts[step] << "\tactual:" << numSms << endl;
            } else {
                printf("Context %d parititioning SM success!\tPlan:%d\tactual:%d\n", step, smCounts[step], numSms);
            }
            if (step == CONTEXT_POOL_SIZE - 1)
                printf("KernelID\tBlockID\tSMID\tStart_time\tEnd_time\tExec_time\n");
            DATATYPE temp = 0;
            init_order(h_data[step], numBlocks[step], temp);

            main_test(step, numThreads, numBlocks, numSms, clockRate, h_data[step], patt,d_arr[step]);
        });

    for (step = 0; step < CONTEXT_POOL_SIZE; step++)
        mythread[step].join();

    cudaDeviceReset();

    //读写文件。文件不存在则创建新文件。读取会从文件的开头开始，写入则只能是追加模式
    fp = fopen(filename, "a+");
    if (fp == NULL) {
        fprintf(stderr, "fopen() failed.\n");
        exit(EXIT_FAILURE);
    }
    // DATATYPE min_time = 4294967000;
    // for (step = 0; step < CONTEXT_POOL_SIZE; step++) {
    //     for (int j = 0; j < numBlocks[step]; j++) {
    //         int index = j * DATA_OUT_NUM;
    //         if (min_time > h_data[step][index + 5]) {
    //             min_time = h_data[step][index + 5];
    //         }
    //     }
    // }
    // printf("min_time is %lf\n", min_time);

    for (step = 0; step < CONTEXT_POOL_SIZE; step++) {
        for (int j = 0; j < numBlocks[step]; j++) {
            int index = j * DATA_OUT_NUM;

            fprintf(fp, "%.0f,%.0f,%.0f,%.0f,%.0f,%.6f,%.6f,%.6f\n", h_data[step][index], h_data[step][index + 1], h_data[step][index + 2],
                    h_data[step][index + 3], h_data[step][index + 4], h_data[step][index + 5], h_data[step][index + 6], h_data[step][index + 6] - h_data[step][index + 5]);
            // fprintf(fp, "%lu,%lu,%lu,%lu,%lu,%lu,%lu,%lu\n", h_data[step][index], h_data[step][index + 1], h_data[step][index + 2],
            //         h_data[step][index + 3], h_data[step][index + 4], h_data[step][index + 5] - min_time, h_data[step][index + 6] - min_time, h_data[step][index + 6] - h_data[step][index + 5]);
        }
    }

    fclose(fp);
    free(filename);

    for (step = 0; step < CONTEXT_POOL_SIZE; step++) {
        free(h_data[step]);
        free(d_arr[step]);
    }
    printf("\nAll done.\n");
    free(h_data);
    free(d_arr);
    return 0;
}