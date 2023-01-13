#include "myutil.hpp"
#include "util.cuh"
int getopt(int argc, char *argv[], int &Index, int &EXEC_TIMES, int &ARR_SIZE, char *filename, int inter_cycle)
{
    if (argc > 1)
    {
        Index = str_to_int(argv[1]);

        if (argc > 2)
        {
            EXEC_TIMES = str_to_int(argv[2]);
            // if (argc > 3)
            // {
            //     ARR_SIZE = str_to_int(argv[3]);
            // }
        }
    }
    char path[96];
    getcwd(path, sizeof(path));
    /**
     * dirname(path): project root path
     * EXEC_TIMES: totally runing times 本次程序运行次数
     * inter_cycle: "for" loop times per kernel running 每次程序运行，kernel 中 for 循环（访存、计算）次数
     * argv[3]: usually date, to distinguish filename 通常是日期，用于区分文件名
     */
    if (argc > 3)
    {
        sprintf(filename, "%s/src/memory-fork/output/2-Ran%d-%d-%s.csv",
                dirname(path), EXEC_TIMES, inter_cycle, argv[3]);
    }
    else
    {
        sprintf(filename, "%s/src/memory-fork/output/2-Ran%d-%d.csv",
                dirname(path), EXEC_TIMES, inter_cycle);
    }
    return argc - 1;
}

__global__ void read_random_arr(int *arr_gpu, const int ARR_SIZE, const int inter_cycle)
{
    uint32_t threadid = getThreadIdInBlock();
    for (int j = 0; j < inter_cycle; j++)
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

__global__ void refresh_L2(int *l2_gpu, const int L2size, const int random_l2_num)
{
    uint32_t threadid = getThreadIdInBlock();
    int i = threadid;
    // #pragma unroll
    while (i < L2size)
    {
        l2_gpu[i] = i + random_l2_num;
        i += 32;
    }
}

int main(int argc, char *argv[])
{
    // Default: array size = 1GB
    int Index = 0, ARR_SIZE = 1024 * 1024 * 256, EXEC_TIMES = 1000, inter_cycle = 8;
    // 获取文件名
    char *filename;
    filename = (char *)malloc(sizeof(char) * 128);
    // get option
    int para_num = getopt(argc, argv, Index, EXEC_TIMES, ARR_SIZE, filename, inter_cycle);
    // printf("You have entered %d parameter.\n", para_num);
    // printf("ARR_SIZE: %d\n", ARR_SIZE);

    int *arr, *arr_gpu, *l2, *l2_gpu;
    float elapsedTime;
    cudaEvent_t start, stop;

    // get GPU L2 cache size
    int device_id = 0;
    cudaDeviceProp prop;
    cudaSetDevice(device_id);
    gpuErrAssert(cudaGetDeviceProperties(&prop, device_id));
    int random_l2_num = get_random_num(2, 6);
    size_t L2size = prop.l2CacheSize * random_l2_num;

    // allocate pinned memory in system memory
    gpuErrAssert(cudaHostAlloc((void **)&arr,
                               ARR_SIZE * sizeof(int),
                               cudaHostAllocDefault));
    // allocate pinned memory to refresh GPU L2 cache
    gpuErrAssert(cudaHostAlloc((void **)&l2,
                               L2size * sizeof(int),
                               cudaHostAllocDefault));
    init_chase_arr<int>(arr, ARR_SIZE, 1);
    init_arr<int>(l2, L2size, 0);

    // allocate & copy L2 cache refresh memory
    gpuErrAssert(cudaMalloc((void **)&l2_gpu, L2size * sizeof(int)));
    gpuErrAssert(cudaMemcpy(l2_gpu, l2, L2size * sizeof(int), cudaMemcpyHostToDevice));
    // refresh L2 cache first
    refresh_L2<<<1, 32>>>(l2_gpu, L2size, random_l2_num);

    // end - start = exection time
    gpuErrAssert(cudaEventCreate(&start));
    gpuErrAssert(cudaEventCreate(&stop));

    gpuErrAssert(cudaMalloc((void **)&arr_gpu, ARR_SIZE * sizeof(int)));

    // copy random memory from host to gpu
    gpuErrAssert(cudaMemcpy(arr_gpu, arr, ARR_SIZE * sizeof(int), cudaMemcpyHostToDevice));

    // record cuda start time
    gpuErrAssert(cudaEventRecord(start, 0));
    // run kernel for random GPU memory access
    read_random_arr<<<1, 32>>>(arr_gpu, ARR_SIZE, inter_cycle);
    // record cuda sop time
    gpuErrAssert(cudaEventRecord(stop, 0));
    // Synchronize
    gpuErrAssert(cudaEventSynchronize(stop));
    // caculate cuda execution time & save it
    gpuErrAssert(cudaEventElapsedTime(&elapsedTime,
                                      start, stop));
    gpuErrAssert(cudaEventDestroy(start));
    gpuErrAssert(cudaEventDestroy(stop));

    // copy back random memory from gpu to host
    gpuErrAssert(cudaMemcpy(arr, arr_gpu, ARR_SIZE * sizeof(int), cudaMemcpyDeviceToHost));

    // char *run_dir = "memory-fork";
    // GetBaseFilename(filename, EXEC_TIMES, run_dir);

    // 如果输出文件不存在，则创建文件并写入标题
    if (!isFileExists(filename))
    {
        // 读写文件。文件存在则追加写入，不存在则创建一个新文件
        FILE *fp = fopen(filename, "a+");
        // 如果打开文件失败
        if (fp == NULL)
        {
            // std::cout << "Can't open file : " << filename << std::endl;
            printf("Can't open file : %s \n", filename);
            fprintf(stderr, "fopen() failed.\n");
            free(filename);
            exit(EXIT_FAILURE);
        }
        // 标题
        fprintf(fp, "Index,Exec_time,addr,GPU_addr\n");
        fclose(fp);
    }

    // 读写文件。文件存在则追加写入，不存在则创建一个新文件
    FILE *fp = fopen(filename, "a+");
    // 如果打开文件失败
    if (fp == NULL)
    {
        // std::cout << "Can't open file : " << filename << std::endl;
        printf("Can't open file : %s \n", filename);
        fprintf(stderr, "fopen() failed.\n");
        free(filename);
        exit(EXIT_FAILURE);
    }
    // elapsedTime ms; elapsedTime*1000 μS
    fprintf(fp, "%d,%f,%p,%p\n", Index, elapsedTime, arr, arr_gpu);
    printf("%d. Exec_time: %.6fms \t arr_addr:%p\t GPU_addr:%p\n", Index, elapsedTime, arr, arr_gpu);
    fclose(fp);

    // 输出运行时间信息
    // printf("min/max/avg = %.6f / %.6f / %.6f ms\n", min, max, avg / (float)EXEC_TIMES);
    // printf("All exection data has stored into %s.\n", filename);

    gpuErrAssert(cudaFree(arr_gpu));
    gpuErrAssert(cudaFreeHost(arr));
    gpuErrAssert(cudaFreeHost(l2));
    // gpuErrAssert(cudaFree(arr_gpu));
    gpuErrAssert(cudaFree(l2_gpu));
    free(filename);

    return 0;
}