#include "myutil.hpp"
#include "util.cuh"
int getopt(int argc, char *argv[], int &EXEC_TIMES, int &ARR_SIZE)
{
    if (argc > 1)
    {
        EXEC_TIMES = str_to_int(argv[1]);

        if (argc > 2)
        {
            ARR_SIZE = str_to_int(argv[2]);
        }
    }
    return argc - 1;
}

__global__ void read_random_arr(int *arr_gpu, const int ARR_SIZE)
{
    uint32_t threadid = getThreadIdInBlock();
    int i = threadid;
#pragma unroll
    while (i < ARR_SIZE)
    {
        i = arr_gpu[i] - arr_gpu[0] + 32;
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
    // Default: execute 1000 times, array size = 1GB
    int EXEC_TIMES = 1000, ARR_SIZE = 1024 * 1024 * 1024;
    // get option
    int para_num = getopt(argc, argv, EXEC_TIMES, ARR_SIZE);
    printf("You have entered %d parameter.\n", para_num);
    printf("EXEC_TIMES: %d \t ARR_SIZE: %d\n", EXEC_TIMES, ARR_SIZE);

    int *arr, *l2, *l2_gpu, random_num_arr[EXEC_TIMES];
    float elapsedTime[EXEC_TIMES];

    // get GPU L2 cache size
    int device_id = 0;
    cudaDeviceProp prop;
    cudaSetDevice(device_id);
    gpuErrAssert(cudaGetDeviceProperties(&prop, device_id));
    size_t L2size = prop.l2CacheSize;

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

    // random number < 3/4 ARR_SIZE
    // copy array size = 1/4 ARR_SIZE * int
    int random_limit = ARR_SIZE / sizeof(int) * 3;

    for (int i = 0; i < EXEC_TIMES; i++)
    {
        // refresh L2 cache first
        int random_l2_num = get_random_num(0, 32);
        refresh_L2<<<1, 32>>>(l2_gpu, L2size, random_l2_num);

        cudaEvent_t start, stop;
        // end - start = exection time
        gpuErrAssert(cudaEventCreate(&start));
        gpuErrAssert(cudaEventCreate(&stop));
        int *arr_gpu;

        gpuErrAssert(cudaMalloc((void **)&arr_gpu, ARR_SIZE));
        // generate random number
        int random_num = get_random_num(0, random_limit);

        // record cuda start time
        gpuErrAssert(cudaEventRecord(start, 0));
        // copy random memory from host to gpu
        gpuErrAssert(cudaMemcpy(arr_gpu, arr + random_num, ARR_SIZE, cudaMemcpyHostToDevice));

        // run kernel for random GPU memory access
        read_random_arr<<<1, 32>>>(arr_gpu, ARR_SIZE);

        // copy back random memory from gpu to host
        gpuErrAssert(cudaMemcpy(arr + random_num, arr_gpu, ARR_SIZE, cudaMemcpyDeviceToHost));
        gpuErrAssert(cudaFree(arr_gpu));

        // record cuda sop time
        gpuErrAssert(cudaEventRecord(stop, 0));
        // Synchronize
        gpuErrAssert(cudaEventSynchronize(stop));
        // caculate cuda execution time & save it
        gpuErrAssert(cudaEventElapsedTime(&elapsedTime[i],
                                          start, stop));
        gpuErrAssert(cudaEventDestroy(start));
        gpuErrAssert(cudaEventDestroy(stop));

        // 保存每一次生成的random_num
        random_num_arr[i] = random_num;

        printf("Run for the %d time, the execution time is %.6f ms.\n", i + 1, elapsedTime[i]);
        // cudaDeviceSynchronize();
    }

    gpuErrAssert(cudaFreeHost(arr));
    gpuErrAssert(cudaFreeHost(l2));
    // gpuErrAssert(cudaFree(arr_gpu));
    gpuErrAssert(cudaFree(l2_gpu));

    char *filename;
    filename = (char *)malloc(sizeof(char) * 128);
    // char *run_dir = "memory";
    GetBaseFilename(filename, EXEC_TIMES);
    // 读写文件。文件存在则被截断为零长度，不存在则创建一个新文件
    FILE *fp = fopen(filename, "w+");
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
    fprintf(fp, "ID,Exec_time,Random_num\n");
    // 导入数据
    float min = elapsedTime[0], max = elapsedTime[0], avg = 0.0;
    for (int i = 0; i < EXEC_TIMES; i++)
    {
        fprintf(fp, "%d,%.6f,%d\n", i + 1, elapsedTime[i], random_num_arr[i]);
        if (min > elapsedTime[i])
        {
            min = elapsedTime[i];
        }

        if (max < elapsedTime[i])
        {
            max = elapsedTime[i];
        }

        avg += elapsedTime[i];
    }
    fclose(fp);
    // 输出运行时间信息
    printf("min/max/avg = %.6f / %.6f / %.6f ms\n", min, max, avg / (float)EXEC_TIMES);
    printf("All exection data has stored into %s.\n", filename);

    free(filename);

    return 0;
}
