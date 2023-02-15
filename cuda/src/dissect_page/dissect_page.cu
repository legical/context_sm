#include "myutil.hpp"
#include "util.cuh"

/*
kernel run: strige is GPU_PAGE, inner access times:inner_cycle    loop:times
sh: kernel run time
need to input: steps
*/

#define GPU_PAGE 4096   // default GPU page size is 4KB
#define INNER_LOOP 2048 // kernel inner loop times:2048, one loop has "steps" access

void getopt(int argc, char *argv[], int &inner_cycle, int &INDEX, char *filename)
{
    if (argc > 1)
    {
        inner_cycle = str_to_int(argv[1]);
    }
    if (argc > 2)
    {
        INDEX = str_to_int(argv[2]);
    }
    char path[96];
    getcwd(path, sizeof(path));
    /**
     * dirname(path): project root path
     * EXEC_TIMES: totally runing times 本次程序运行次数
     */
    if (argc > 3)
    {
        sprintf(filename, "%s/src/dissect_page/data-%s/Dissect-inner%d.csv",
                dirname(path), inner_cycle, argv[3]);
    }
    sprintf(filename, "%s/src/dissect_page/data-1070/Dissect-inner%d.csv",
            dirname(path), inner_cycle);
}

__global__ void dissect_page(unsigned int *my_array, int inner_cycle, int array_length, long long int *duration, unsigned int *index)
{
    // 2MB L2 cache : 512*32=16K
    const int it = 100;
    long long int start_time, end_time;
    unsigned int j = 0, k = 0;
    // const int it = 4096;

    __shared__ long long int s_tvalue;
    __shared__ unsigned int s_index[it];

    for (k = 0; k < it; k++)
    {
        s_index[k] = 0;
    }
    s_tvalue = 0;

    /* for loop inner_cycle times */
    start_time = clock64();
    for (k = 0; k < it; k++)
    {
        j = 0;
        for (int i = 0; i < inner_cycle; i++)
        {
            j = my_array[j];
            s_index[k] += j & k;
        }
        // printf("%d over.\n", k);
    }
    end_time = clock64();

    /* record execution time */
    s_tvalue = end_time - start_time;

    my_array[array_length] = j;
    my_array[array_length + 1] = my_array[j];

    for (k = 0; k < it; k++)
    {
        index[k] = s_index[k];
    }
    duration[0] = s_tvalue;
    // printf("s_tvalue is %d. \n", s_tvalue);
}

void measure_cache(int inner_cycle, int INDEX, char *filename)
{
    cudaDeviceReset();

    int i, stride = GPU_PAGE / sizeof(unsigned int);
    // Data_Size = sizeof(unsigned int) * N
    size_t Data_Size = inner_cycle * GPU_PAGE;
    int N = Data_Size / sizeof(unsigned int);
    size_t Array_Size = sizeof(unsigned int) * 2 + Data_Size;

    unsigned int *h_a;
    /* allocate arrays on CPU */
    h_a = (unsigned int *)malloc(Array_Size);
    unsigned int *d_a;
    /* allocate arrays on GPU */
    cudaMalloc((void **)&d_a, Array_Size);

    /* initialize array elements on CPU with pointers into d_a. */
    for (i = 0; i < N; i++)
    {
        // original:
        h_a[i] = (i + stride) % N;
    }

    h_a[N] = 0;
    h_a[N + 1] = 0;
    /* copy array elements from CPU to GPU */
    cudaMemcpy(d_a, h_a, Array_Size, cudaMemcpyHostToDevice);

    const int it = 100; // 512*100
    unsigned int *h_index = (unsigned int *)malloc(sizeof(unsigned int) * it);

    long long int *h_duration;
    h_duration = (long long int *)malloc(sizeof(long long int) * 1);
    long long int *duration;
    unsigned int *d_index;
    cudaMalloc((void **)&duration, sizeof(long long int) * 1);
    cudaMalloc((void **)&d_index, sizeof(unsigned int) * it);

    // printf("Starting running kernel, inner cycles %d * 100\n", inner_cycle);

    cudaThreadSynchronize();
    /* launch kernel*/
    dim3 Db = dim3(1);
    dim3 Dg = dim3(1, 1, 1);
    dissect_page<<<Dg, Db>>>(d_a, inner_cycle, N, duration, d_index);
    cudaThreadSynchronize();

    cudaError_t error_id = cudaGetLastError();
    if (error_id != cudaSuccess)
    {
        printf("Error kernel is %s\n", cudaGetErrorString(error_id));
    }

    /* copy results from GPU to CPU */
    cudaThreadSynchronize();
    cudaMemcpy((void *)h_index, (void *)d_index, sizeof(unsigned int) * it, cudaMemcpyDeviceToHost);
    cudaMemcpy((void *)h_duration, (void *)duration, sizeof(long long int) * 1, cudaMemcpyDeviceToHost);
    cudaThreadSynchronize();

    // printf("duration is %d. \n", h_duration[0]);

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
            exit(EXIT_FAILURE);
        }
        // 标题
        fprintf(fp, "Index,Exec_time,stride,inner_cycle,out_cycle,hit_rate\n");
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
        exit(EXIT_FAILURE);
    }

    // for (i = 0; i < it; i++)
    // {
    fprintf(fp, "%d,%lld,%d,%d,%d,", INDEX, h_duration[0], stride, inner_cycle, it);
    printf("%d\t %lld\n", INDEX, h_duration[0]);
    // }
    fclose(fp);

    /* free memory on GPU */
    cudaFree(d_a);
    cudaFree(d_index);
    cudaFree(duration);

    /*free memory on CPU */
    free(h_a);
    free(h_index);
    free(h_duration);

    cudaDeviceReset();
}

int main(int argc, char *argv[])
{
    // get L2 Cache Size
    size_t L2size = getL2CacheSize();
    // default inner_cycle: 恰好访问完整个L2 cache
    int inner_cycle = L2size / GPU_PAGE;
    int INDEX = 0;

    char *filename;
    filename = (char *)malloc(sizeof(char) * 256);
    // get input inner_cycle&filename
    getopt(argc, argv, inner_cycle, INDEX, filename);

    // Array size = inner_cycle * GPU_PAGE
    measure_cache(inner_cycle, INDEX, filename);

    free(filename);
    return 0;
}