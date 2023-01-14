#include "myutil.hpp"
#include "util.cuh"

#define N (1024 * 1024 * 4)
#define THREADS_PER_BLOCK 1024
#define COUNT 4

void serial_add(double *a, double *b, double *c, int n)
{
    for (size_t i = 0; i < COUNT; i++)
    {
        for (int index = 0; index < n; index++)
        {
            c[index] = a[index] * a[index] + b[index] * b[index];
        }
    }
}

__global__ void vector_add(double *a, double *b, double *c, int n)
{
    for (size_t i = 0; i < COUNT; i++)
    {
        int index = getThreadIdInBlock();
        if (index < n)
        {
            double a_data = a[index];
            double b_data = b[index];
            c[index] = a_data * a_data + b_data * b_data;
        }
    }
}

void compare(double *cpu, double *gpu, int n)
{
    for (int i = 0; i < n; i++)
        if (cpu[i] != gpu[i])
        {
            printf("ERROR: Comparision failed\n");
            return;
        }

    printf("Comparision success\n");
}

int main(int argc, char **argv)
{
    double *a, *b, *c_cpu, *c_gpu;
    int size = N * sizeof(double);

    a = (double *)malloc(size);
    b = (double *)malloc(size);
    c_cpu = (double *)malloc(size);
    c_gpu = (double *)malloc(size);

    for (int i = 0; i < N; i++)
    {
        a[i] = b[i] = i;
        c_cpu[i] = c_gpu[i] = 0;
    }

    serial_add(a, b, c_cpu, N);

    double *d_a, *d_b, *d_c;

    cudaMalloc((void **)&d_a, size);
    cudaMalloc((void **)&d_b, size);
    cudaMalloc((void **)&d_c, size);

    cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);

    vector_add<<<(N + (THREADS_PER_BLOCK - 1)) / THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(d_a, d_b, d_c, N);
    cudaDeviceSynchronize();
    cudaMemcpy(c_gpu, d_c, size, cudaMemcpyDeviceToHost);

    compare(c_cpu, c_gpu, N);
    
    free(a);
    free(b);
    free(c_cpu);
    free(c_gpu);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;
}