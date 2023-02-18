#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>
#include <time.h>
#include "jpeglib.h"

#define WIDTH 1024
#define HEIGHT 1024
#define NUM_POINTS 1024
#define BLOCK_SIZE 16

// 定义点结构体
typedef struct
{
    float x, y;
    int index;
} Point;

// 定义颜色结构体
typedef struct
{
    int r, g, b;
} Color;

// 定义生成Voronoi图的CUDA内核
__global__ void voronoi(Point *points, Color *colors, unsigned char *image)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int minIndex = 0;
    float minDistance = sqrtf(WIDTH * WIDTH + HEIGHT * HEIGHT);
    for (int i = 0; i < NUM_POINTS; i++)
    {
        float dx = points[i].x - x;
        float dy = points[i].y - y;
        float distance = sqrtf(dx * dx + dy * dy);
        if (distance < minDistance)
        {
            minDistance = distance;
            minIndex = i;
        }
    }
    Color color = colors[minIndex];
    int index = (y * WIDTH + x) * 3;
    image[index] = color.r;
    image[index + 1] = color.g;
    image[index + 2] = color.b;
}

int main()
{
    // 初始化CUDA设备
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);
    printf("Using device %s\n", deviceProp.name);
    cudaSetDevice(0);

    // 分配设备内存
    Point *d_points;
    Color *d_colors;
    unsigned char *d_image;
    cudaMalloc(&d_points, sizeof(Point) * NUM_POINTS);
    cudaMalloc(&d_colors, sizeof(Color) * NUM_POINTS);
    cudaMalloc(&d_image, sizeof(unsigned char) * WIDTH * HEIGHT * 3);

    // 初始化随机数生成器
    curandState *d_state;
    cudaMalloc(&d_state, sizeof(curandState) * NUM_POINTS);
    srand(time(NULL));
    initCurand<<<NUM_POINTS / BLOCK_SIZE, BLOCK_SIZE>>>(d_state, time(NULL));

    // 初始化点和颜色
    Point *points = (Point *)malloc(sizeof(Point) * NUM_POINTS);
    Color *colors = (Color *)malloc(sizeof(Color) * NUM_POINTS);
    for (int i = 0; i < NUM_POINTS; i++)
    {
        points[i].x = curand_uniform(&d_state[i]) * WIDTH;
        points[i].y = curand_uniform(&d_state[i]) * HEIGHT;
        points[i].index = i;
        colors[i].r = rand() % 256;
        colors[i].g = rand() % 256;
        colors[i].b = rand() % 256;
    }

    // 将点和颜色拷贝到设备
    cudaMemcpy(d_points, points, sizeof(Point) * NUM_POINTS, cudaMemcpyHostToDevice);
    cudaMemcpy(d_colors, colors, sizeof(Color) * NUM_POINTS, cudaMemcpyHostToDevice);

    // 调用CUDA内核生成Voronoi图
    dim3 blocks(WIDTH / BLOCK_SIZE, HEIGHT / BLOCK_SIZE);
    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
    voronoi<<<blocks, threads>>>(d_points, d_colors, d_image);
    cudaDeviceSynchronize();
    // 将结果拷贝回主机
    unsigned char *image = (unsigned char *)malloc(sizeof(unsigned char) * WIDTH * HEIGHT * 3);
    cudaMemcpy(image, d_image, sizeof(unsigned char) * WIDTH * HEIGHT * 3, cudaMemcpyDeviceToHost);

    // 保存Voronoi图为JPG文件
    struct jpeg_compress_struct cinfo;
    struct jpeg_error_mgr jerr;
    FILE *outfile;
    JSAMPROW row_pointer[1];
    cinfo.err = jpeg_std_error(&jerr);
    jpeg_create_compress(&cinfo);
    if ((outfile = fopen("../src/voronoi/voronoi2.jpg", "wb")) == NULL)
    {
        fprintf(stderr, "Can't open output file\n");
        exit(1);
    }
    jpeg_stdio_dest(&cinfo, outfile);
    cinfo.image_width = WIDTH;
    cinfo.image_height = HEIGHT;
    cinfo.input_components = 3;
    cinfo.in_color_space = JCS_RGB;
    jpeg_set_defaults(&cinfo);
    jpeg_start_compress(&cinfo, TRUE);
    while (cinfo.next_scanline < cinfo.image_height)
    {
        row_pointer[0] = &image[cinfo.next_scanline * cinfo.image_width * 3];
        jpeg_write_scanlines(&cinfo, row_pointer, 1);
    }
    jpeg_finish_compress(&cinfo);
    fclose(outfile);

    // 释放内存
    free(points);
    free(colors);
    free(image);
    cudaFree(d_points);
    cudaFree(d_colors);
    cudaFree(d_image);
    cudaFree(d_state);

    return 0;
}