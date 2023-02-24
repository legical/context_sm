#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "jpeglib.h"

#define BLOCK_SIZE 32 // 线程块大小
#define RADIUS 10     // 每个像素点的影响范围

typedef struct
{
    float x;
    float y;
} Point;

__global__ void compute_voronoi(Point *sites, int site_num, int *output, int width, int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x; // 计算当前线程处理的像素横坐标
    int y = blockIdx.y * blockDim.y + threadIdx.y; // 计算当前线程处理的像素纵坐标

    if (x >= width || y >= height)
    {
        return; // 超出图像范围的像素不进行计算
    }

    int index = y * width + x;          // 当前像素在输出数组中的索引
    int closest_site_index = 0;         // 记录离当前像素最近的点的索引
    float closest_site_dist = INFINITY; // 记录当前像素距离最近的点的距离

    __shared__ Point shared_sites[BLOCK_SIZE * BLOCK_SIZE]; // 定义shared memory，每个线程块都共享一份
    for (int i = 0; i < site_num; i += blockDim.x * blockDim.y)
    {                                                                       // 将所有点分成若干个组，每个组中的点数为线程块大小
        int group_start_index = i + threadIdx.y * blockDim.x + threadIdx.x; // 计算当前线程在该组中处理的点的索引
        if (group_start_index < site_num)
        {                                                                   // 如果该线程在当前组中处理的点存在
            shared_sites[group_start_index - i] = sites[group_start_index]; // 将该点从全局内存中读入shared memory
        }
        __syncthreads(); // 等待该线程块中所有线程都将自己负责的点读入shared memory

        for (int j = 0; j < BLOCK_SIZE * BLOCK_SIZE && i + j < site_num; j++)
        { // 遍历当前组中的所有点，计算当前像素到该点的距离
            float dx = x - shared_sites[j].x;
            float dy = y - shared_sites[j].y;
            float dist = dx * dx + dy * dy;
            if (dist < closest_site_dist)
            { // 如果当前点距离更近，则更新最近点的索引和距离
                closest_site_index = i + j;
                closest_site_dist = dist;
            }
        }
        __syncthreads(); // 等待该线程块中所有线程都完成对shared memory的读取和计算，以便下一组点的读入
    }

    output[index] = closest_site_index; // 将离当前像素最近的点的索引存入输出数组中
}

// 该函数接受三个参数：指向图像数据的指针data、图像宽度width和图像高度height
// 在函数中，首先使用libjpeg库创建一个JPEG压缩对象，并指定输出文件名和输出流，然后设置图像的宽度、高度、像素格式等参数
// 接着循环遍历每一行像素，将RGB值存储到一个临时的image_buffer中
// 最后将image_buffer中的数据写入JPEG压缩对象中，直到处理完所有行，然后完成压缩过程，关闭输出文件并释放相关资源
void jpeg_compress(int* data, int width, int height) {
    struct jpeg_compress_struct cinfo;
    struct jpeg_error_mgr jerr;

    JSAMPROW row_pointer[1];
    int row_stride;

    cinfo.err = jpeg_std_error(&jerr);
    jpeg_create_compress(&cinfo);

    char filename[] = "../src/voronoi/voronoi-sm.jpg"; // 输出文件名
    FILE* outfile;
    if ((outfile = fopen(filename, "wb")) == NULL) {
        fprintf(stderr, "can't open %s\n", filename);
        exit(1);
    }
    jpeg_stdio_dest(&cinfo, outfile);

    cinfo.image_width = width;
    cinfo.image_height = height;
    cinfo.input_components = 3;
    cinfo.in_color_space = JCS_RGB;
    jpeg_set_defaults(&cinfo);
    jpeg_start_compress(&cinfo, TRUE);

    row_stride = width * 3;
    JSAMPLE* image_buffer = (JSAMPLE*)malloc(row_stride);
    while (cinfo.next_scanline < cinfo.image_height) {
        for (int i = 0; i < width; i++) {
            int index = (cinfo.image_height - cinfo.next_scanline - 1) * width + i;
            int color = data[index];
            int r = (color >> 16) & 0xff;
            int g = (color >> 8) & 0xff;
            int b = color & 0xff;
            image_buffer[i * 3] = (JSAMPLE)r;
            image_buffer[i * 3 + 1] = (JSAMPLE)g;
            image_buffer[i * 3 + 2] = (JSAMPLE)b;
        }
        row_pointer[0] = image_buffer;
        jpeg_write_scanlines(&cinfo, row_pointer, 1);
    }

    jpeg_finish_compress(&cinfo);
    fclose(outfile);
    jpeg_destroy_compress(&cinfo);
    free(image_buffer);
}

int main()
{
    int width = 1024;                                         // 图像宽度
    int height = 1024;                                        // 图像高度
    int site_num = 100;                                       // 点的数量
    Point *sites = (Point *)malloc(site_num * sizeof(Point)); // 生成随机点
    for (int i = 0; i < site_num; i++)
    {
        sites[i].x = (float)(rand() % width);
        sites[i].y = (float)(rand() % height);
    }
    int *output = (int *)malloc(width * height * sizeof(int)); // 分配输出数组的空间
    Point *d_sites;                                            // 在device上分配点的空间
    cudaMalloc(&d_sites, site_num * sizeof(Point));
    cudaMemcpy(d_sites, sites, site_num * sizeof(Point), cudaMemcpyHostToDevice);

    int *d_output; // 在device上分配输出数组的空间
    cudaMalloc(&d_output, width * height * sizeof(int));
    cudaMemset(d_output, -1, width * height * sizeof(int)); // 初始化输出数组为-1

    dim3 block_size(BLOCK_SIZE, BLOCK_SIZE);                                                               // 定义线程块大小
    dim3 grid_size((width + block_size.x - 1) / block_size.x, (height + block_size.y - 1) / block_size.y); // 计算线程块数量

    compute_voronoi<<<grid_size, block_size>>>(d_sites, site_num, d_output, width, height); // 调用kernel函数计算Voronoi图

    cudaMemcpy(output, d_output, width * height * sizeof(int), cudaMemcpyDeviceToHost); // 将输出数组从device复制到host

    jpeg_compress(output, width, height); // 调用自己实现的jpeg_compress函数保存图像

    free(sites); // 释放内存
    free(output);
    cudaFree(d_sites);
    cudaFree(d_output);

    return 0;
}