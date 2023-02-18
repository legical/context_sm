#include <iostream>
#include <cstdlib>
#include <ctime>
#include <random>
#include <cmath>
#include <jpeglib.h>

// 定义常量
const int WIDTH = 1024;
const int HEIGHT = 1024;
const int NUM_POINTS = 10000;
const int BLOCK_SIZE = 16;

// 生成Voronoi图的CUDA核
__global__ void voronoi(float2 *points, float3 *colors, unsigned char *image)
{
    // 获取当前线程对应的像素坐标
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    if (x >= WIDTH || y >= HEIGHT)
    {
        return;
    }

    // 计算当前像素到每个点的距离，并找到最近的点
    float min_distance = INFINITY;
    int min_index = -1;
    for (int i = 0; i < NUM_POINTS; i++)
    {
        float2 point = points[i];
        float distance = sqrtf((point.x - x) * (point.x - x) + (point.y - y) * (point.y - y));
        if (distance < min_distance)
        {
            min_distance = distance;
            min_index = i;
        }
    }

    // 将当前像素设置为最近点的颜色
    float3 color = colors[min_index];
    int index = (y * WIDTH + x) * 3;
    image[index] = (unsigned char)(color.x * 255);
    image[index + 1] = (unsigned char)(color.y * 255);
    image[index + 2] = (unsigned char)(color.z * 255);
}

int main()
{
    // 初始化随机数生成器
    std::default_random_engine generator(time(nullptr));
    std::uniform_real_distribution<float> distribution(0.0f, 1.0f);

    // 分配内存并初始化点和颜色数组
    float2 *points = (float2 *)malloc(sizeof(float2) * NUM_POINTS);
    float3 *colors = (float3 *)malloc(sizeof(float3) * NUM_POINTS);
    for (int i = 0; i < NUM_POINTS; i++)
    {
        points[i].x = distribution(generator) * WIDTH;
        points[i].y = distribution(generator) * HEIGHT;
        colors[i].x = distribution(generator);
        colors[i].y = distribution(generator);
        colors[i].z = distribution(generator);
    }

    // 在GPU上分配内存
    float2 *d_points;
    float3 *d_colors;
    unsigned char *d_image;
    cudaMalloc((void **)&d_points, sizeof(float2) * NUM_POINTS);
    cudaMalloc((void **)&d_colors, sizeof(float3) * NUM_POINTS);
    cudaMalloc((void **)&d_image, sizeof(unsigned char) * WIDTH * HEIGHT * 3);

    // 将数据拷贝到GPU
    cudaMemcpy(d_points, points, sizeof(float2) * NUM_POINTS, cudaMemcpyHostToDevice);
    cudaMemcpy(d_colors, colors, sizeof(float3) * NUM_POINTS, cudaMemcpyHostToDevice);

    // 设置CUDA内核的网格和块大小
    dim3 gridSize((WIDTH + BLOCK_SIZE - 1) / BLOCK_SIZE, (HEIGHT + BLOCK_SIZE - 1) / BLOCK_SIZE);
    dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);

    // 调用Voronoi图生成核函数
    voronoi<<<gridSize, blockSize>>>(d_points, d_colors, d_image);

    // 将结果拷贝回主机
    unsigned char *image = (unsigned char *)malloc(sizeof(unsigned char) * WIDTH * HEIGHT * 3);
    cudaMemcpy(image, d_image, sizeof(unsigned char) * WIDTH * HEIGHT * 3, cudaMemcpyDeviceToHost);

    // 保存结果为JPEG文件
    struct jpeg_compress_struct cinfo;
    struct jpeg_error_mgr jerr;
    cinfo.err = jpeg_std_error(&jerr);
    jpeg_create_compress(&cinfo);
    FILE *outfile;
    if ((outfile = fopen("voronoi.jpg", "wb")) == NULL)
    {
        std::cout << "Error opening output JPEG file!" << std::endl;
        return 1;
    }
    jpeg_stdio_dest(&cinfo, outfile);
    cinfo.image_width = WIDTH;
    cinfo.image_height = HEIGHT;
    cinfo.input_components = 3;
    cinfo.in_color_space = JCS_RGB;
    jpeg_set_defaults(&cinfo);
    jpeg_start_compress(&cinfo, TRUE);
    JSAMPROW row_pointer[1];
    while (cinfo.next_scanline < cinfo.image_height)
    {
        row_pointer[0] = &image[cinfo.next_scanline * WIDTH * 3];
        jpeg_write_scanlines(&cinfo, row_pointer, 1);
    }
    jpeg_finish_compress(&cinfo);
    fclose(outfile);
    jpeg_destroy_compress(&cinfo);

    // 释放内存
    free(points);
    free(colors);
    free(image);
    cudaFree(d_points);
    cudaFree(d_colors);
    cudaFree(d_image);

    return 0;
}
