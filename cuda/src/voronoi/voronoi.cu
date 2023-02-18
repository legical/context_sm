#include <iostream>
#include <cstdlib>
#include <cmath>
#include <ctime>

// 这个代码生成了一个随机着色的Voronoi图，并将其保存为JPG文件
// 在此代码中，我们使用了CUDA中的curand库来生成随机颜色，并使用了libjpeg库来保存图像文件

#ifdef _WIN32 // 判断操作系统是否为 Windows
#include <windows.h>
#else
#include <sys/time.h>
#endif

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

// 在 Ubuntu 上：
// 打开终端并运行以下命令以安装 libjpeg 库：sudo apt-get install libjpeg-dev
#include "jpeglib.h"

const int WIDTH = 1024;    // 图像的宽度
const int HEIGHT = 1024;   // 图像的高度
const int NUM_POINTS = 50; // 随机点的个数

// CUDA 核函数，计算每个像素点的颜色
__global__ void calculate_pixel_colors(int *pixels, int width, int height, int *x, int *y, int num_points)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;

    if (i < width && j < height)
    {
        // 找出距离当前像素最近的随机点
        int closest_point_index = 0;
        int closest_distance = INT_MAX;
        for (int k = 0; k < num_points; k++)
        {
            int distance = (i - x[k]) * (i - x[k]) + (j - y[k]) * (j - y[k]);
            if (distance < closest_distance)
            {
                closest_point_index = k;
                closest_distance = distance;
            }
        }

        // 使用随机点的颜色填充当前像素
        int r = (closest_point_index * 107 % 255);
        int g = (closest_point_index * 197 % 255);
        int b = (closest_point_index * 37 % 255);
        pixels[j * width + i] = r << 16 | g << 8 | b;
    }
}

// 保存像素数组为 JPEG 图像
void save_jpeg_image(const int *pixels, int width, int height, const char *filename)
{
    // 定义并初始化 libjpeg 的结构体和错误管理对象
    struct jpeg_compress_struct cinfo;
    struct jpeg_error_mgr jerr;
    cinfo.err = jpeg_std_error(&jerr);
    jpeg_create_compress(&cinfo);

// 打开输出文件
#ifdef _WIN32 // 判断操作系统是否为 Windows
    FILE *outfile;
    if ((outfile = fopen(filename, "wb")) == NULL)
    {
        std::cerr << "Cannot open file " << filename << std::endl;
        exit(1);
    }
#else
    FILE *outfile = fopen(filename, "wb");
    if (outfile == NULL)
    {
        std::cerr << "Cannot open file " << filename << std::endl;
        exit(1);
    }
#endif

    // 设置 JPEG 压缩参数
    jpeg_stdio_dest(&cinfo, outfile);
    cinfo.image_width = width;
    cinfo.image_height = height;
    cinfo.input_components = 3;
    cinfo.in_color_space = JCS_RGB;
    jpeg_set_defaults(&cinfo);
    jpeg_set_quality(&cinfo, 90, TRUE);

    // 开始压缩图像
    jpeg_start_compress(&cinfo, TRUE);

    // 定义一个指向一行像素数据的指针
    JSAMPROW row_pointer;

    // 循环压缩每一行
    while (cinfo.next_scanline < cinfo.image_height)
    {
        // 设置当前行的指针
        row_pointer = (JSAMPROW)&pixels[cinfo.next_scanline * width];

        // 压缩当前行
        jpeg_write_scanlines(&cinfo, &row_pointer, 1);
    }

    // 完成图像压缩并关闭文件
    jpeg_finish_compress(&cinfo);
    fclose(outfile);
    jpeg_destroy_compress(&cinfo);
}

int main()
{
    // 初始化随机数生成器
    srand(time(NULL));

    // 定义随机点数组
    int x[NUM_POINTS];
    int y[NUM_POINTS];

    // 生成随机点
    for (int i = 0; i < NUM_POINTS; i++)
    {
        x[i] = rand() % WIDTH;
        y[i] = rand() % HEIGHT;
    }

    // 分配显存空间
    int *dev_x, *dev_y, *dev_pixels;
    cudaMalloc(&dev_x, NUM_POINTS * sizeof(int));
    cudaMalloc(&dev_y, NUM_POINTS * sizeof(int));
    cudaMalloc(&dev_pixels, WIDTH * HEIGHT * sizeof(int));

    // 将随机点传输到显存中
    cudaMemcpy(dev_x, x, NUM_POINTS * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_y, y, NUM_POINTS * sizeof(int), cudaMemcpyHostToDevice);

    // 计算线程块和线程数
    dim3 num_threads_per_block(16, 16);
    dim3 num_blocks((WIDTH + num_threads_per_block.x - 1) / num_threads_per_block.x,
                    (HEIGHT + num_threads_per_block.y - 1) / num_threads_per_block.y);

    // 调用 CUDA 核函数，计算每个像素点的颜色
    calculate_pixel_colors<<<num_blocks, num_threads_per_block>>>(dev_pixels, WIDTH, HEIGHT, dev_x, dev_y, NUM_POINTS);

    // 将像素数据传回主机内存
    int *pixels = new int[WIDTH * HEIGHT];
    cudaMemcpy(pixels, dev_pixels, WIDTH * HEIGHT * sizeof(int), cudaMemcpyDeviceToHost);

    // 释放显存空间
    cudaFree(dev_x);
    cudaFree(dev_y);
    cudaFree(dev_pixels);

    // 保存图像
    save_jpeg_image(pixels, WIDTH, HEIGHT, "voronoi.jpg");

    // 释放主机内存空间
    delete[] pixels;

    return 0;
}
