#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <jpeglib.h>

#define WIDTH 1024      // 输出图像宽度
#define HEIGHT 1024     // 输出图像高度
#define BLOCK_SIZE 32   // 每个block的线程数
#define MAX_POINTS 1024 // 最大点数

// 存储Voronoi图中的点的结构体
struct Point
{
    float x;
    float y;
};

// 存储输出图像中每个像素的颜色的结构体
struct Pixel
{
    int r;
    int g;
    int b;
};

// 存储每个block中shared memory的结构体
struct SharedMemory
{
    Point points[MAX_POINTS]; // 存储Voronoi图中的点
    int num_points;           // 存储点的数量
};

// 计算两个点之间的欧几里得距离
__device__ float distance(Point p1, Point p2)
{
    return sqrt((p1.x - p2.x) * (p1.x - p2.x) + (p1.y - p2.y) * (p1.y - p2.y));
}

// 计算指定像素点的颜色
__device__ Pixel getPixelColor(Point p, Point *points, int num_points)
{
    Pixel pixel;
    int min_distance = INT_MAX;
    for (int i = 0; i < num_points; i++)
    {
        float d = distance(p, points[i]);
        if (d < min_distance)
        {
            min_distance = d;
            pixel.r = (int)(255 * points[i].x / WIDTH);
            pixel.g = (int)(255 * points[i].y / HEIGHT);
            pixel.b = (int)(255 * (1 - (points[i].x + points[i].y) / (WIDTH + HEIGHT)));
        }
    }
    return pixel;
}

// 每个线程生成一个像素的颜色并写入图像
__global__ void generateVoronoi(Pixel *pixels, SharedMemory *shared_memory)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < WIDTH && y < HEIGHT)
    {
        Point p = {x + 0.5, y + 0.5};
        pixels[x + y * WIDTH] = getPixelColor(p, shared_memory->points, shared_memory->num_points);
    }
}

// 将Voronoi图中的点写入shared memory中
__device__ void loadPointsToSharedMemory(SharedMemory *shared_memory, Point *points, int num_points)
{
    shared_memory->num_points = num_points;
    for (int i = 0; i < num_points; i++)
    {
        shared_memory->points[i] = points[i];
    }
}

int main()
{
    // 随机生成Voronoi图中的点
    int num_points = 100;
    Point *points = (Point *)malloc(num_points * sizeof(Point));
    for (int i = 0; i < num_points; i++)
    {
        points[i].x = (float)(rand() % WIDTH);
        points[i].y = (float)(rand() % HEIGHT);
    }
    // 将点数据拷贝到device memory中
    Point *d_points;
    cudaMalloc(&d_points, num_points * sizeof(Point));
    cudaMemcpy(d_points, points, num_points * sizeof(Point), cudaMemcpyHostToDevice);

    // 初始化输出图像的数组
    Pixel *pixels = (Pixel *)malloc(WIDTH * HEIGHT * sizeof(Pixel));

    // 将输出图像数据拷贝到device memory中
    Pixel *d_pixels;
    cudaMalloc(&d_pixels, WIDTH * HEIGHT * sizeof(Pixel));

    // 初始化shared memory的大小
    SharedMemory *shared_memory;
    cudaMalloc(&shared_memory, sizeof(SharedMemory));

    // 启动kernel函数
    dim3 block_size(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid_size((WIDTH + BLOCK_SIZE - 1) / BLOCK_SIZE, (HEIGHT + BLOCK_SIZE - 1) / BLOCK_SIZE);
    loadPointsToSharedMemory<<<1, 1>>>(shared_memory, d_points, num_points);
    generateVoronoi<<<grid_size, block_size>>>(d_pixels, shared_memory);

    // 将输出图像数据拷贝回host memory中
    cudaMemcpy(pixels, d_pixels, WIDTH * HEIGHT * sizeof(Pixel), cudaMemcpyDeviceToHost);

    // 使用libjpeg库将输出图像保存为jpeg文件，并在图像上绘制点和边界线
    FILE *outfile = fopen("../src/voronoi/voronoi.jpg", "wb");
    if (outfile == NULL)
    {
        printf("Error: could not open output file\n");
        return 1;
    }
    struct jpeg_compress_struct cinfo;
    struct jpeg_error_mgr jerr;
    cinfo.err = jpeg_std_error(&jerr);
    jpeg_create_compress(&cinfo);
    jpeg_stdio_dest(&cinfo, outfile);
    cinfo.image_width = WIDTH;
    cinfo.image_height = HEIGHT;
    cinfo.input_components = 3;
    cinfo.in_color_space = JCS_RGB;
    jpeg_set_defaults(&cinfo);
    jpeg_start_compress(&cinfo, TRUE);
    JSAMPROW row_pointer;
    while (cinfo.next_scanline < cinfo.image_height)
    {
        row_pointer = (JSAMPROW)&pixels[cinfo.next_scanline * cinfo.image_width];
        jpeg_write_scanlines(&cinfo, &row_pointer, 1);
    }
    for (int i = 0; i < num_points; i++)
    {
        int x = (int)points[i].x;
        int y = (int)points[i].y;
        for (int j = x - 3; j <= x + 3; j++)
        {
            for (int k = y - 3; k <= y + 3; k++)
            {
                if (j >= 0 && j < WIDTH && k >= 0 && k < HEIGHT)
                {
                    int idx = j + k * WIDTH;
                    pixels[idx].r = 255;
                    pixels[idx].g = 255;
                    pixels[idx].b = 255;
                }
            }
        }
    }
    for (int i = 0; i < num_points; i++)
    {
        for (int j = i + 1; j < num_points; j++)
        {
            if (points[i].x == points[j].x && points[i].y == points[j].y)
            {
                continue;
            }
            // 计算Voronoi边界线
            float A = points[j].x - points[i].x;
            float B = points[j].y - points[i].y;
            float C = (points[i].x * points[i].x + points[i].y * points[i].y - points[j].x * points[j].x - points[j].y * points[j].y) / 2.0;
            for (int k = 0; k < WIDTH; k++)
            {
                int l = (int)round((-A * k - C) / B);
                if (l >= 0 && l < HEIGHT)
                {
                    int idx = k + l * WIDTH;
                    pixels[idx].r = 255;
                    pixels[idx].g = 0;
                    pixels[idx].b = 0;
                }
            }
        }
    }
    jpeg_finish_compress(&cinfo);
    fclose(outfile);
    // 释放device memory和host memory
    cudaFree(d_points);
    cudaFree(d_pixels);
    free(points);
    free(pixels);

    return 0;
}