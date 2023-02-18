#include <iostream>
#include <cstdlib>
#include <ctime>
#include <cmath>
#include <algorithm>
#include <chrono>

#ifdef _WIN32 // 判断操作系统是否为 Windows
#include <windows.h>
#include <wingdi.h>
#include <csetjmp>
#include <jpeglib.h>
#else
#include <jpeglib.h>
#include <sys/time.h>
#endif

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// 定义宏，用于检查 CUDA 函数是否出错
#define CHECK_CUDA_ERROR(call)                                                                                \
    {                                                                                                         \
        cudaError_t error = call;                                                                             \
        if (error != cudaSuccess)                                                                             \
        {                                                                                                     \
            std::cerr << "CUDA error: " << cudaGetErrorString(error) << " at line " << __LINE__ << std::endl; \
            exit(1);                                                                                          \
        }                                                                                                     \
    }

// 定义宏，用于计算 CUDA 核函数的线程块数量
#define CALCULATE_NUM_BLOCKS(num_threads, threads_per_block) ( \
    (num_threads + threads_per_block - 1) / threads_per_block)

// 定义结构体，用于表示 2D 向量
struct Vector2D
{
    float x, y;

    __host__ __device__ Vector2D() {}

    __host__ __device__ Vector2D(float x, float y)
    {
        this->x = x;
        this->y = y;
    }

    // 计算两个向量的距离
    __host__ __device__ float distance(const Vector2D &other) const
    {
        float dx = x - other.x;
        float dy = y - other.y;
        return std::sqrt(dx * dx + dy * dy);
    }
};

// CUDA 核函数，用于生成 Voronoi 图
__global__ void generate_voronoi_map(const Vector2D *points, int num_points,
                                     int width, int height, int *output)
{
    // 计算像素的 x 和 y 坐标
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    // 检查坐标是否在图像范围内
    if (x >= width || y >= height)
    {
        return;
    }

    // 将像素坐标转换为向量坐标
    Vector2D pixel(static_cast<float>(x), static_cast<float>(y));

    // 找到距离当前像素最近的种子点
    float min_distance = std::numeric_limits<float>::max();
    int closest_point = -1;
    for (int i = 0; i < num_points; i++)
    {
        float distance = points[i].distance(pixel);
        if (distance < min_distance)
        {
            min_distance = distance;
            closest_point = i;
        }
    }

    // 将像素的颜色设置为最近种子点的颜色
    output[y * width + x] = closest_point;
}

// 保存 Voronoi 图为 JPEG 图像
void save_jpeg_image(int *pixels, int width, int height, const char *filename)
{
    struct jpeg_compress_struct cinfo;
    // 分配内存
    JSAMPROW row_pointer[1];
    unsigned char *row_buffer = new unsigned char[width * 3];
    cinfo.image_width = width;
    cinfo.image_height = height;
    cinfo.input_components = 3;
    cinfo.in_color_space = JCS_RGB;
    jpeg_set_defaults(&cinfo);
    jpeg_set_quality(&cinfo, 75, TRUE);
    FILE *outfile = fopen(filename, "wb");
    if (outfile == NULL)
    {
        std::cerr << "Error: Failed to open file " << filename << " for writing." << std::endl;
        exit(1);
    }
    jpeg_stdio_dest(&cinfo, outfile);
    jpeg_start_compress(&cinfo, TRUE);

    // 写入像素数据
    while (cinfo.next_scanline < cinfo.image_height)
    {
        int y = cinfo.next_scanline;
        for (int x = 0; x < cinfo.image_width; x++)
        {
            int index = y * width + x;
            row_buffer[x * 3] = static_cast<unsigned char>(pixels[index] % 256);
            row_buffer[x * 3 + 1] = static_cast<unsigned char>(pixels[index] % 256);
            row_buffer[x * 3 + 2] = static_cast<unsigned char>(pixels[index] % 256);
        }
        row_pointer[0] = row_buffer;
        jpeg_write_scanlines(&cinfo, row_pointer, 1);
    }

    // 完成保存操作
    jpeg_finish_compress(&cinfo);
    fclose(outfile);
    delete[] row_buffer;
    jpeg_destroy_compress(&cinfo);
}

int main(int argc, char *argv[])
{
    // 检查输入参数是否正确
    // if (argc != 4)
    // {
    //     std::cerr << "Usage: " << argv[0] << " <num_points> <width> <height>" << std::endl;
    //     exit(1);
    // }

    // 读取输入参数
    // int num_points = std::atoi(argv[1]);
    // int width = std::atoi(argv[2]);
    // int height = std::atoi(argv[3]);
    int num_points = 512;
    int width = 1024;
    int height = 1024;

    // 分配内存并生成随机种子点
    Vector2D *points = new Vector2D[num_points];
    std::srand(static_cast<unsigned int>(std::time(nullptr)));
    for (int i = 0; i < num_points; i++)
    {
        points[i].x = static_cast<float>(std::rand() % width);
        points[i].y = static_cast<float>(std::rand() % height);
    }

    // 分配内存并生成 Voronoi 图
    int *output = new int[width * height];
    int *dev_output = nullptr;
    Vector2D *dev_points = nullptr;
    CHECK_CUDA_ERROR(cudaMalloc(&dev_output, sizeof(int) * width * height));
    CHECK_CUDA_ERROR(cudaMalloc(&dev_points, sizeof(Vector2D) * num_points));
    CHECK_CUDA_ERROR(cudaMemcpy(dev_points, points, sizeof(Vector2D) * num_points, cudaMemcpyHostToDevice));
    dim3 threads_per_block(16, 16);
    dim3 num_blocks(CALCULATE_NUM_BLOCKS(width, threads_per_block.x), CALCULATE_NUM_BLOCKS(height, threads_per_block.y));
    auto start_time = std::chrono::high_resolution_clock::now();
    generate_voronoi_map<<<num_blocks, threads_per_block>>>(dev_points, num_points, width, height, dev_output);
    // CHECK_CUDA_ERROR(cudaMemcpy(dev_points, points, sizeof(Vector2D) * num_points, cudaMemcpyHostToDevice));
    // 将生成的 Voronoi 图复制回主机端
    CHECK_CUDA_ERROR(cudaMemcpy(output, dev_output, sizeof(int) * width * height, cudaMemcpyDeviceToHost));
    auto end_time = std::chrono::high_resolution_clock::now();
    auto elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    std::cout << "Elapsed time for generating: " << elapsed_time.count() << " ms" << std::endl;

    // 随机上色并保存为 JPG 文件
    unsigned int *colors = new unsigned int[num_points];
    for (int i = 0; i < num_points; i++)
    {
        colors[i] = static_cast<unsigned int>(std::rand() % (1 << 24)); // 生成随机颜色
    }
    unsigned int *dev_colors = nullptr;
    CHECK_CUDA_ERROR(cudaMalloc(&dev_colors, sizeof(unsigned int) * num_points));
    CHECK_CUDA_ERROR(cudaMemcpy(dev_colors, colors, sizeof(unsigned int) * num_points, cudaMemcpyHostToDevice));
    start_time = std::chrono::high_resolution_clock::now();
    // 调用 colorize_voronoi_map 函数对 Voronoi 图进行上色
    colorize_voronoi_map<<<num_blocks, threads_per_block>>>(dev_output, dev_colors, num_points, width, height);
    CHECK_CUDA_ERROR(cudaMemcpy(output, dev_output, sizeof(int) * width * height, cudaMemcpyDeviceToHost));
    end_time = std::chrono::high_resolution_clock::now();
    elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    std::cout << "Elapsed time for coloring: " << elapsed_time.count() << " ms" << std::endl;
    save_jpeg_image(output, width, height, "../src/voronoi/voronoi2d.jpg");

    // 释放内存
    delete[] points;
    delete[] output;
    delete[] colors;
    CHECK_CUDA_ERROR(cudaFree(dev_points));
    CHECK_CUDA_ERROR(cudaFree(dev_output));
    CHECK_CUDA_ERROR(cudaFree(dev_colors));
    return 0;
}
