#ifndef __MYUTIL_HPP__
#define __MYUTIL_HPP__

#include <cuda.h>
#include <cuda_runtime.h>

#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <stdlib.h>

#include <sstream>
#include <iomanip>
#include <time.h>

#include <iostream>
#include <utility>
#include <thread>

#include <random>

/* Compile time assertion */
#define COMPILE_ASSERT(val) typedef char assertion_typedef[(val)*2 - 1];

/* Assertion for CUDA functions */
#define gpuErrAssert(ans) gpuAssert((ans), __FILE__, __LINE__, true)
#define gpuErrCheck(ans) gpuAssert((ans), __FILE__, __LINE__, false)

inline int gpuAssert(cudaError_t code, const char *file, int line, bool abort = true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr, "GPUcheck: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort)
            exit(code);
        else
            return -1;
    }
    return 0;
}

#define LOG2(x)                                          \
    {                                                    \
        (uint32_t)(sizeof(x) * 8 - 1 - __builtin_clz(x)) \
    }

/* Assumes size of power of 2 */
#define ROUND_UP(a, size) (((a) + (size)-1) & ~((size)-1))

/**
 * @brief use Initial_Value to initial array
 *
 * @tparam T int/float/double
 * @param arr
 * @param Initial_Value
 */
template <class T = int>
void init_arr(T *arr, int lenth, T Initial_Value = 0)
{
    // 计算数组长度
    //  int length = std::end(arr) - std::begin(arr);
    for (int i = 0; i < lenth; i++)
    {
        arr[i] = Initial_Value;
    }
}

template <class T = int>
void init_chase_arr(T *arr, int lenth, T step = 1)
{
    // 计算数组长度
    //  auto length = std::end(arr) - std::begin(arr);
    for (int i = 0; i < lenth; i++)
    {
        arr[i] = i + step;
    }
}

/**
 * @brief change string to Integer
 *
 * @param buf, orignal string
 * @return int
 */
inline int str_to_int(char buf[])
{
    int num = 0;
    for (int i = 0; i < strlen(buf); i++)
    {
        // 通过减去'0'可以将字符转换为int类型的数值
        num = num * 10 + buf[i] - '0';
    }
    return num;
}

#define random(min, max) (gen() * time(NULL) % (max - min + 1)) + min
/**
 * @brief Get the random num object
 *
 * @param min
 * @param max
 * @return int
 */
inline int get_random_num(int min, int max)
{
    std::random_device rd;  // 随机数发生器
    std::mt19937 gen(rd()); // 随机数引擎
    return random(min, max);
}

/**
 * @brief Use Time as filename
 * 
 * @return std::string
 * @example std::string filename = "Random" + GetTimeString() + ".csv"; 
 */
std::string GetTimeString()
{
    struct ::tm tm_time;
    time_t timestamp = time(0);
    localtime_r(&timestamp, &tm_time);

    std::ostringstream oss;

    oss << std::setfill('0')
        // << 1900+tm_time.tm_year
        // << std::setw(2) << 1+tm_time.tm_mon
        << std::setw(2) << tm_time.tm_mday
        << '-'
        << std::setw(2) << tm_time.tm_hour
        << std::setw(2) << tm_time.tm_min
        << std::setw(2) << tm_time.tm_sec;
    return oss.str();
}

#endif