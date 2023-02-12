# Redo the experiment of Dissecting GPU Memory Hierarchy through Microbenchmarking

- [Original website](http://www.comp.hkbu.edu.hk/~chxw/gpu_benchmark.html).
- [GitHub](https://github.com/zhanglx13/GPU-Benchmark-Dissecting-GPU-Mem).
- [Paper](https://ieeexplore.ieee.org/document/7445236).

## L2 cache

在以前的基准函数中，我们在计时内存访问延迟之前加载一些数组元素，以避免冷缓存错过。在这个基准内核函数中，我们也对冷缓存的错过（如果有的话）进行计时，以观察预取机制。
In previous benchmark functions, we load some array elements before timing the memory access latency, to avoid cold cache misses. In this benchmark kernel function, we also time the cold cache misses (if any) to observe the pre-fetch mechanism.

证明 L2 cache line 的大小为32字节。
源代码： fine_grain_L2-cold_1GB.cu  结果：Kepler_L2_cacheline_stride4byte.xlsx (s = 4 bytes, iterations =4096)
在输出文件中，每8个数据中的1个被遗漏，因此 cache line 的大小为8*4=32字节。
Proof for the L2 cache line size is 32 bytes:
Source Code: fine_grain_L2-cold_1GB.cu     Result: Kepler_L2_cacheline_stride4byte.xlsx (s = 4 bytes, iterations =4096)
In the output file, every 1 of 8 data is missed, thus the cache line size is 8*4 = 32 bytes.

硬件级预取的证明。
源代码： fine_grain_L2-cold_4KB.cu  结果：Kepler_L2_prefetch_4KB.txt
在输出文件中，除了第一个数据的加载，所有的数据都是缓存命中。没有冷缓存错过，所以存在硬件级预取。
Proof for the hardware-level pre-fetch:
Source Code: fine_grain_L2-cold_4KB.cu     Result: Kepler_L2_prefetch_4KB.txt
In the output file, except the first data loading, all the data are cache hit. There is no cold cache miss so that an hardware-level pre-fetch exists.

## Texture Cache

`fine_grain_texture_L1.cu` contains code for all GPU architectures.
Results for Kepler (cc3.5, TITAN) GPU is saved in `result_Kepler_texture_3073_1.txt`.
Results for Pascal (cc6.1, GTX1080) GPU is saved in `result_Pascal_texture_6145_1.txt`.
The texture cache structure can be inferred as

| GPU | cache size | line size | \#Sets | Policy |
| --- | -----------| --------- | ------ | ------ |
| Kepler | 12 KB | 32 B | 4 | LRU |
| Pascal | 24 KB | 32 B | 4 | LRU |
