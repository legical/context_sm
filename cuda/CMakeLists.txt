# required cmake version
cmake_minimum_required(VERSION 3.18.2)
set(CMAKE_BUILD_TYPE Release)

project(CUDArandom VERSION 1.0.0 LANGUAGES CXX CUDA DESCRIPTION "The Influence of Random Factors on the Running Time of GPU Programs")

find_package(CUDA  REQUIRED)

# 架构 3060
set(TARGET sm_86)
# nvcc flags
# set(CUDA_NVCC_FLAGS -gencode arch=compute_86,code=sm_86;-G;-g)
# set(CMAKE_NVCC_FLAGS "CMAKE_NVCC_FLAGS -std=c++14")
string(APPEND CMAKE_CUDA_FLAGS " -m 64 -arch=${TARGET}")

# add the binary tree to the search path for include files
include_directories("${CUDA_INCLUDE_DIRS}")
include_directories(${PROJECT_SOURCE_DIR}/lib)

# 添加所有源文件
# aux_source_directory(${PROJECT_SOURCE_DIR}/src/lib SRCLIST)
# 将源文件打包成动态库
# add_library(cuda_utl SHARED ${SRCLIST})
# 指定目标的编译特性c++ 11
# target_compile_features(cuda_utl PUBLIC cxx_std_11)
# 将cuda等3个目标文件与 cuda_utl 库文件进行链接
# target_link_libraries(cuda_utl cuda cudart nvidia-ml)

function(add_my_target target include_dir)
    add_executable(${target} ${ARGN})
    target_compile_features(${target} PUBLIC cxx_std_11)
    target_link_libraries(${target} cuda cudart nvidia-ml)
    target_include_directories(${target}  PUBLIC ${include_dir})
endfunction(add_my_target)

add_my_target(cu_ran src/memory src/memory/random.cu)
# add_my_target(get_file_dir src/filedir src/filedir/dir.cu)