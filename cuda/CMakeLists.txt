# required cmake version
cmake_minimum_required(VERSION 3.18)
set(CMAKE_BUILD_TYPE Release)

project(CUDArandom VERSION 1.0.0 LANGUAGES CXX CUDA DESCRIPTION "The Influence of Random Factors on the Running Time of GPU Programs")

find_package(CUDA REQUIRED)

# 查找 libjpeg 库
find_package(JPEG REQUIRED)
include_directories(${JPEG_INCLUDE_DIR})

# GPU choose
option(GPU_1070_IN "running in 1070" OFF)

# 架构 3060
# set(TARGET sm_86)

# 架构 1070
# set(TARGET sm_61)
# nvcc flags -m 64 Specify 64 bit architecture.
# set(CUDA_NVCC_FLAGS -gencode arch=compute_86,code=sm_86;-G;-g)
# set(CMAKE_NVCC_FLAGS "CMAKE_NVCC_FLAGS -std=c++14")
# string(APPEND CMAKE_CUDA_FLAGS " -m 64 -arch=${TARGET}")
string(APPEND CMAKE_CUDA_FLAGS " --generate-code arch=compute_86,code=sm_86 --generate-code arch=compute_61,code=sm_61")

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

    if(GPU_1070_IN)
        add_definitions(-DGPU_1070_IN)
    endif(GPU_1070_IN)

    target_compile_features(${target} PUBLIC cxx_std_11)
    target_link_libraries(${target} cuda cudart nvidia-ml jpeg)
    target_include_directories(${target} PUBLIC ${include_dir})
endfunction(add_my_target)

add_my_target(l2_dissect_test src/dissect_page src/dissect_page/dissect_page.cu)
# add_my_target(voronoi src/voronoi src/voronoi/voronoi.cu)
# add_my_target(voronoi-sm src/voronoi src/voronoi/voronoi-sm.cu)

# add_my_target(l2_set_test src/dissect/reference src/dissect/reference/fine_grain_L2_set.cu)
# add_my_target(l2_test src/dissect/reference src/dissect/reference/fine_grain_L2-cold_1GB.cu)
# add_my_target(cu_ran src/memory src/memory/random.cu)
# add_my_target(cu_ran_fork src/memory-fork src/memory-fork/random-fork.cu)
# add_my_target(cu_ran_test src/memory-test src/memory-test/memory-test.cu)
# add_my_target(vector_add src/vector-add src/vector-add/add.cu)
# add_my_target(get_file_dir src/filedir src/filedir/dir.cu)