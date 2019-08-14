cmake_minimum_required(VERSION 3.0)
project(hardware-effects-gpu)

find_package(CUDA REQUIRED)

set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)
list(APPEND CUDA_NVCC_FLAGS "-std=c++11")
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -lineinfo -O2")
string(APPEND CMAKE_CUDA_FLAGS " -gencode arch=compute_75,code=sm_75")
enable_language(CUDA)



add_subdirectory(bank-conflicts)
add_subdirectory(memory-coalescing)
add_subdirectory(shared-memory-limit)
add_subdirectory(divergence)
