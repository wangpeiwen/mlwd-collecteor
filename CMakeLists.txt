cmake_minimum_required(VERSION 3.22)
project(mlwd_stress_kernels LANGUAGES CXX CUDA)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES 70)

add_subdirectory(cuda)
