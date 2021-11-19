project(RAMA LANGUAGES CXX CUDA)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -use_fast_math")

cmake_minimum_required(VERSION 2.8.12)

set(CMAKE_CUDA_ARCHITECTURES OFF)

include_directories(external/thrust/)
include_directories(external/thrust/dependencies/cub)
include_directories(external/ECL-CC)
include_directories(external/CLI11/include)
include_directories(include)

add_library(RAMA INTERFACE)
target_include_directories(RAMA INTERFACE include/)
target_compile_features(RAMA INTERFACE cxx_std_17)
target_compile_options(RAMA INTERFACE -march=native)
target_compile_options(RAMA INTERFACE -fPIC)
target_include_directories(RAMA INTERFACE external/Eigen)

find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(RAMA INTERFACE OpenMP::OpenMP_CXX)
endif()

option(WITH_TORCH "Compile with PyTorch support" OFF)

if(WITH_TORCH)
    message(STATUS "Compiling with PyTorch support")
    find_package(Torch REQUIRED)
    add_definitions(-DWITH_TORCH)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
else()
    message("Compiling without PyTorch support")
endif()

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(PYBIND11_CPP_STANDARD -std=c++17)
set(PYBIND11_INSTALL ON CACHE BOOL "enable pybind11 bindings.")
add_subdirectory(external/pybind11)

add_library(ECL-CC_11 external/ECL-CC/ECL-CC_11.cu)
target_include_directories(ECL-CC_11 INTERFACE external/ECL-CC)

add_library(gpuMST_boruvka external/cudaMST/gpuMST/gpuMST.cu)
target_include_directories(gpuMST_boruvka PUBLIC external/cudaMST/gpuMST/)
target_include_directories(gpuMST_boruvka PUBLIC external/cudaMST/common/)

add_library(gpuMST_kruskal external/cudaMST/gpuMSTdpk/gpuMST.cu)
target_include_directories(gpuMST_kruskal PUBLIC external/cudaMST/gpuMSTdpk/)
target_include_directories(gpuMST_kruskal PUBLIC external/cudaMST/common/)

add_subdirectory(src)
add_subdirectory(test)
