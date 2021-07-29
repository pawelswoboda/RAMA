project(parallel_gaec LANGUAGES CXX CUDA)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -use_fast_math")

cmake_minimum_required(VERSION 2.8.12)

set(CMAKE_CUDA_ARCHITECTURES OFF)
 
include_directories(external/thrust/)
include_directories(external/thrust/dependencies/cub)
include_directories(external/ECL-CC)
include_directories(external/CLI11/include)
include_directories(include)

add_library(parallel_gaec INTERFACE)
target_include_directories(parallel_gaec INTERFACE include/)
target_compile_features(parallel_gaec INTERFACE cxx_std_17)
target_compile_options(parallel_gaec INTERFACE -march=native)
target_compile_options(parallel_gaec INTERFACE -fPIC)
target_include_directories(parallel_gaec INTERFACE external/Eigen)

find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(parallel_gaec INTERFACE OpenMP::OpenMP_CXX)
endif()

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(PYBIND11_CPP_STANDARD -std=c++17)
set(PYBIND11_INSTALL ON CACHE BOOL "enable pybind11 bindings.")
add_subdirectory(external/pybind11)

add_library(ECL-CC_11 external/ECL-CC/ECL-CC_11.cu)
target_include_directories(ECL-CC_11 INTERFACE external/ECL-CC)

add_subdirectory(src)
add_subdirectory(test)
