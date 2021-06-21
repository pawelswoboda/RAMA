project(parallel-gaec LANGUAGES CXX CUDA)

cmake_minimum_required(VERSION 2.8.12)

include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

include_directories(external/thrust/)
include_directories(external/thrust/dependencies/cub)

add_library(parallel-gaec INTERFACE)
target_include_directories(parallel-gaec INTERFACE include/)
target_compile_features(parallel-gaec INTERFACE cxx_std_17)
target_compile_options(parallel-gaec INTERFACE -march=native)
target_compile_options(parallel-gaec INTERFACE -fPIC)
target_include_directories(parallel-gaec INTERFACE external/Eigen)

find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(parallel-gaec INTERFACE OpenMP::OpenMP_CXX)
endif()

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(PYBIND11_CPP_STANDARD -std=c++17)
set(PYBIND11_INSTALL ON CACHE BOOL "enable pybind11 bindings.")
add_subdirectory(external/pybind11)

add_library(multicut_text_parser multicut_text_parser.cpp)

add_library(parallel-gaec-eigen parallel-gaec-eigen.cpp)
target_link_libraries(parallel-gaec-eigen parallel-gaec) 

add_executable(parallel-gaec-eigen-test parallel-gaec-eigen-test.cpp)
target_link_libraries(parallel-gaec-eigen-test parallel-gaec-eigen)

add_library(dCSR dCSR.cu)
target_link_libraries(dCSR cusparse ECL-CC_11)

add_library(ECL-CC_11 external/ECL-CC/ECL-CC_11.cu)
target_include_directories(ECL-CC_11 INTERFACE external/ECL-CC)

add_executable(cc_gpu_test cc_gpu_test.cu)
target_link_libraries(cc_gpu_test ECL-CC_11)

add_library(parallel_gaec_cuda parallel_gaec_cuda.cu)
target_link_libraries(parallel_gaec_cuda dCSR cusparse)

add_executable(parallel_gaec_cuda_test parallel_gaec_cuda_test.cpp)
target_link_libraries(parallel_gaec_cuda_test parallel_gaec_cuda)

pybind11_add_module(parallel_gaec_py parallel_gaec_py.cpp)
target_link_libraries(parallel_gaec_py PRIVATE parallel-gaec-eigen multicut_text_parser parallel_gaec_cuda)

add_executable(multicut_parallel_gaec_eigen_text_input multicut_parallel_gaec_eigen_text_input.cpp)
target_link_libraries(multicut_parallel_gaec_eigen_text_input multicut_text_parser parallel-gaec-eigen)

add_executable(multicut_parallel_gaec_cuda_text_input multicut_parallel_gaec_cuda_text_input.cpp)
target_link_libraries(multicut_parallel_gaec_cuda_text_input multicut_text_parser parallel_gaec_cuda dCSR)
