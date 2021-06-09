project(parallel-gaec)

cmake_minimum_required(VERSION 2.8.12)


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

add_library(parallel-gaec-eigen parallel-gaec-eigen.cpp)
target_link_libraries(parallel-gaec-eigen parallel-gaec) 

add_executable(parallel-gaec-eigen-test parallel-gaec-eigen-test.cpp)
target_link_libraries(parallel-gaec-eigen-test parallel-gaec-eigen)

pybind11_add_module(parallel_gaec_py parallel_gaec_py.cpp)
target_link_libraries(parallel_gaec_py PRIVATE parallel-gaec-eigen)
