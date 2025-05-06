project(RAMA LANGUAGES CXX CUDA)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -use_fast_math")

cmake_minimum_required(VERSION 3.22.1)

set(CMAKE_CUDA_ARCHITECTURES OFF)

include_directories(external/ECL-CC)
include_directories(include)

add_library(RAMA INTERFACE)
target_include_directories(RAMA INTERFACE include/)
target_compile_features(RAMA INTERFACE cxx_std_17)
#set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -arch=compute_60 -code=sm_89")

target_compile_options(RAMA INTERFACE -march=native)
target_compile_options(RAMA INTERFACE -fPIC)
target_include_directories(RAMA INTERFACE external/Eigen)

# external dependencies
set(CPM_DOWNLOAD_VERSION 0.34.0)

if(CPM_SOURCE_CACHE)
  set(CPM_DOWNLOAD_LOCATION "${CPM_SOURCE_CACHE}/cpm/CPM_${CPM_DOWNLOAD_VERSION}.cmake")
elseif(DEFINED ENV{CPM_SOURCE_CACHE})
  set(CPM_DOWNLOAD_LOCATION "$ENV{CPM_SOURCE_CACHE}/cpm/CPM_${CPM_DOWNLOAD_VERSION}.cmake")
else()
  set(CPM_DOWNLOAD_LOCATION "${CMAKE_BINARY_DIR}/cmake/CPM_${CPM_DOWNLOAD_VERSION}.cmake")
endif()

if(NOT (EXISTS ${CPM_DOWNLOAD_LOCATION}))
  message(STATUS "Downloading CPM.cmake to ${CPM_DOWNLOAD_LOCATION}")
  file(DOWNLOAD
       https://github.com/TheLartians/CPM.cmake/releases/download/v${CPM_DOWNLOAD_VERSION}/CPM.cmake
       ${CPM_DOWNLOAD_LOCATION}
  )
endif()

include(${CPM_DOWNLOAD_LOCATION})

CPMAddPackage(
    NAME pybind11
    GIT_TAG master
    GITHUB_REPOSITORY pybind/pybind11
    OPTIONS
    "PYBIND11_CPP_STANDARD -std=c++17"
    "PYBIND11_INSTALL ON CACHE BOOL"
)
find_package(pybind11 REQUIRED)

CPMAddPackage(
    NAME CCCL
    GITHUB_REPOSITORY nvidia/cccl
    GIT_TAG v2.8.3
)
target_link_libraries(RAMA INTERFACE CCCL::CCCL)

CPMAddPackage("gl:libeigen/eigen#master")
target_include_directories(RAMA INTERFACE "${eigen_SOURCE_DIR}")

CPMAddPackage("gh:CLIUtils/CLI11#main")
target_include_directories(RAMA INTERFACE "${CLI11_SOURCE_DIR}/include")

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

add_library(ECL-CC_11 external/ECL-CC/ECL-CC_11.cu)
target_include_directories(ECL-CC_11 INTERFACE external/ECL-CC)

add_library(gpuMST_boruvka external/cudaMST/gpuMST/gpuMST.cu)
target_link_libraries(gpuMST_boruvka RAMA)
target_include_directories(gpuMST_boruvka PUBLIC external/cudaMST/gpuMST/)
target_include_directories(gpuMST_boruvka PUBLIC external/cudaMST/common/)

add_library(gpuMST_kruskal external/cudaMST/gpuMSTdpk/gpuMST.cu)
target_link_libraries(gpuMST_kruskal RAMA)
target_include_directories(gpuMST_kruskal PUBLIC external/cudaMST/gpuMSTdpk/)
target_include_directories(gpuMST_kruskal PUBLIC external/cudaMST/common/)

add_subdirectory(src)
enable_testing()
add_subdirectory(test)
