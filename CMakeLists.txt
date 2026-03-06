cmake_minimum_required(VERSION 3.22.1)

option(WITH_CUDA "Compile GPU components" ON)

if(WITH_CUDA)
    project(RAMA LANGUAGES CXX CUDA)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -use_fast_math --extended-lambda")
    set(CMAKE_CUDA_ARCHITECTURES native)
else()
    project(RAMA LANGUAGES CXX)
endif()

include_directories(include)

add_library(RAMA INTERFACE)
target_include_directories(RAMA INTERFACE include/)
target_compile_features(RAMA INTERFACE cxx_std_17)

target_compile_options(RAMA INTERFACE -march=native)
target_compile_options(RAMA INTERFACE -fPIC)

if(WITH_CUDA)
    # Define global Thrust execution backend flags
    # GPU targets use CUDA device system
    set(GPU_COMPILE_DEFINITIONS
        THRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_CUDA
        THRUST_HOST_SYSTEM=THRUST_HOST_SYSTEM_CPP)
endif()

# CPU targets use CPP device system (sequential, no OpenMP)
set(CPU_COMPILE_DEFINITIONS
    THRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_CPP
    THRUST_HOST_SYSTEM=THRUST_HOST_SYSTEM_CPP)

# Compiler options to undefine and redefine Thrust backend for CPU targets
set(CPU_COMPILE_OPTIONS
    -UTHRUST_DEVICE_SYSTEM
    -UTHRUST_HOST_SYSTEM
    -DTHRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_CPP
    -DTHRUST_HOST_SYSTEM=THRUST_HOST_SYSTEM_CPP)

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

if(WITH_CUDA)
    CPMAddPackage(
        NAME pybind11
        GIT_TAG master
        GITHUB_REPOSITORY pybind/pybind11
        OPTIONS
        "PYBIND11_CPP_STANDARD -std=c++17"
        "PYBIND11_INSTALL ON CACHE BOOL"
    )
    find_package(pybind11 REQUIRED)
endif()

CPMAddPackage(
    NAME CCCL
    GITHUB_REPOSITORY nvidia/cccl
    GIT_TAG v3.2.1
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
    if(NOT WITH_CUDA)
        message(FATAL_ERROR "WITH_TORCH requires WITH_CUDA=ON")
    endif()
    message(STATUS "Compiling with PyTorch support")
    find_package(Torch REQUIRED)
    add_definitions(-DWITH_TORCH)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
else()
    message("Compiling without PyTorch support")
endif()

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

add_subdirectory(src)
enable_testing()
add_subdirectory(test)
