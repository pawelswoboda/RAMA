add_library(multicut_text_parser multicut_text_parser.cpp)

# Explicit template instantiation libraries.
# As more components get explicit instantiation, add their .cpp/.cu files here.
add_library(rama_cpu STATIC
    graph_cpu.cpp
    find_triangles_cpu.cpp
    find_quadrangles_cpu.cpp
    find_pentagons_cpu.cpp
    find_cycles_bfs_cpu.cpp
    maximum_matching_cpu.cpp
    mst_boruvka_cpu.cpp
    edge_contractions_cpu.cpp
    lifted_cut_constraints_cpu.cpp
    lifted_edge_contraction_cpu.cpp
    connected_components_cpu.cpp
    conflicted_cycles_cpu.cpp
    multicut_message_passing_cpu.cpp
    dual_solver_cpu.cpp
    rama_solver_cpu.cpp)
target_link_libraries(rama_cpu PUBLIC RAMA)

# CCCL v3 references cudaGetErrorString when cuda_runtime_api.h is on the
# include path, even for CPU-only builds.  Link cudart_static to satisfy
# the symbol whenever CUDA is installed.
find_package(CUDAToolkit QUIET)
if(CUDAToolkit_FOUND)
    target_link_libraries(rama_cpu PUBLIC CUDA::cudart_static)
endif()

if(WITH_CUDA)
    add_library(rama_gpu STATIC
        graph_gpu.cu
        find_triangles_gpu.cu
        find_quadrangles_gpu.cu
        find_pentagons_gpu.cu
        find_cycles_bfs_gpu.cu
        maximum_matching_gpu.cu
        mst_boruvka_gpu.cu
        edge_contractions_gpu.cu
        lifted_cut_constraints_gpu.cu
        lifted_edge_contraction_gpu.cu
        connected_components_gpu.cu
        conflicted_cycles_gpu.cu
        multicut_message_passing_gpu.cu
        dual_solver_gpu.cu
        rama_solver_gpu.cu)
    target_compile_definitions(rama_gpu PRIVATE ${GPU_COMPILE_DEFINITIONS})
    target_link_libraries(rama_gpu PUBLIC RAMA)
endif()

if(WITH_CUDA)
    add_executable(rama_text_input_gpu rama_text_input_gpu.cu)
    target_compile_definitions(rama_text_input_gpu PRIVATE ${GPU_COMPILE_DEFINITIONS})
    target_link_libraries(rama_text_input_gpu rama_gpu multicut_text_parser)
endif()

add_executable(rama_text_input_cpu rama_text_input_cpu.cpp)
# Compiled with g++ (not nvcc), CPP backend defined in source file
target_link_libraries(rama_text_input_cpu rama_cpu multicut_text_parser)

if(WITH_CUDA)
    pybind11_add_module(rama_py rama_py.cu)
    target_link_libraries(rama_py PRIVATE multicut_text_parser rama_gpu)

    if(WITH_TORCH)
        find_library(TORCH_PYTHON_LIBRARY torch_python PATHS "${TORCH_INSTALL_PREFIX}/lib")
        message(STATUS "TORCH_PYTHON_LIBRARY: ${TORCH_PYTHON_LIBRARY}") # https://github.com/pytorch/pytorch/issues/38122#issuecomment-694203281
        target_link_libraries(rama_py PRIVATE "${TORCH_LIBRARIES}" ${TORCH_PYTHON_LIBRARY})
    endif()
endif()
