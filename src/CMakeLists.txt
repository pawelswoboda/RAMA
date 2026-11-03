add_library(multicut_text_parser multicut_text_parser.cpp)

# Explicit template instantiation libraries.
# As more components get explicit instantiation, add their .cpp/.cu files here.
add_library(rama_cpu STATIC
    graph_cpu.cpp
    find_triangles_cpu.cpp
    find_quadrangles_cpu.cpp
    find_pentagons_cpu.cpp
    find_cycles_bfs_cpu.cpp
    find_cycles_mst_cpu.cpp
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

if(WITH_CUDA)
    add_library(rama_gpu STATIC
        graph_gpu.cu
        find_triangles_gpu.cu
        find_quadrangles_gpu.cu
        find_pentagons_gpu.cu
        find_cycles_bfs_gpu.cu
        find_cycles_mst_gpu.cu
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
    target_link_libraries(rama_gpu PUBLIC RAMA CCCL::CCCL)
endif()

if(WITH_CUDA)
    add_executable(rama_text_input_gpu rama_text_input_gpu.cu)
    target_compile_definitions(rama_text_input_gpu PRIVATE ${GPU_COMPILE_DEFINITIONS})
    target_link_libraries(rama_text_input_gpu rama_gpu multicut_text_parser)
endif()

add_executable(rama_text_input_cpu rama_text_input_cpu.cpp)
# Compiled with g++ (not nvcc), CPP backend defined in source file
target_link_libraries(rama_text_input_cpu rama_cpu multicut_text_parser)

# The module itself (rama_py.cpp) is plain C++; each backend lives in its own
# file so that CPU and GPU Thrust types never meet in one translation unit.
pybind11_add_module(rama_py rama_py.cpp rama_py_cpu.cpp)
target_link_libraries(rama_py PRIVATE multicut_text_parser rama_cpu)

if(WITH_CUDA)
    target_sources(rama_py PRIVATE rama_py_gpu.cu)
    target_compile_definitions(rama_py PRIVATE RAMA_WITH_CUDA)
    target_link_libraries(rama_py PRIVATE rama_gpu)
    # rama_gpu brings in CUDA Thrust backend definitions; the CPU backend must not see them.
    set_source_files_properties(rama_py_cpu.cpp PROPERTIES COMPILE_OPTIONS "${CPU_COMPILE_OPTIONS}")

    if(WITH_TORCH)
        find_library(TORCH_PYTHON_LIBRARY torch_python PATHS "${TORCH_INSTALL_PREFIX}/lib")
        message(STATUS "TORCH_PYTHON_LIBRARY: ${TORCH_PYTHON_LIBRARY}") # https://github.com/pytorch/pytorch/issues/38122#issuecomment-694203281
        target_link_libraries(rama_py PRIVATE "${TORCH_LIBRARIES}" ${TORCH_PYTHON_LIBRARY})
    endif()
endif()
