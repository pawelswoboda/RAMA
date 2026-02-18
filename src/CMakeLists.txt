add_library(multicut_text_parser multicut_text_parser.cpp)

# All core components are now header-only (templatized):
# graph.h, dual_solver.h, edge_contractions.h, maximum_matching.h,
# multicut_message_passing.h, conflicted_cycles.h, rama_solver.h

add_executable(rama_text_input_gpu rama_text_input_gpu.cu)
target_compile_definitions(rama_text_input_gpu PRIVATE ${GPU_COMPILE_DEFINITIONS})
target_link_libraries(rama_text_input_gpu RAMA multicut_text_parser)

add_executable(rama_text_input_cpu rama_text_input_cpu.cpp)
# Compiled with g++ (not nvcc), CPP backend defined in source file
target_link_libraries(rama_text_input_cpu RAMA multicut_text_parser)

pybind11_add_module(rama_py rama_py.cu)
target_link_libraries(rama_py PRIVATE multicut_text_parser RAMA)

if(WITH_TORCH)
    find_library(TORCH_PYTHON_LIBRARY torch_python PATHS "${TORCH_INSTALL_PREFIX}/lib")
    message(STATUS "TORCH_PYTHON_LIBRARY: ${TORCH_PYTHON_LIBRARY}") # https://github.com/pytorch/pytorch/issues/38122#issuecomment-694203281
    target_link_libraries(rama_py PRIVATE "${TORCH_LIBRARIES}" ${TORCH_PYTHON_LIBRARY})
endif()