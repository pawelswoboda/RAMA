add_library(multicut_text_parser multicut_text_parser.cpp)

add_library(dCOO dCOO.cu)
target_link_libraries(dCOO ECL-CC_11 RAMA)

# multicut_message_passing is now header-only (templatized)
# Tests instantiate templates directly

add_library(conflicted_cycles_cuda conflicted_cycles_cuda.cu)
target_link_libraries(conflicted_cycles_cuda dCOO RAMA)

# dual_solver is now header-only (templatized in include/dual_solver.h)

add_library(maximum_matching_vertex_based maximum_matching_vertex_based.cu)
target_link_libraries(maximum_matching_vertex_based dCOO RAMA)

#add_library(maximum_matching_thrust maximum_matching_thrust.cu)
#target_link_libraries(maximum_matching_thrust dCOO)

# edge_contractions is now header-only (templatized in include/edge_contractions.h)

add_library(rama_cuda rama_cuda.cu)
target_link_libraries(rama_cuda dCOO maximum_matching_vertex_based RAMA)

add_executable(rama_text_input rama_text_input.cu)
target_link_libraries(rama_text_input RAMA multicut_text_parser rama_cuda)

pybind11_add_module(rama_py rama_py.cu)
target_link_libraries(rama_py PRIVATE multicut_text_parser rama_cuda RAMA)

if(WITH_TORCH)
    find_library(TORCH_PYTHON_LIBRARY torch_python PATHS "${TORCH_INSTALL_PREFIX}/lib")
    message(STATUS "TORCH_PYTHON_LIBRARY: ${TORCH_PYTHON_LIBRARY}") # https://github.com/pytorch/pytorch/issues/38122#issuecomment-694203281
    target_link_libraries(rama_py PRIVATE "${TORCH_LIBRARIES}" ${TORCH_PYTHON_LIBRARY})
endif()
