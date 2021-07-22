add_library(multicut_text_parser multicut_text_parser.cpp)

add_library(parallel_gaec_eigen parallel_gaec_eigen.cpp)
target_link_libraries(parallel_gaec_eigen parallel_gaec) 

add_library(dCOO dCOO.cu)
target_link_libraries(dCOO ECL-CC_11)

add_library(multicut_message_passing multicut_message_passing.cu)

add_library(icp_small_cycles icp_small_cycles.cu)
target_link_libraries(icp_small_cycles dCOO)

add_library(conflicted_cycles conflicted_cycles.cu)
target_link_libraries(conflicted_cycles dCOO)

add_library(maximum_matching_vertex_based maximum_matching_vertex_based.cu)
target_link_libraries(maximum_matching_vertex_based dCOO)

add_library(parallel_gaec_cuda parallel_gaec_cuda.cu)
target_link_libraries(parallel_gaec_cuda dCOO icp_small_cycles conflicted_cycles maximum_matching_vertex_based)

add_executable(multicut_parallel_gaec_eigen_text_input multicut_parallel_gaec_eigen_text_input.cpp)
target_link_libraries(multicut_parallel_gaec_eigen_text_input multicut_text_parser parallel_gaec_eigen)

add_executable(multicut_parallel_gaec_cuda_text_input multicut_parallel_gaec_cuda_text_input.cpp)
target_link_libraries(multicut_parallel_gaec_cuda_text_input multicut_text_parser parallel_gaec_cuda dCOO)

add_executable(multicut_parallel_icp_gaec_cuda_text_input multicut_parallel_icp_gaec_cuda_text_input.cpp)
target_link_libraries(multicut_parallel_icp_gaec_cuda_text_input multicut_text_parser parallel_gaec_cuda multicut_message_passing dCOO)

add_executable(icp_cuda_text_input icp_cuda_text_input.cu)
target_link_libraries(icp_cuda_text_input icp_small_cycles multicut_text_parser)

pybind11_add_module(parallel_gaec_py parallel_gaec_py.cpp)
target_link_libraries(parallel_gaec_py PRIVATE parallel_gaec_eigen multicut_text_parser parallel_gaec_cuda) 