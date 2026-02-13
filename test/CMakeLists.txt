add_executable(multicut_message_passing_gpu_test multicut_message_passing_gpu_test.cu)
target_compile_definitions(multicut_message_passing_gpu_test PRIVATE ${GPU_COMPILE_DEFINITIONS})
target_link_libraries(multicut_message_passing_gpu_test RAMA)
add_test(NAME multicut_message_passing_gpu COMMAND multicut_message_passing_gpu_test)

add_executable(multicut_message_passing_cpu_test multicut_message_passing_cpu_test.cpp)
# Compiled with g++ (not nvcc), CPP backend defined in source file
target_link_libraries(multicut_message_passing_cpu_test RAMA)
add_test(NAME multicut_message_passing_cpu COMMAND multicut_message_passing_cpu_test)

add_executable(cc_gpu_test cc_gpu_test.cu)
target_link_libraries(cc_gpu_test ECL-CC_11)
add_test(NAME cc_gpu COMMAND cc_gpu_test)

add_executable(rama_cuda_test rama_cuda_test.cu)
target_link_libraries(rama_cuda_test dCOO rama_cuda)
add_test(NAME rama_cuda COMMAND rama_cuda_test)

add_executable(maximum_matching_vertex_based_test maximum_matching_vertex_based_test.cu)
target_link_libraries(maximum_matching_vertex_based_test dCOO maximum_matching_vertex_based)
add_test(NAME maximum_matching_vertex_based COMMAND maximum_matching_vertex_based_test)

#add_executable(maximum_matching_thrust_test maximum_matching_thrust_test.cu)
#target_link_libraries(maximum_matching_thrust_test dCOO maximum_matching_thrust)

add_executable(edge_contractions_woc_test edge_contractions_woc_test.cu)
target_link_libraries(edge_contractions_woc_test dCOO edge_contractions_woc)
add_test(NAME edge_contractions_woc COMMAND edge_contractions_woc_test)

add_executable(compute_offsets_test compute_offsets_test.cu)
add_test(NAME compute_offsets COMMAND compute_offsets_test)

add_executable(compute_unique_test compute_unique_test.cu)
add_test(NAME compute_unique COMMAND compute_unique_test)

add_executable(conflicted_cycles_test conflicted_cycles_test.cu)
target_link_libraries(conflicted_cycles_test conflicted_cycles_cuda)
add_test(NAME conflicted_cycles COMMAND conflicted_cycles_test)

add_executable(graph_gpu_test graph_gpu_test.cu)
target_compile_definitions(graph_gpu_test PRIVATE ${GPU_COMPILE_DEFINITIONS})
target_link_libraries(graph_gpu_test RAMA)
add_test(NAME graph_gpu COMMAND graph_gpu_test)

add_executable(graph_cpu_test graph_cpu_test.cpp)
# Compiled with g++ (not nvcc), CPP backend defined in source file
target_link_libraries(graph_cpu_test RAMA)
add_test(NAME graph_cpu COMMAND graph_cpu_test)

add_executable(maximum_matching_gpu_test maximum_matching_gpu_test.cu)
target_compile_definitions(maximum_matching_gpu_test PRIVATE ${GPU_COMPILE_DEFINITIONS})
target_link_libraries(maximum_matching_gpu_test RAMA)
add_test(NAME maximum_matching_gpu COMMAND maximum_matching_gpu_test)

add_executable(maximum_matching_cpu_test maximum_matching_cpu_test.cpp)
# Compiled with g++ (not nvcc), CPP backend defined in source file
target_link_libraries(maximum_matching_cpu_test RAMA)
add_test(NAME maximum_matching_cpu COMMAND maximum_matching_cpu_test)

add_executable(find_triangles_gpu_test find_triangles_gpu_test.cu)
target_compile_definitions(find_triangles_gpu_test PRIVATE ${GPU_COMPILE_DEFINITIONS})
target_link_libraries(find_triangles_gpu_test RAMA)
add_test(NAME find_triangles_gpu COMMAND find_triangles_gpu_test)

add_executable(find_triangles_cpu_test find_triangles_cpu_test.cpp)
# Compiled with g++ (not nvcc), CPP backend defined in source file
target_link_libraries(find_triangles_cpu_test RAMA)
add_test(NAME find_triangles_cpu COMMAND find_triangles_cpu_test)

add_executable(find_quadrangles_gpu_test find_quadrangles_gpu_test.cu)
target_compile_definitions(find_quadrangles_gpu_test PRIVATE ${GPU_COMPILE_DEFINITIONS})
target_link_libraries(find_quadrangles_gpu_test RAMA)
add_test(NAME find_quadrangles_gpu COMMAND find_quadrangles_gpu_test)

add_executable(find_quadrangles_cpu_test find_quadrangles_cpu_test.cpp)
# Compiled with g++ (not nvcc), CPP backend defined in source file
target_link_libraries(find_quadrangles_cpu_test RAMA)
add_test(NAME find_quadrangles_cpu COMMAND find_quadrangles_cpu_test)

add_executable(find_pentagons_gpu_test find_pentagons_gpu_test.cu)
target_compile_definitions(find_pentagons_gpu_test PRIVATE ${GPU_COMPILE_DEFINITIONS})
target_link_libraries(find_pentagons_gpu_test RAMA)
add_test(NAME find_pentagons_gpu COMMAND find_pentagons_gpu_test)

add_executable(find_pentagons_cpu_test find_pentagons_cpu_test.cpp)
# Compiled with g++ (not nvcc), CPP backend defined in source file
target_link_libraries(find_pentagons_cpu_test RAMA)
add_test(NAME find_pentagons_cpu COMMAND find_pentagons_cpu_test)

add_executable(conflicted_cycles_gpu_test conflicted_cycles_gpu_test.cu)
target_compile_definitions(conflicted_cycles_gpu_test PRIVATE ${GPU_COMPILE_DEFINITIONS})
target_link_libraries(conflicted_cycles_gpu_test RAMA)
add_test(NAME conflicted_cycles_gpu COMMAND conflicted_cycles_gpu_test)

add_executable(conflicted_cycles_cpu_test conflicted_cycles_cpu_test.cpp)
# Compiled with g++ (not nvcc), CPP backend defined in source file
target_link_libraries(conflicted_cycles_cpu_test RAMA)
add_test(NAME conflicted_cycles_cpu COMMAND conflicted_cycles_cpu_test)

add_executable(dual_solver_gpu_test dual_solver_gpu_test.cu)
target_compile_definitions(dual_solver_gpu_test PRIVATE ${GPU_COMPILE_DEFINITIONS})
target_link_libraries(dual_solver_gpu_test RAMA)
add_test(NAME dual_solver_gpu COMMAND dual_solver_gpu_test)

add_executable(dual_solver_cpu_test dual_solver_cpu_test.cpp)
# Compiled with g++ (not nvcc), CPP backend defined in source file
target_link_libraries(dual_solver_cpu_test RAMA)
add_test(NAME dual_solver_cpu COMMAND dual_solver_cpu_test)

add_executable(mst_boruvka_gpu_test mst_boruvka_gpu_test.cu)
target_compile_definitions(mst_boruvka_gpu_test PRIVATE ${GPU_COMPILE_DEFINITIONS})
target_link_libraries(mst_boruvka_gpu_test RAMA)
add_test(NAME mst_boruvka_gpu COMMAND mst_boruvka_gpu_test)

add_executable(mst_boruvka_cpu_test mst_boruvka_cpu_test.cpp)
# Compiled with g++ (not nvcc), CPP backend defined in source file
target_link_libraries(mst_boruvka_cpu_test RAMA)
add_test(NAME mst_boruvka_cpu COMMAND mst_boruvka_cpu_test)

add_executable(connected_components_gpu_test connected_components_gpu_test.cu)
target_compile_definitions(connected_components_gpu_test PRIVATE ${GPU_COMPILE_DEFINITIONS})
target_link_libraries(connected_components_gpu_test RAMA)
add_test(NAME connected_components_gpu COMMAND connected_components_gpu_test)

add_executable(connected_components_cpu_test connected_components_cpu_test.cpp)
# Compiled with g++ (not nvcc), CPP backend defined in source file
target_link_libraries(connected_components_cpu_test RAMA)
add_test(NAME connected_components_cpu COMMAND connected_components_cpu_test)