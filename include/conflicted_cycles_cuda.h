#pragma once

#include <thrust/device_vector.h>
#include "dCOO.h"

std::tuple<thrust::device_vector<int>, thrust::device_vector<int>, thrust::device_vector<int>> conflicted_cycles_cuda(
    const dCOO& A, const int max_cycle_length, const float tri_memory_factor = 2.0, const float tol_ratio = 1e-4, const bool verbose = true);
std::tuple<dCOO, thrust::device_vector<int>, thrust::device_vector<int>>
    create_matrices(const dCOO& A, const float thresh);
__global__ void find_triangles_parallel(const int num_rep_edges,
                                    const int* const __restrict__ row_ids_rep,
                                    const int* const __restrict__ col_ids_rep,
                                    const int* const __restrict__ A_symm_row_offsets,
                                    const int* const __restrict__ A_symm_col_ids,
                                    const float* const __restrict__ A_symm_data,
                                    int* __restrict__ triangle_v1,
                                    int* __restrict__ triangle_v2,
                                    int* __restrict__ triangle_v3,
                                    int* __restrict__ empty_tri_index,
                                    const int max_triangles);

__global__ void find_quadrangles_parallel(const long num_expansions, const int num_rep_edges,
                                        const int* const __restrict__ row_ids_rep,
                                        const int* const __restrict__ col_ids_rep,
                                        const long* const __restrict__ rep_row_offsets,
                                        const int* const __restrict__ A_symm_row_offsets,
                                        const int* const __restrict__ A_symm_col_ids,
                                        const float* const __restrict__ A_symm_data,
                                        int* __restrict__ triangle_v1,
                                        int* __restrict__ triangle_v2,
                                        int* __restrict__ triangle_v3,
                                        int* __restrict__ empty_tri_index,
                                        const int max_triangles);

__global__ void find_pentagons_parallel(const long num_expansions, const int num_rep_edges,
                                        const int* const __restrict__ row_ids_rep,
                                        const int* const __restrict__ col_ids_rep,
                                        const long* const __restrict__ rep_edge_offsets,
                                        const int* const __restrict__ A_symm_row_offsets,
                                        const int* const __restrict__ A_symm_col_ids,
                                        const float* const __restrict__ A_symm_data,
                                        int* __restrict__ triangle_v1,
                                        int* __restrict__ triangle_v2,
                                        int* __restrict__ triangle_v3,
                                        int* __restrict__ empty_tri_index,
                                        const int max_triangles);