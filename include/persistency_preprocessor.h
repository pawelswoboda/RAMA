#pragma once

#include <vector>
#include "dCOO.h"
#include "multicut_solver_options.h"

thrust::device_vector<float> calculate_sums(const dCOO& A);
thrust::device_vector<float> calculate_contracting_edges_edge_criterion(const dCOO& A, const thrust::device_vector<float>& node_costs);
dCOO calculate_connected_subgraph(const dCOO& A, thrust::device_vector<float>& edges);
std::tuple<dCOO,thrust::device_vector<int>> calculate_contracted_graph(dCOO& A, const dCOO& B);
thrust::device_vector<float> calculate_contracting_edges(dCOO& A, const thrust::device_vector<float>& node_costs, const multicut_solver_options opts);
std::tuple<dCOO, thrust::device_vector<int>> preprocessor_cuda(dCOO& A, const multicut_solver_options &opts, const int n);
thrust::device_vector<float> calculate_contracting_edges_triangle_criterion(const dCOO& A, const thrust::device_vector<float>& node_costs, const int max_cycle_length, const float tri_memory_factor, const bool verbose) ;