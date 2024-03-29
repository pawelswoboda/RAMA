#pragma once

#include <vector>
#include "dCOO.h"
#include "multicut_solver_options.h"

thrust::device_vector<float> calculate_sums(const dCOO& A);
thrust::device_vector<float> calculate_contracting_edges(const dCOO& A, const thrust::device_vector<float>& node_costs);
dCOO calculate_connected_subgraph(const dCOO& A, thrust::device_vector<float>& edges);
dCOO calculate_contracted_graph(dCOO& A, const dCOO& B);

dCOO preprocessor_cuda(const std::vector<int> &i, const std::vector<int> &j, const std::vector<float> &costs,
                       const multicut_solver_options &opts);