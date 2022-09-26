#pragma once

#include <vector>
#include <tuple>
#include "dCOO.h"
#include "multicut_solver_options.h"

std::tuple<std::vector<int>, double, int, std::vector<std::vector<int>> > rama_cuda(const std::vector<int>& i, const std::vector<int>& j, const std::vector<float>& costs, const multicut_solver_options& opts); 
std::tuple<thrust::device_vector<int>, double> rama_cuda(thrust::device_vector<int>&& i, thrust::device_vector<int>&& j, thrust::device_vector<float>&& costs, const multicut_solver_options& opts, const int device);
std::tuple<bool, thrust::device_vector<int>> single_primal_iteration(thrust::device_vector<int>& node_mapping, dCOO& A, bool& try_edges_to_contract_by_maximum_matching, const multicut_solver_options& opts, const int num_dual_itr);
