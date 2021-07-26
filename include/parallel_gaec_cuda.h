#pragma once

#include <vector>
#include <tuple>
#include "dCOO.h"
#include "multicut_solver_options.h"

std::vector<int> parallel_gaec_cuda(const std::vector<int>& i, const std::vector<int>& j, const std::vector<float>& costs, const multicut_solver_options& opts); 

std::vector<int> parallel_gaec_cuda(thrust::device_vector<int>&& i, thrust::device_vector<int>&& j, thrust::device_vector<float>&& costs, const multicut_solver_options& opts); 

std::vector<int> parallel_gaec_cuda(dCOO& A, const multicut_solver_options& opts);

void print_obj_original(const std::vector<int>& h_node_mapping, const std::vector<int>& i, const std::vector<int>& j, const std::vector<float>& costs);
