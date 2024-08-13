#pragma once

#include <vector>
#include <tuple>
#include "dCOO.h"
#include "multicut_solver_options.h"

std::tuple<std::vector<int>, double, int, std::vector<std::vector<int>> > rama_cuda(const std::vector<int>& i, const std::vector<int>& j, const std::vector<float>& costs, const multicut_solver_options& opts); 
std::tuple<thrust::device_vector<int>, double, std::vector<std::vector<int>>> rama_cuda(thrust::device_vector<int>&& i, thrust::device_vector<int>&& j, thrust::device_vector<float>&& costs, const multicut_solver_options& opts, const int device);
std::tuple<thrust::device_vector<int>, double, std::vector<std::vector<int>> > rama_cuda(dCOO& A, const multicut_solver_options& opts);
void map_node_labels(const thrust::device_vector<int>& cur_node_mapping, thrust::device_vector<int>& orig_node_mapping);