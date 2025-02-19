#pragma once

#include <vector>
#include <tuple>
#include <thrust/device_vector.h>
#include "multicut_solver_options.h"

std::tuple<std::vector<int>, double, int, std::vector<std::vector<int>> > rama_cuda(const std::vector<int>& i, const std::vector<int>& j, const std::vector<float>& costs, const multicut_solver_options& opts); 
std::tuple<thrust::device_vector<int>, double, std::vector<std::vector<int>>> rama_cuda(thrust::device_vector<int>&& i, thrust::device_vector<int>&& j, thrust::device_vector<float>&& costs, const multicut_solver_options& opts, const int device);
