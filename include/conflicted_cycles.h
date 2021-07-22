#pragma once

#include"dCOO.h"
#include <thrust/device_vector.h>

std::tuple<thrust::device_vector<int>, thrust::device_vector<int>, thrust::device_vector<int>> enumerate_conflicted_cycles(const dCOO& A, const int max_cycle_length);