#pragma once

#include <thrust/device_vector.h>
#include "dCOO.h"

std::tuple<thrust::device_vector<int>, thrust::device_vector<int>, thrust::device_vector<int>, thrust::device_vector<float>> conflicted_cycles_cuda(
    const dCOO& A, const int max_cycle_length, const float tri_memory_factor = 2.0, const float tol_ratio = 1e-4);