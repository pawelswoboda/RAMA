#pragma once

#include <thrust/device_vector.h>
#include "dCOO.h"

std::tuple<thrust::device_vector<int>, thrust::device_vector<int>, thrust::device_vector<int>> conflicted_cycles_cuda(const dCOO& A, const int, const float);