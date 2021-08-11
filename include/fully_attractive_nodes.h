#pragma once

#include "dCOO.h"

std::tuple<thrust::device_vector<int>, thrust::device_vector<int>> edges_of_fully_attractive_nodes(const dCOO& A, const float min_attr);