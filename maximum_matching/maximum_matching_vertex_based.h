#pragma once

#include "../dCSR.h"

std::tuple<thrust::device_vector<int>, thrust::device_vector<int>> filter_edges_by_matching_vertex_based(const dCSR& A);
