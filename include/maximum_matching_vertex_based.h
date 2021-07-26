#pragma once

#include "dCOO.h"

std::tuple<thrust::device_vector<int>, int> filter_edges_by_matching_vertex_based(const dCOO& A, const float retain_ratio = 0.5);
