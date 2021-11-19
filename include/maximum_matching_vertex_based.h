#pragma once

#include "dCOO.h"

std::tuple<thrust::device_vector<int>, int> filter_edges_by_matching_vertex_based(const dCOO& A, const float mean_multiplier_mm = 0.0, const bool verbose = true);
