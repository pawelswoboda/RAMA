#pragma once

#include "dCOO.h"
#include <cusparse.h>

std::tuple<thrust::device_vector<int>, int> filter_edges_by_matching_vertex_based(cusparseHandle_t handle, const dCOO& A);
