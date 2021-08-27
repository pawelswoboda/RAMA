#pragma once

#include <thrust/device_vector.h>

namespace MST_kruskal
{
    thrust::device_vector<int> maximum_spanning_tree(const thrust::device_vector<int>& i, const thrust::device_vector<int>& j, const thrust::device_vector<float>& neg_costs);
}
