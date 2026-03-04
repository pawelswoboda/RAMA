#include "connected_components.h"

template
thrust::device_vector<int>
connected_components::compute_cc<thrust::device_vector>(
    int, const thrust::device_vector<int>&, const thrust::device_vector<int>&);
