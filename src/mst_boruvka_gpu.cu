#include "mst_boruvka.h"

template
std::tuple<thrust::device_vector<int>, thrust::device_vector<int>, thrust::device_vector<float>>
MST_boruvka::maximum_spanning_tree<thrust::device_vector>(
    const thrust::device_vector<int>&, const thrust::device_vector<int>&, const thrust::device_vector<float>&);
