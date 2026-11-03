#include "find_cycles_mst.h"

template
std::tuple<thrust::device_vector<int>, thrust::device_vector<int>, thrust::device_vector<int>, thrust::device_vector<float>>
find_conflicted_cycles_mst<thrust::device_vector>(
    const thrust::device_vector<int>&, const thrust::device_vector<int>&, const thrust::device_vector<float>&,
    const thrust::device_vector<int>&, const thrust::device_vector<int>&, const thrust::device_vector<float>&,
    int, int, bool);