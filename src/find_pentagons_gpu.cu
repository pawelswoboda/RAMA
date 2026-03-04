#include "find_pentagons.h"

template
std::tuple<thrust::device_vector<int>, thrust::device_vector<int>, thrust::device_vector<int>, thrust::device_vector<float>>
find_pentagons<thrust::device_vector>(
    const thrust::device_vector<int>&, const thrust::device_vector<int>&,
    const thrust::device_vector<int>&, const thrust::device_vector<int>&,
    const thrust::device_vector<float>&, const thrust::device_vector<float>&);
