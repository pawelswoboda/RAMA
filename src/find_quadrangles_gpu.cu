#include "find_quadrangles.h"

template
void deduplicate_triangles<thrust::device_vector>(
    thrust::device_vector<int>&, thrust::device_vector<int>&, thrust::device_vector<int>&);

template
void deduplicate_triangles<thrust::device_vector>(
    thrust::device_vector<int>&, thrust::device_vector<int>&, thrust::device_vector<int>&,
    thrust::device_vector<float>&);

template
std::tuple<thrust::device_vector<int>, thrust::device_vector<int>, thrust::device_vector<int>, thrust::device_vector<float>>
find_quadrangles<thrust::device_vector>(
    const thrust::device_vector<int>&, const thrust::device_vector<int>&,
    const thrust::device_vector<int>&, const thrust::device_vector<int>&,
    const thrust::device_vector<float>&, const thrust::device_vector<float>&);
