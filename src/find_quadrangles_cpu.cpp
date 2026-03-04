#define THRUST_DEVICE_SYSTEM THRUST_DEVICE_SYSTEM_CPP
#define THRUST_HOST_SYSTEM THRUST_HOST_SYSTEM_CPP

#include "find_quadrangles.h"

template
void deduplicate_triangles<thrust::host_vector>(
    thrust::host_vector<int>&, thrust::host_vector<int>&, thrust::host_vector<int>&);

template
void deduplicate_triangles<thrust::host_vector>(
    thrust::host_vector<int>&, thrust::host_vector<int>&, thrust::host_vector<int>&,
    thrust::host_vector<float>&);

template
std::tuple<thrust::host_vector<int>, thrust::host_vector<int>, thrust::host_vector<int>, thrust::host_vector<float>>
find_quadrangles<thrust::host_vector>(
    const thrust::host_vector<int>&, const thrust::host_vector<int>&,
    const thrust::host_vector<int>&, const thrust::host_vector<int>&,
    const thrust::host_vector<float>&, const thrust::host_vector<float>&);
