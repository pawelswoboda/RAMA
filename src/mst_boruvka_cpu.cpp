#define THRUST_DEVICE_SYSTEM THRUST_DEVICE_SYSTEM_CPP
#define THRUST_HOST_SYSTEM THRUST_HOST_SYSTEM_CPP

#include "mst_boruvka.h"

template
std::tuple<thrust::host_vector<int>, thrust::host_vector<int>, thrust::host_vector<float>>
MST_boruvka::maximum_spanning_tree<thrust::host_vector>(
    const thrust::host_vector<int>&, const thrust::host_vector<int>&, const thrust::host_vector<float>&);
