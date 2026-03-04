#define THRUST_DEVICE_SYSTEM THRUST_DEVICE_SYSTEM_CPP
#define THRUST_HOST_SYSTEM THRUST_HOST_SYSTEM_CPP

#include "connected_components.h"

template
thrust::host_vector<int>
connected_components::compute_cc<thrust::host_vector>(
    int, const thrust::host_vector<int>&, const thrust::host_vector<int>&);
