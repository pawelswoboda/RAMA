#define THRUST_DEVICE_SYSTEM THRUST_DEVICE_SYSTEM_CPP
#define THRUST_HOST_SYSTEM THRUST_HOST_SYSTEM_CPP

#include "conflicted_cycles.h"

template
std::tuple<thrust::host_vector<int>, thrust::host_vector<int>, thrust::host_vector<int>>
conflicted_cycles<thrust::host_vector>(
    const Graph<thrust::host_vector>&, int, float, bool, const std::string&, float);
