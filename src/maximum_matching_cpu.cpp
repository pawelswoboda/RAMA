#define THRUST_DEVICE_SYSTEM THRUST_DEVICE_SYSTEM_CPP
#define THRUST_HOST_SYSTEM THRUST_HOST_SYSTEM_CPP

#include "maximum_matching.h"

template
std::tuple<thrust::host_vector<int>, int>
filter_edges_by_matching<thrust::host_vector>(const Graph<thrust::host_vector>&, float, bool);
