#define THRUST_DEVICE_SYSTEM THRUST_DEVICE_SYSTEM_CPP
#define THRUST_HOST_SYSTEM THRUST_HOST_SYSTEM_CPP

#include "lifted_edge_contraction.h"

template
std::tuple<thrust::host_vector<int>, int>
find_lifted_contraction_mapping<thrust::host_vector>(
    const Graph<thrust::host_vector>&, const Graph<thrust::host_vector>&, bool);
