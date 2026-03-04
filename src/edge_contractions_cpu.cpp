#define THRUST_DEVICE_SYSTEM THRUST_DEVICE_SYSTEM_CPP
#define THRUST_HOST_SYSTEM THRUST_HOST_SYSTEM_CPP

#include "edge_contractions.h"

template
std::tuple<thrust::host_vector<int>, int>
find_contraction_mapping<thrust::host_vector>(const Graph<thrust::host_vector>&, const Graph<thrust::host_vector>&, bool);
