#define THRUST_DEVICE_SYSTEM THRUST_DEVICE_SYSTEM_CPP
#define THRUST_HOST_SYSTEM THRUST_HOST_SYSTEM_CPP

#include "lifted_cut_constraints.h"

template
LiftedCutFactors<thrust::host_vector>
find_lifted_cut_constraints<thrust::host_vector>(
    const Graph<thrust::host_vector>&, const Graph<thrust::host_vector>&, bool, float);
