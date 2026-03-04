#define THRUST_DEVICE_SYSTEM THRUST_DEVICE_SYSTEM_CPP
#define THRUST_HOST_SYSTEM THRUST_HOST_SYSTEM_CPP

#include "dual_solver.h"

template
double dual_solver<thrust::host_vector>(
    Graph<thrust::host_vector>&, const Graph<thrust::host_vector>&,
    const Graph<thrust::host_vector>&, bool, int, int, int, float, bool,
    const std::string&, float);

template
double dual_solver<thrust::host_vector>(
    Graph<thrust::host_vector>&, int, int, int, float, bool, const std::string&, float);
