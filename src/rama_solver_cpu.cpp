#define THRUST_DEVICE_SYSTEM THRUST_DEVICE_SYSTEM_CPP
#define THRUST_HOST_SYSTEM THRUST_HOST_SYSTEM_CPP

#include "rama_solver.h"

template
std::tuple<thrust::host_vector<int>, double, std::vector<std::vector<int>>>
rama_solver<thrust::host_vector>(
    Graph<thrust::host_vector>&, Graph<thrust::host_vector>&,
    const multicut_solver_options&);

template
std::tuple<thrust::host_vector<int>, double, std::vector<std::vector<int>>>
rama_solver<thrust::host_vector>(
    Graph<thrust::host_vector>&, const multicut_solver_options&);
