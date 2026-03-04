#include "rama_solver.h"

template
std::tuple<thrust::device_vector<int>, double, std::vector<std::vector<int>>>
rama_solver<thrust::device_vector>(
    Graph<thrust::device_vector>&, Graph<thrust::device_vector>&,
    const multicut_solver_options&);

template
std::tuple<thrust::device_vector<int>, double, std::vector<std::vector<int>>>
rama_solver<thrust::device_vector>(
    Graph<thrust::device_vector>&, const multicut_solver_options&);
