#include "dual_solver.h"

template
double dual_solver<thrust::device_vector>(
    Graph<thrust::device_vector>&, const Graph<thrust::device_vector>&,
    const Graph<thrust::device_vector>&, bool, int, int, int, float, bool,
    const std::string&, float);

template
double dual_solver<thrust::device_vector>(
    Graph<thrust::device_vector>&, int, int, int, float, bool, const std::string&, float);
