#include "conflicted_cycles.h"

template
std::tuple<thrust::device_vector<int>, thrust::device_vector<int>, thrust::device_vector<int>>
conflicted_cycles<thrust::device_vector>(
    const Graph<thrust::device_vector>&, int, float, bool, const std::string&, float);
