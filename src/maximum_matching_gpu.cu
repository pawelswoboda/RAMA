#include "maximum_matching.h"

template
std::tuple<thrust::device_vector<int>, int>
filter_edges_by_matching<thrust::device_vector>(const Graph<thrust::device_vector>&, float, bool);
