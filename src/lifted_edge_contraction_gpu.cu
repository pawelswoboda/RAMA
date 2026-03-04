#include "lifted_edge_contraction.h"

template
std::tuple<thrust::device_vector<int>, int>
find_lifted_contraction_mapping<thrust::device_vector>(
    const Graph<thrust::device_vector>&, const Graph<thrust::device_vector>&, bool);
