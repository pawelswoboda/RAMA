#include "edge_contractions.h"

template
std::tuple<thrust::device_vector<int>, int>
find_contraction_mapping<thrust::device_vector>(const Graph<thrust::device_vector>&, const Graph<thrust::device_vector>&, bool);
