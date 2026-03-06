#include "lifted_cut_constraints.h"

template
LiftedCutFactors<thrust::device_vector>
find_lifted_cut_constraints<thrust::device_vector>(
    const Graph<thrust::device_vector>&, const Graph<thrust::device_vector>&, bool, float);
