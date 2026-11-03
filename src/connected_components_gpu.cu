#include "connected_components.h"

template
thrust::device_vector<int>
connected_components::compute_cc<thrust::device_vector>(
    int, const thrust::device_vector<int>&, const thrust::device_vector<int>&);

// GPU code that pulls results back to the host still needs the host_vector
// variant -- the extern template in the header suppresses implicit
// instantiation, and the copy in rama_cpu is compiled against a different
// Thrust backend, so its symbols do not match those of a CUDA translation
// unit. Instantiating it here keeps rama_gpu self-contained.
template
thrust::host_vector<int>
connected_components::compute_cc<thrust::host_vector>(
    int, const thrust::host_vector<int>&, const thrust::host_vector<int>&);
