#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

std::tuple<thrust::device_vector<int>, thrust::device_vector<int>, thrust::device_vector<float>> parallel_cycle_packing_cuda(
    const thrust::device_vector<int>& row_ids, const thrust::device_vector<int>& col_ids, const thrust::device_vector<float>& costs, 
    const int max_cycle_length);
