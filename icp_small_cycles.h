#pragma once

#include <thrust/device_vector.h>
#include <cusparse.h>

std::tuple<thrust::device_vector<int>, thrust::device_vector<int>, thrust::device_vector<float>> parallel_small_cycle_packing_cuda(
    cusparseHandle_t handle, 
    const thrust::device_vector<int>& row_ids, 
    const thrust::device_vector<int>& col_ids, 
    const thrust::device_vector<float>& costs, 
    const int max_tries);
