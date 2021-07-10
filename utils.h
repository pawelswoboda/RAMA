#pragma once

#include <cuda_runtime.h>
#include <cusparse.h>
#include <thrust/copy.h>
#include <thrust/device_vector.h>

namespace {

int get_cuda_device()
{   
    return 0; // Get first possible GPU. CUDA_VISIBLE_DEVICES automatically masks the rest of GPUs.
}

void print_gpu_memory_stats()
{
    size_t free, total;
    cudaMemGetInfo(&free, &total);
    std::cout<<"Total memory(MB): "<<total / (1024 * 1024)<<", Free(MB): "<<free / (1024 * 1024)<<std::endl;
}

void checkCuSparseError(cusparseStatus_t status, std::string errorMsg)
{
    if (status != CUSPARSE_STATUS_SUCCESS) {
        std::cout << "CuSparse error: " << errorMsg << ", status: "<< cusparseGetErrorString(status) << std::endl;
        throw std::exception();
    }
}

void checkCudaError(cudaError_t status, std::string errorMsg)
{
    if (status != cudaSuccess) {
        std::cout << "CUDA error: " << errorMsg << ", status" <<cudaGetErrorString(status) << std::endl;
        throw std::exception();
    }
}

template<typename ROW_ITERATOR, typename COL_ITERATOR>
std::tuple<thrust::device_vector<int>, thrust::device_vector<int>> to_undirected(
        ROW_ITERATOR row_id_begin, ROW_ITERATOR row_id_end,
        COL_ITERATOR col_id_begin, COL_ITERATOR col_id_end)
{
    assert(std::distance(row_id_begin, row_id_end) == std::distance(col_id_begin, col_id_end));

    const size_t nr_edges = std::distance(row_id_begin, row_id_end);
    thrust::device_vector<int> row_ids_u(2 * nr_edges);
    thrust::device_vector<int> col_ids_u(2 * nr_edges);

    thrust::copy(row_id_begin, row_id_end, row_ids_u.begin());
    thrust::copy(row_id_begin, row_id_end, col_ids_u.begin() + nr_edges);

    thrust::copy(col_id_begin, col_id_end, col_ids_u.begin());
    thrust::copy(col_id_begin, col_id_end, row_ids_u.begin() + nr_edges);

    return {row_ids_u, col_ids_u};
}


template<typename ROW_ITERATOR, typename COL_ITERATOR, typename DATA_ITERATOR>
std::tuple<thrust::device_vector<int>, thrust::device_vector<int>, thrust::device_vector<float>> to_undirected(
        ROW_ITERATOR row_id_begin, ROW_ITERATOR row_id_end,
        COL_ITERATOR col_id_begin, COL_ITERATOR col_id_end,
        DATA_ITERATOR data_begin, DATA_ITERATOR data_end)
{
    assert(std::distance(data_begin, data_end) == std::distance(col_id_begin, col_id_end));
    assert(std::distance(data_begin, data_end) == std::distance(row_id_begin, row_id_end));

    const size_t nr_edges = std::distance(data_begin, data_end);
    thrust::device_vector<int> col_ids_u(2 * nr_edges);
    thrust::device_vector<int> row_ids_u(2 * nr_edges);
    thrust::device_vector<float> costs_u(2 * nr_edges);

    thrust::copy(row_id_begin, row_id_end, row_ids_u.begin());
    thrust::copy(row_id_begin, row_id_end, col_ids_u.begin() + nr_edges);

    thrust::copy(col_id_begin, col_id_end, col_ids_u.begin());
    thrust::copy(col_id_begin, col_id_end, row_ids_u.begin() + nr_edges);

    thrust::copy(data_begin, data_end, costs_u.begin());
    thrust::copy(data_begin, data_end, costs_u.begin() + nr_edges);

    return {row_ids_u, col_ids_u, costs_u};
}

std::tuple<thrust::device_vector<int>, thrust::device_vector<int>> to_undirected(const thrust::device_vector<int>& i, const thrust::device_vector<int>& j)
{
    assert(i.size() == j.size());
    return to_undirected(i.begin(), i.end(), j.begin(), j.end());
}

std::tuple<thrust::device_vector<int>, thrust::device_vector<int>, thrust::device_vector<float>> to_undirected(const thrust::device_vector<int>& i, const thrust::device_vector<int>& j, const thrust::device_vector<float>& costs)
{
    assert(i.size() == j.size() && i.size() == costs.size());
    return to_undirected(i.begin(), i.end(), j.begin(), j.end(), costs.begin(), costs.end());
}

struct compute_lb
{
    __host__ __device__ double operator()(const float& val) const
    {
        return val < 0.0 ? val : 0.0;
    }
};

double get_lb(const thrust::device_vector<float>& costs)
{
    return thrust::transform_reduce(costs.begin(), costs.end(), compute_lb(), 0.0, thrust::plus<double>());
}

struct remove_reverse_edges_func {
    __host__ __device__
        inline int operator()(const thrust::tuple<int,int,float> e)
        {
            return thrust::get<0>(e) >= thrust::get<1>(e);
        }
};

std::tuple<thrust::device_vector<int>, thrust::device_vector<int>, thrust::device_vector<float>> to_directed(const thrust::device_vector<int>& i_symm, const thrust::device_vector<int>& j_symm, const thrust::device_vector<float>& costs_symm)
{
    assert(i_symm.size() == j_symm.size() && i_symm.size() == costs_symm.size());
    thrust::device_vector<int> i = i_symm;
    thrust::device_vector<int> j = j_symm;
    thrust::device_vector<float> costs = costs_symm;
    auto first = thrust::make_zip_iterator(thrust::make_tuple(i.begin(), j.begin(), costs.begin()));
    auto last = thrust::make_zip_iterator(thrust::make_tuple(i.end(), j.end(), costs.end()));
    auto new_last = thrust::remove_if(first, last, remove_reverse_edges_func());
    i.resize(std::distance(first, new_last));
    j.resize(std::distance(first, new_last));
    costs.resize(std::distance(first, new_last)); 

    return {i, j, costs};
}

}