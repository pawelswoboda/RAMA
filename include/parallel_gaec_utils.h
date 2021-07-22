#pragma once

#include <cuda_runtime.h>
#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <thrust/gather.h>
#include <thrust/iterator/discard_iterator.h>

inline int get_cuda_device()
{   
    return 0; // Get first possible GPU. CUDA_VISIBLE_DEVICES automatically masks the rest of GPUs.
}

inline void print_gpu_memory_stats()
{
    size_t free, total;
    cudaMemGetInfo(&free, &total);
    std::cout<<"Total memory(MB): "<<total / (1024 * 1024)<<", Free(MB): "<<free / (1024 * 1024)<<std::endl;
}

inline void checkCudaError(cudaError_t status, std::string errorMsg)
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

inline std::tuple<thrust::device_vector<int>, thrust::device_vector<int>> to_undirected(const thrust::device_vector<int>& i, const thrust::device_vector<int>& j)
{
    assert(i.size() == j.size());
    return to_undirected(i.begin(), i.end(), j.begin(), j.end());
}

inline std::tuple<thrust::device_vector<int>, thrust::device_vector<int>, thrust::device_vector<float>> to_undirected(const thrust::device_vector<int>& i, const thrust::device_vector<int>& j, const thrust::device_vector<float>& costs)
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

inline double get_lb(const thrust::device_vector<float>& costs)
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

inline std::tuple<thrust::device_vector<int>, thrust::device_vector<int>, thrust::device_vector<float>> to_directed(const thrust::device_vector<int>& i_symm, const thrust::device_vector<int>& j_symm, const thrust::device_vector<float>& costs_symm)
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

template<typename T>
void apply_permutation(thrust::device_vector<T>& keys, thrust::device_vector<int>& permutation)
{
    // copy keys to temporary vector
    thrust::device_vector<T> temp(keys.begin(), keys.end());

    // permute the keys
    thrust::gather(permutation.begin(), permutation.end(), temp.begin(), keys.begin());
}

inline void update_permutation(thrust::device_vector<int>& keys, thrust::device_vector<int>& permutation)
{
    // temporary storage for keys
    thrust::device_vector<int> temp(keys.size());

    // permute the keys with the current reordering
    thrust::gather(permutation.begin(), permutation.end(), keys.begin(), temp.begin());

    // stable_sort the permuted keys and update the permutation
    thrust::stable_sort_by_key(temp.begin(), temp.end(), permutation.begin());
}

inline void coo_sorting(thrust::device_vector<int>& i, thrust::device_vector<int>& j, thrust::device_vector<int>& k)
{
    assert(i.size() == j.size());
    assert(i.size() == k.size());
    const int N = i.size();
    thrust::device_vector<int> permutation(N);
    thrust::sequence(permutation.begin(), permutation.end());

    update_permutation(k,  permutation);
    update_permutation(j, permutation);
    update_permutation(i, permutation);

    apply_permutation(k,  permutation);
    apply_permutation(j, permutation);
    apply_permutation(i, permutation);
    assert(thrust::is_sorted(i.begin(), i.end())); 
}

inline void coo_sorting(thrust::device_vector<int>& i, thrust::device_vector<int>& j)
{
    assert(i.size() == j.size());
    const size_t N = i.size();
    thrust::device_vector<int> permutation(N);
    thrust::sequence(permutation.begin(), permutation.end());

    update_permutation(j,  permutation);
    update_permutation(i, permutation);

    apply_permutation(j,  permutation);
    apply_permutation(i, permutation);
    assert(thrust::is_sorted(i.begin(), i.end()));
}

inline void coo_sorting(thrust::device_vector<int>& i, thrust::device_vector<int>& j, thrust::device_vector<float>& data)
{
    assert(i.size() == j.size());
    assert(i.size() == data.size());
    const size_t N = i.size();
    thrust::device_vector<int> permutation(N);
    thrust::sequence(permutation.begin(), permutation.end());

    update_permutation(j,  permutation);
    update_permutation(i, permutation);

    apply_permutation(j,  permutation);
    apply_permutation(i, permutation);
    apply_permutation(data, permutation);
    assert(thrust::is_sorted(i.begin(), i.end()));
}

inline thrust::device_vector<int> compute_offsets(const thrust::device_vector<int>& i)
{
    assert(thrust::is_sorted(i.begin(), i.end()));
    thrust::device_vector<int> offsets(i.back()+2);

    thrust::unique_by_key_copy(i.begin(), i.end(),
            thrust::make_counting_iterator(0),
            thrust::make_discard_iterator(),   // discard the compacted keys
            offsets.begin());
    offsets.back() = i.size();

    return offsets;
}

inline std::tuple<thrust::device_vector<int>, thrust::device_vector<int>> get_unique_with_counts(const thrust::device_vector<int>& input)
{
    assert(thrust::is_sorted(input.begin(), input.end()));
    thrust::device_vector<int> unique_counts(input.size() + 1);
    thrust::device_vector<int> unique_values(input.size());

    auto new_end = thrust::unique_by_key_copy(input.begin(), input.end(), thrust::make_counting_iterator(0), unique_values.begin(), unique_counts.begin());
    int num_unique = std::distance(unique_values.begin(), new_end.first);
    unique_values.resize(num_unique);
    unique_counts.resize(num_unique + 1); // contains smallest index of each unique element.
    
    unique_counts[num_unique] = input.size();
    thrust::adjacent_difference(unique_counts.begin(), unique_counts.end(), unique_counts.begin());
    unique_counts = thrust::device_vector<int>(unique_counts.begin() + 1, unique_counts.end());

    return {unique_values, unique_counts};
}

inline thrust::device_vector<int> invert_unique(const thrust::device_vector<int>& values, const thrust::device_vector<int>& counts)
{
    thrust::device_vector<int> counts_sum(counts.size() + 1);
    counts_sum[0] = 0;
    thrust::inclusive_scan(counts.begin(), counts.end(), counts_sum.begin() + 1);
    
    int out_size = counts_sum.back();
    thrust::device_vector<int> output_indices(out_size, 0);

    thrust::scatter(thrust::constant_iterator<int>(1), thrust::constant_iterator<int>(1) + values.size(), counts_sum.begin(), output_indices.begin());

    thrust::inclusive_scan(output_indices.begin(), output_indices.end(), output_indices.begin());
    thrust::transform(output_indices.begin(), output_indices.end(), thrust::make_constant_iterator(1), output_indices.begin(), thrust::minus<int>());

    thrust::device_vector<int> out_values(out_size);
    thrust::gather(output_indices.begin(), output_indices.end(), values.begin(), out_values.begin());

    return out_values;
}

inline thrust::device_vector<int> compute_offsets_non_contiguous(const int max_value, const thrust::device_vector<int>& i)
{
    thrust::device_vector<int> offsets(max_value + 1, 0);
    thrust::device_vector<int> unique_ids, counts;
    std::tie(unique_ids, counts) = get_unique_with_counts(i);
    thrust::transform(unique_ids.begin(), unique_ids.end(), thrust::make_constant_iterator<int>(1), unique_ids.begin(), thrust::plus<int>());
    thrust::scatter(counts.begin(), counts.end(), unique_ids.begin(), offsets.begin());
    thrust::inclusive_scan(offsets.begin(), offsets.end(), offsets.begin());
    return offsets;
}

inline thrust::device_vector<int> concatenate(const thrust::device_vector<int>& a, const thrust::device_vector<int>& b)
{
    thrust::device_vector<int> ab(a.size() + b.size());
    thrust::copy(a.begin(), a.end(), ab.begin());
    thrust::copy(b.begin(), b.end(), ab.begin() + a.size());
    return ab;
}
/*
__host__ __device__
int min(const int a, const int b)
{
    if(a < b)
        return a;
    else
        return b; 
}
__host__ __device__
int max(const int a, const int b)
{
    if(a > b)
        return a;
    else
        return b; 
}
*/
