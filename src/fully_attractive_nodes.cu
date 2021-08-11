#include <iostream>
#include <thrust/device_vector.h>
#include <cuda_runtime.h>
#include "fully_attractive_nodes.h"
#include "time_measure_util.h"
#include "parallel_gaec_utils.h"

#define numThreads 256

__device__ int check_neighbourhood(const int row_index, float max_value,
                                const int* const __restrict__ row_offsets,
                                const int* const __restrict__ col_ids, 
                                const float* const __restrict__ data)
{
    int n_id = -1;
    for(int l = row_offsets[row_index]; l < row_offsets[row_index + 1]; ++l)
    {
        float current_val = data[l];
        int current_n = col_ids[l];
        // There is a repulsive edge so break.
        if (current_val < 0)
            return -1;

        if (current_val <= max_value || current_n == row_index)
            continue;

        max_value = current_val;
        n_id = current_n; 
    }
    return n_id;
}

__global__ void find_fully_att_nodes(const int num_nodes, 
                                    const float max_value,
                                    const int* const __restrict__ row_offsets, 
                                    const int* const __restrict__ col_ids, 
                                    const float* const __restrict__ costs,
                                    int* __restrict__ best_neighbour)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int num_threads = blockDim.x * gridDim.x;
    for (int v = tid; v < num_nodes; v += num_threads)
    {
        int best_n = check_neighbourhood(v, max_value, row_offsets, col_ids, costs);
        if (best_n != -1)
            best_neighbour[v] = best_n;
    }
}

struct discard_uncontracted
{
    __host__ __device__
        bool operator()(const thrust::tuple<int, int>& t)
        {
            return thrust::get<1>(t);
        }
};

std::tuple<thrust::device_vector<int>, thrust::device_vector<int>> edges_of_fully_attractive_nodes(const dCOO& A, const float min_attr)
{
    assert(!A.is_directed());
    thrust::device_vector<int> best_neighbour(A.rows(), -1);

    int numBlocks = ceil(A.rows() / (float) numThreads);
    const thrust::device_vector<int> A_row_offsets = A.compute_row_offsets();

    find_fully_att_nodes<<<numBlocks, numThreads>>>(A.rows(), min_attr,
        thrust::raw_pointer_cast(A_row_offsets.data()), 
        A.get_col_ids_ptr(), 
        A.get_data_ptr(), 
        thrust::raw_pointer_cast(best_neighbour.data()));

    thrust::device_vector<int> node_ids(A.rows());
    thrust::sequence(node_ids.begin(), node_ids.end(), 0);
    
    {
        auto first = thrust::make_zip_iterator(thrust::make_tuple(node_ids.begin(), best_neighbour.begin()));
        auto last = thrust::make_zip_iterator(thrust::make_tuple(node_ids.end(), best_neighbour.end()));
        auto last_valid = thrust::remove_if(first, last, discard_uncontracted());
        int num_valid = std::distance(first, last_valid);
        node_ids.resize(num_valid);
        best_neighbour.resize(num_valid);
    }

    sort_edge_nodes(node_ids, best_neighbour);
    coo_sorting(node_ids, best_neighbour);

    {
        auto first = thrust::make_zip_iterator(thrust::make_tuple(node_ids.begin(), best_neighbour.begin()));
        auto last = thrust::make_zip_iterator(thrust::make_tuple(node_ids.end(), best_neighbour.end()));
        auto new_last = thrust::unique(first, last);
        node_ids.resize(std::distance(first, new_last)); 
        best_neighbour.resize(std::distance(first, new_last)); 
    }

    std::tie(node_ids, best_neighbour) = to_undirected(node_ids, best_neighbour);
    coo_sorting(node_ids, best_neighbour);

    return {node_ids, best_neighbour}; 
}
