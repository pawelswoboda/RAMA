#include <iostream>
#include <thrust/device_vector.h>
#include <cuda_runtime.h>
#include "maximum_matching_vertex_based.h"
#include "../time_measure_util.h"
#include "../utils.h"

#define numThreads 256

__device__ int get_best_neighbour(const int row_index,
                                const int* const __restrict__ row_offsets,
                                const int* const __restrict__ col_ids, 
                                const float* const __restrict__ data,
                                const int* const __restrict__ ignore_vertices)
{
    int n_id = -1;
    float max_value = 0.0;
    for(int l = row_offsets[row_index]; l < row_offsets[row_index + 1]; ++l)
    {
        float current_val = data[l];
        int current_n = col_ids[l];
        if (current_val <= max_value || ignore_vertices[current_n] || current_n == row_index) // self edges can be present with 0 cost.
            continue;

        max_value = current_val;
        n_id = current_n; 
    }
    return n_id;
}

__global__ void pick_best_neighbour(const int num_nodes, 
                                    const int* const __restrict__ row_offsets, 
                                    const int* const __restrict__ col_ids, 
                                    const float* const __restrict__ costs,
                                    const int* const __restrict__ v_matched, 
                                    int* __restrict__ v_best_neighbours)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int num_threads = blockDim.x * gridDim.x;
    for (int v = tid; v < num_nodes; v += num_threads)
    {
        if (v_matched[v])
            continue;

        v_best_neighbours[v] = get_best_neighbour(v, row_offsets, col_ids, costs, v_matched);
    }
}

__global__ void match_neighbours(const int num_nodes, 
                                const int* const __restrict__ v_best_neighbours,
                                int* __restrict__ v_matched,
                                bool* __restrict__ still_running)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int num_threads = blockDim.x * gridDim.x;
    for (int v = tid; v < num_nodes; v += num_threads)
    {
        if (v_matched[v])
            continue;

        int v_best = v_best_neighbours[v];
        if (v_best == -1 || v_matched[v_best])
            continue;
            
        if (v_best_neighbours[v_best] == v)
        {
            v_matched[v] = 1;
            v_matched[v_best] = 1;
            *still_running = true;
        }
    }
}

struct is_unmatched {
    __host__ __device__
        inline int operator()(const thrust::tuple<int,int,int> e)
        {
            return thrust::get<2>(e) == 0;
        }
};

std::tuple<thrust::device_vector<int>, thrust::device_vector<int>> filter_edges_by_matching_vertex_based(const dCSR& A)
{
    thrust::device_vector<int> v_best_neighbours(A.rows(), -1);
    thrust::device_vector<int> v_matched(A.rows(), 0);

    int numBlocks = ceil(A.rows() / (float) numThreads);
    thrust::device_vector<bool> still_running(1);

    for (int t = 0; t < 5; t++)
    {
        thrust::fill(thrust::device, still_running.begin(), still_running.end(), false);

        pick_best_neighbour<<<numBlocks, numThreads>>>(A.rows(), 
            thrust::raw_pointer_cast(A.get_row_offsets().data()), 
            thrust::raw_pointer_cast(A.get_col_ids().data()), 
            thrust::raw_pointer_cast(A.get_data().data()), 
            thrust::raw_pointer_cast(v_matched.data()),
            thrust::raw_pointer_cast(v_best_neighbours.data()));

        match_neighbours<<<numBlocks, numThreads>>>(A.rows(), 
            thrust::raw_pointer_cast(v_best_neighbours.data()),
            thrust::raw_pointer_cast(v_matched.data()),
            thrust::raw_pointer_cast(still_running.data()));

        std::cout << "matched sum = " << thrust::reduce(v_matched.begin(), v_matched.end(), 0) << "\n";
        if (!still_running[0])
            break;
    }
    thrust::device_vector<int> matched_rows(A.rows());
    thrust::sequence(matched_rows.begin(), matched_rows.end(), 0);

    auto first_m = thrust::make_zip_iterator(thrust::make_tuple(matched_rows.begin(), v_best_neighbours.begin(), v_matched.begin()));
    auto last_m = thrust::make_zip_iterator(thrust::make_tuple(matched_rows.end(), v_best_neighbours.end(), v_matched.end()));
    auto matched_last = thrust::remove_if(first_m, last_m, is_unmatched());
    const int nr_matched_edges = std::distance(first_m, matched_last);
    matched_rows.resize(nr_matched_edges);
    v_best_neighbours.resize(nr_matched_edges);

    std::cout << "# vertices = " << A.rows() << "\n";
    std::cout << "# matched edges = " << nr_matched_edges / 2 << " / "<< A.nnz() / 2 << "\n";
    
    // thrust::copy(matched_rows.begin(), matched_rows.end(), std::ostream_iterator<int>(std::cout, " "));
    // std::cout<<"\n";
    // thrust::copy(v_best_neighbours.begin(), v_best_neighbours.end(), std::ostream_iterator<int>(std::cout, " "));
    // std::cout<<"\n";

    return {matched_rows, v_best_neighbours}; 
}