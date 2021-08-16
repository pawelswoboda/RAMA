#include <iostream>
#include <thrust/device_vector.h>
#include <cuda_runtime.h>
#include "maximum_matching_vertex_based.h"
#include "time_measure_util.h"
#include "parallel_gaec_utils.h"

#define numThreads 256

__device__ int get_best_neighbour(const int row_index, float max_value,
                                const int* const __restrict__ row_offsets,
                                const int* const __restrict__ col_ids, 
                                const float* const __restrict__ data,
                                const int* const __restrict__ ignore_vertices)
{
    int n_id = -1;
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
                                    const float max_value,
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

        v_best_neighbours[v] = get_best_neighbour(v, max_value, row_offsets, col_ids, costs, v_matched);
    }
}

__global__ void match_neighbours(const int num_nodes, 
                                const int* const __restrict__ v_best_neighbours,
                                int* __restrict__ v_matched,
                                int* __restrict__ node_mapping,
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
            node_mapping[max(v_best, v)] = min(v_best, v); 
            v_matched[v] = 1;
            v_matched[v_best] = 1;
            *still_running = true;
        }
    }
}

struct pos_part
{
    __host__ __device__
        thrust::tuple<int, float> operator()(const thrust::tuple<int, float>& t)
        {
            if(thrust::get<1>(t) >= 0.0)
                return t;
            return thrust::make_tuple(0, 0.0f);
        }
};

struct tuple_sum
{
    __host__ __device__
        thrust::tuple<int, float> operator()(const thrust::tuple<int, float>& t1, const thrust::tuple<int, float>& t2)
        {
            return {thrust::get<0>(t1) + thrust::get<0>(t2), thrust::get<1>(t1) + thrust::get<1>(t2)};
        }
};

float determine_matching_threshold(const dCOO& A, const float mean_multiplier_mm)
{
    thrust::device_vector<float> data = A.get_data();
    auto first = thrust::make_zip_iterator(thrust::make_tuple(thrust::constant_iterator<int>(1), data.begin()));
    auto last = thrust::make_zip_iterator(thrust::make_tuple(thrust::constant_iterator<int>(1) + data.size(), data.end()));
    // compute average of positive edge costs:
    auto red = thrust::transform_reduce(first, last, pos_part(), thrust::make_tuple(0, 0.0f), tuple_sum());
    return mean_multiplier_mm * thrust::get<1>(red) / thrust::get<0>(red);
}

std::tuple<thrust::device_vector<int>, int> filter_edges_by_matching_vertex_based(const dCOO& A, const float mean_multiplier_mm)
{
    assert(!A.is_directed());
    thrust::device_vector<int> v_best_neighbours(A.rows(), -1);
    thrust::device_vector<int> v_matched(A.rows(), 0);
    thrust::device_vector<int> node_mapping(A.rows());
    thrust::sequence(node_mapping.begin(), node_mapping.end(), 0);

    int numBlocks = ceil(A.rows() / (float) numThreads);
    thrust::device_vector<bool> still_running(1);
    const thrust::device_vector<int> A_row_offsets = A.compute_row_offsets();
    const float min_edge_weight_to_match = determine_matching_threshold(A, mean_multiplier_mm);
    if (min_edge_weight_to_match < 0)
        return {node_mapping, 0}; 
    
    int prev_num_edges = 0;
    for (int t = 0; t < 10; t++)
    {
        thrust::fill(thrust::device, still_running.begin(), still_running.end(), false);

        pick_best_neighbour<<<numBlocks, numThreads>>>(A.rows(), min_edge_weight_to_match,
            thrust::raw_pointer_cast(A_row_offsets.data()), 
            A.get_col_ids_ptr(), 
            A.get_data_ptr(), 
            thrust::raw_pointer_cast(v_matched.data()),
            thrust::raw_pointer_cast(v_best_neighbours.data()));

        match_neighbours<<<numBlocks, numThreads>>>(A.rows(), 
            thrust::raw_pointer_cast(v_best_neighbours.data()),
            thrust::raw_pointer_cast(v_matched.data()),
            thrust::raw_pointer_cast(node_mapping.data()),
            thrust::raw_pointer_cast(still_running.data()));

        int current_num_edges = thrust::reduce(v_matched.begin(), v_matched.end(), 0);
        float rel_increase = (current_num_edges - prev_num_edges) / (prev_num_edges + 1.0f);
        std::cout << "matched sum: " << current_num_edges << ", rel_increase: " << rel_increase <<"\n";
        prev_num_edges = current_num_edges;
        if (!still_running[0] || rel_increase < 0.1)
            break;
    }

    std::cout << "# vertices = " << A.rows() << "\n";
    std::cout << "# matched edges = " << prev_num_edges / 2 << " / "<< A.nnz() / 2 << "\n";
    
    return {node_mapping, prev_num_edges}; 
}
