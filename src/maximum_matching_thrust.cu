#include <iostream>
#include <thrust/device_vector.h>
#include <cuda_runtime.h>
#include "maximum_matching_thrust.h"
#include "time_measure_util.h"
#include "rama_utils.h"

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

        const int v_best = v_best_neighbours[v];
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

struct set_neighbour_func
{
    int* v_best_neighbours;
    __host__ __device__
        void operator()(const thrust::tuple<int, int>& t)
        {
            const int n1 = thrust::get<0>(t);
            const int n2 = thrust::get<1>(t);
            if(n1 < 0 || n2 < 0)
                return;
            v_best_neighbours[n1] = n2;
        }
};

struct invalid_edge_func
{
    int* v_matched;
    float min_thresh;
    __host__ __device__
        bool operator()(const thrust::tuple<int, int, float>& t)
        {
            const int n1 = thrust::get<0>(t);
            const int n2 = thrust::get<1>(t);
            const float cost = thrust::get<2>(t);
            return v_matched[n1] > 0 || v_matched[n2] > 0 || cost < min_thresh;
        }
};

struct max_reduction_func
{
    int* v_matched;
    float min_thresh;
    int num_v;
    __host__ __device__
        thrust::tuple<int, int, float> operator()(const thrust::tuple<int, int, float>& t1, const thrust::tuple<int, int, float>& t2) const
        {
            const int r1 = thrust::get<0>(t1);
            assert(r1 < num_v);
            if(v_matched[r1] > 0)
                return {r1, -1, -1};

            const int c1 = thrust::get<1>(t1);
            const int c2 = thrust::get<1>(t2);
            assert(c1 < num_v);
            assert(c2 < num_v);
            assert(r1 != c1);
            assert(r1 != c2);
            const float cost1 = thrust::get<2>(t1);
            const float cost2 = thrust::get<2>(t2);
            bool c1_valid = false;
            if (c1 >= 0)
                c1_valid = v_matched[c1] == 0 && cost1 >= min_thresh;
            bool c2_valid = false;
            if (c2 >= 0)
                c2_valid = v_matched[c2] == 0 && cost2 >= min_thresh;

            if(!c1_valid && !c2_valid)
                return {r1, -1, -1};

            else if(c1_valid && !c2_valid)
                return t1;

            else if(!c1_valid && c2_valid)
                return t2;

            else if(cost1 < cost2)
                return t2;
            return t1;
        }
};

float determine_matching_threshold(const dCOO& A, const float mean_multiplier_mm)
{
    thrust::device_vector<float> data = A.get_data();
    auto first = thrust::make_zip_iterator(thrust::make_tuple(thrust::constant_iterator<int>(1), data.begin()));
    auto last = thrust::make_zip_iterator(thrust::make_tuple(thrust::constant_iterator<int>(1) + data.size(), data.end()));
    // compute average of positive edge costs:
    auto red = thrust::transform_reduce(first, last, pos_part(), thrust::make_tuple(0, 0.0f), tuple_sum());
    if (thrust::get<0>(red) == 0)
        return -1.0;
    return mean_multiplier_mm * thrust::get<1>(red) / thrust::get<0>(red);
}

std::tuple<thrust::device_vector<int>, int> filter_edges_by_matching_thrust(const dCOO& A, const float mean_multiplier_mm)
{
    assert(!A.is_directed());
    thrust::device_vector<int> node_mapping(A.rows());
    thrust::sequence(node_mapping.begin(), node_mapping.end(), 0);

    const float min_edge_weight_to_match = determine_matching_threshold(A, mean_multiplier_mm); //TODO: Remove edges below this thresh?
    std::cout<<"min_edge_weight_to_match: "<<min_edge_weight_to_match<<"\n";
    if (min_edge_weight_to_match < 0)
        return {node_mapping, 0}; 

    thrust::device_vector<int> v_best_neighbours(A.rows(), -1);
    thrust::device_vector<int> v_matched(A.rows(), 0);

    thrust::device_vector<int> row_ids = A.get_row_ids();
    thrust::device_vector<int> col_ids = A.get_col_ids();
    thrust::device_vector<float> costs = A.get_data();

    int valid_num_edges = row_ids.size();

    thrust::device_vector<int> row_ids_max(row_ids.size());
    thrust::device_vector<int> col_ids_max(row_ids.size());
    thrust::device_vector<int> row_ids_max_temp(row_ids.size());
    thrust::device_vector<float> costs_max(row_ids.size());

    thrust::equal_to<int> binary_pred;

    int numBlocks = ceil(A.rows() / (float) numThreads);
    thrust::device_vector<bool> still_running(1);

    int prev_num_edges = 0;
    for (int t = 0; t < 10; t++)
    {
        thrust::fill(thrust::device, still_running.begin(), still_running.end(), false);

        auto first_edge = thrust::make_zip_iterator(thrust::make_tuple(row_ids.begin(), col_ids.begin(), costs.begin()));
        auto last_edge = thrust::make_zip_iterator(thrust::make_tuple(row_ids.begin() + valid_num_edges, col_ids.begin() + valid_num_edges, costs.begin() + valid_num_edges));

        invalid_edge_func invalid_edge({thrust::raw_pointer_cast(v_matched.data()), min_edge_weight_to_match});
        auto last_valid_edge = thrust::remove_if(first_edge, last_edge, invalid_edge);
        valid_num_edges = std::distance(first_edge, last_valid_edge);
        row_ids.resize(valid_num_edges);
        col_ids.resize(valid_num_edges);
        costs.resize(valid_num_edges);

        auto first_edge_max = thrust::make_zip_iterator(thrust::make_tuple(row_ids_max_temp.begin(), col_ids_max.begin(), costs_max.begin()));
        max_reduction_func max_red({thrust::raw_pointer_cast(v_matched.data()), min_edge_weight_to_match, A.rows()});

        // For each row find col with maximum similarity:
        auto last_max = thrust::reduce_by_key(row_ids.begin(), row_ids.begin() + valid_num_edges, first_edge, 
                                            row_ids_max.begin(), first_edge_max,
                                            binary_pred, max_red);
        const int num_max = std::distance(first_edge_max, last_max.second);

        auto first_max_edge = thrust::make_zip_iterator(thrust::make_tuple(row_ids_max.begin(), col_ids_max.begin()));
        auto last_max_edge = thrust::make_zip_iterator(thrust::make_tuple(row_ids_max.begin() + num_max, col_ids_max.begin() + num_max));

        set_neighbour_func set_neigh({thrust::raw_pointer_cast(v_best_neighbours.data())});
        thrust::for_each(first_max_edge, last_max_edge, set_neigh);
        // pick_best_neighbour<<<numBlocks, numThreads>>>(A.rows(), min_edge_weight_to_match,
        //     thrust::raw_pointer_cast(A_row_offsets.data()), 
        //     A.get_col_ids_ptr(), 
        //     A.get_data_ptr(), 
        //     thrust::raw_pointer_cast(v_matched.data()),
        //     thrust::raw_pointer_cast(v_best_neighbours.data()));

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

        thrust::fill(v_best_neighbours.begin(), v_best_neighbours.end(), -1);
    }

    std::cout << "# vertices = " << A.rows() << "\n";
    std::cout << "# matched edges = " << prev_num_edges / 2 << " / "<< A.nnz() / 2 << "\n";
    
    return {node_mapping, prev_num_edges}; 
}
