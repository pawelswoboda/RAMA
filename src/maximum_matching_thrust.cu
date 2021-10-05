#include <iostream>
#include <thrust/device_vector.h>
#include <cuda_runtime.h>
#include "maximum_matching_thrust.h"
#include "time_measure_util.h"
#include "rama_utils.h"

#define numThreads 256

__global__ void match_neighbours(const int num_nodes, 
                                const int* const __restrict__ v_best_neighbours,
                                unsigned char* __restrict__ v_matched,
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
    float* best_neighbours_val;
    __host__ __device__
        void operator()(const thrust::tuple<int, int, float>& t)
        {
            const int n1 = thrust::get<0>(t);
            if(n1 < 0)
                return;
            const int n2 = thrust::get<1>(t);
            if(n2 < 0)
                return;
            v_best_neighbours[n1] = n2;
            best_neighbours_val[n1] = thrust::get<2>(t);
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

struct invalid_edge_indices_func
{
    __host__ __device__
        bool operator()(const thrust::tuple<int, int>& t)
        {
            if (thrust::get<0>(t) < 0)
                return true;
            if (thrust::get<1>(t) < 0)
                return true;
            return false;
        }
};

struct max_reduction_func
{
    const unsigned char* v_matched;
    float min_thresh;
    __host__ __device__
        thrust::tuple<int, int, float> operator()(const thrust::tuple<int, int, float>& t1, const thrust::tuple<int, int, float>& t2) const
        {
            const int r1 = thrust::get<0>(t1);
            if(v_matched[r1] > 0)
                return {r1, -1, -1};

            const int c1 = thrust::get<1>(t1);
            const int c2 = thrust::get<1>(t2);

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

struct get_best_neigh_func
{
    __host__ __device__
        thrust::tuple<int, float> operator()(const thrust::tuple<int, float>& t1, const thrust::tuple<int, float>& t2) const
        {
            const int n1 = thrust::get<0>(t1);
            const int n2 = thrust::get<0>(t2);
            if (n1 < 0 && n2 < 0)
                return {-1, 0.0f};
            else if(n1 < 0)
                return n2;
            else if(n2 < 0)
                return n1;
            else
            {
                const int c1 = thrust::get<1>(t1);
                const int c2 = thrust::get<1>(t2);
                if (c1 < c2)
                    return t2;
                return t1;
            }
        }
};

std::tuple<thrust::device_vector<int>, thrust::device_vector<int>, thrust::device_vector<float>> 
    get_strongest_edge_coo(const dCOO& A, const thrust::device_vector<unsigned char>& v_matched, const float thresh)
{
    const thrust::device_vector<int> row_ids = A.get_row_ids();
    const thrust::device_vector<int> col_ids = A.get_col_ids();
    const thrust::device_vector<float> costs = A.get_data();

    // Make use of COO sorting to directly get the max of each row_id.
    thrust::device_vector<int> row_ids_max(A.max_dim());
    thrust::device_vector<int> col_ids_max(A.max_dim());
    thrust::device_vector<float> costs_max(A.max_dim());
    thrust::equal_to<int> binary_pred;

    auto first_edge = thrust::make_zip_iterator(thrust::make_tuple(row_ids.begin(), col_ids.begin(), costs.begin()));
    auto last_edge = thrust::make_zip_iterator(thrust::make_tuple(row_ids.end(), col_ids.end(), costs.end()));

    auto first_edge_max = thrust::make_zip_iterator(thrust::make_tuple(thrust::make_discard_iterator(), col_ids_max.begin(), costs_max.begin()));
    max_reduction_func max_red({thrust::raw_pointer_cast(v_matched.data()), thresh});

    // For each row find col with maximum similarity:
    auto last_max = thrust::reduce_by_key(row_ids.begin(), row_ids.end(), first_edge, 
                                        row_ids_max.begin(), first_edge_max,
                                        binary_pred, max_red);
    const int num_max = std::distance(first_edge_max, last_max.second);
    row_ids_max.resize(num_max);
    col_ids_max.resize(num_max);
    costs_max.resize(num_max);

    return {row_ids_max, col_ids_max, costs_max};
}

__global__ void pick_best_neighbour_col_to_row(const int num_edges, 
                                                const float thresh,
                                                const int* const __restrict__ row_ids, 
                                                const int* const __restrict__ col_ids, 
                                                const float* const __restrict__ costs,
                                                const unsigned char* const __restrict__ v_matched, 
                                                int* __restrict__ v_best_neighbour,
                                                float* __restrict__ v_best_val)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int num_threads = blockDim.x * gridDim.x;
    for (int e = tid; e < num_edges; e += num_threads)
    {
        const int row = row_ids[e];
        const int col = col_ids[e];

        if (v_matched[row] || v_matched[col])
            continue;

        const float cost = costs[e];
        if (cost < thresh)
            continue;
        // col will send proposal to its best row.
        if (atomicMax(&v_best_val[col], cost) < cost)
            v_best_neighbour[col] = row;
    }
}

thrust::device_vector<int> get_strongest_neighbours(const dCOO& A, const thrust::device_vector<unsigned char>& v_matched, const float thresh)
{
    thrust::device_vector<int> best_r_red;
    thrust::device_vector<int> best_c_red;
    {
        thrust::device_vector<int> best_r;
        thrust::device_vector<int> best_c;
        thrust::device_vector<float> best_val;
        std::tie(best_r, best_c, best_val) = get_strongest_edge_coo(A, v_matched, thresh);

        thrust::device_vector<int> best_neighbour_index_invert(A.max_dim(), -1);
        thrust::device_vector<float> best_val_invert(A.max_dim(), 0.0f);

        int numBlocks = ceil(A.nnz() / (float) numThreads);
        pick_best_neighbour_col_to_row<<<numBlocks, numThreads>>>(A.nnz(), thresh, 
                A.get_row_ids_ptr(),
                A.get_col_ids_ptr(),
                A.get_data_ptr(),
                thrust::raw_pointer_cast(v_matched.data()),
                thrust::raw_pointer_cast(best_neighbour_index_invert.data()),
                thrust::raw_pointer_cast(best_val_invert.data()));

        // Now merge incoming info with this:
        thrust::device_vector<int> seq(A.max_dim());
        thrust::sequence(seq.begin(), seq.end());

        best_r = concatenate(seq, best_r);
        best_c = concatenate(best_neighbour_index_invert, best_c);
        seq.clear();
        seq.shrink_to_fit();
        
        best_val = concatenate(best_val_invert, best_val);
        best_val_invert.clear();
        best_val_invert.shrink_to_fit();

        coo_sorting(best_r, best_c, best_val);

        auto first_val = thrust::make_zip_iterator(thrust::make_tuple(best_c.begin(), best_val.begin()));
        
        best_r_red = thrust::device_vector<int>(A.max_dim());
        best_c_red = thrust::device_vector<int>(A.max_dim());
        auto first_val_out = thrust::make_zip_iterator(thrust::make_tuple(best_c_red.begin(), thrust::make_discard_iterator()));

        thrust::equal_to<int> binary_pred;
        auto red_end = thrust::reduce_by_key(best_r.begin(), best_r.end(), first_val, best_r_red.begin(), first_val_out, binary_pred, get_best_neigh_func());
        int num_reduced = std::distance(best_r_red.begin(), red_end.first);
        best_r_red.resize(num_reduced);
        best_c_red.resize(num_reduced);
    }

    auto first_red = thrust::make_zip_iterator(thrust::make_tuple(best_r_red.begin(), best_c_red.begin()));
    auto last_red = thrust::make_zip_iterator(thrust::make_tuple(best_r_red.end(), best_c_red.end()));

    auto last_valid_red = thrust::remove_if(first_red, last_red, invalid_edge_indices_func());
    int num_valid = std::distance(first_red, last_valid_red);

    thrust::device_vector<int> v_best_neighbours(A.max_dim(), -1);
    thrust::scatter(best_c_red.begin(), best_c_red.begin() + num_valid, best_r_red.begin(), v_best_neighbours.begin());
    return v_best_neighbours;
}

template<typename T1, typename T2>
struct char2int
{
    __host__ __device__ T2 operator()(const T1 &x) const
    {
    return static_cast<T2>(x);
    }
};

std::tuple<thrust::device_vector<int>, int> filter_edges_by_matching_thrust(const dCOO& A, const float mean_multiplier_mm)
{
    assert(A.is_directed());
    thrust::device_vector<int> node_mapping(A.max_dim());
    thrust::sequence(node_mapping.begin(), node_mapping.end(), 0);

    const float min_edge_weight_to_match = determine_matching_threshold(A, mean_multiplier_mm); //TODO: Remove edges below this thresh?
    std::cout<<"min_edge_weight_to_match: "<<min_edge_weight_to_match<<"\n";
    if (min_edge_weight_to_match < 0)
        return {node_mapping, 0}; 

    thrust::device_vector<unsigned char> v_matched(A.max_dim(), 0);

    int numBlocks = ceil(A.max_dim() / (float) numThreads);
    thrust::device_vector<bool> still_running(1);

    thrust::device_vector<int> v_best_neighbours;

    int prev_num_edges = 0;
    for (int t = 0; t < 10; t++)
    {
        thrust::fill(thrust::device, still_running.begin(), still_running.end(), false);
        v_best_neighbours = get_strongest_neighbours(A, v_matched, min_edge_weight_to_match);

        match_neighbours<<<numBlocks, numThreads>>>(A.max_dim(), 
            thrust::raw_pointer_cast(v_best_neighbours.data()),
            thrust::raw_pointer_cast(v_matched.data()),
            thrust::raw_pointer_cast(node_mapping.data()),
            thrust::raw_pointer_cast(still_running.data()));

        int current_num_edges = thrust::transform_reduce(v_matched.begin(), v_matched.end(), char2int<unsigned char,int>(), 0, thrust::plus<int>());
        float rel_increase = (current_num_edges - prev_num_edges) / (prev_num_edges + 1.0f);
        std::cout << "matched sum: " << current_num_edges << ", rel_increase: " << rel_increase <<"\n";
        prev_num_edges = current_num_edges;

        if (!still_running[0] || rel_increase < 0.1)
            break;
    }

    std::cout << "# vertices = " << A.max_dim() << "\n";
    std::cout << "# matched edges = " << prev_num_edges << " / "<< A.nnz() << "\n";
    
    return {node_mapping, prev_num_edges}; 
}
