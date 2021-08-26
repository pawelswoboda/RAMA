#include "edge_contractions_woc.h"
#include "edge_contractions_woc_thrust.h"
#include <thrust/binary_search.h>
#include <gpuMST.h>
#include "ECLgraph.h"
#include "rama_utils.h"

__global__ void check_triangles_cuda(const int num_rep_edges,
    const int* const __restrict__ rep_row_ids,
    const int* const __restrict__ rep_col_ids,
    const int* const __restrict__ mst_row_offsets,
    const int* const __restrict__ mst_col_ids,
    const float* const __restrict__ mst_data,
    bool* __restrict__ mst_edge_valid)
{
    const int start_index = blockIdx.x * blockDim.x + threadIdx.x;
    const int num_threads = blockDim.x * gridDim.x;
    for (int edge = start_index; edge < num_rep_edges; edge += num_threads) 
    {
        const int v1 = rep_row_ids[edge];
        const int v2 = rep_col_ids[edge];
        int v1_mid_edge_index = mst_row_offsets[v1];
        int v2_mid_edge_index = mst_row_offsets[v2];
        const int mid = compute_lowest_common_neighbour(v1, v2, mst_row_offsets, mst_col_ids, mst_data, v1_mid_edge_index, v2_mid_edge_index);
        if (mid >= 0)
        {
            const float v1_mid_data = mst_data[v1_mid_edge_index - 1];
            const float v2_mid_data = mst_data[v2_mid_edge_index - 1];
            if (v1_mid_data < v2_mid_data)
                mst_edge_valid[v1_mid_edge_index - 1] = false;
            else
                mst_edge_valid[v2_mid_edge_index - 1] = false;
        }
    }
}

__global__ void check_quadrangles_cuda(const long num_expansions, const int num_rep_edges,
        const int* const __restrict__ rep_row_ids,
        const int* const __restrict__ rep_col_ids,
        const long* const __restrict__ rep_row_offsets,
        const int* const __restrict__ mst_row_offsets,
        const int* const __restrict__ mst_col_ids,
        const float* const __restrict__ mst_data,
        bool* __restrict__ mst_edge_valid)
{
    const long start_index = blockIdx.x * blockDim.x + threadIdx.x;
    const long num_threads = blockDim.x * gridDim.x;
    for (long c_index = start_index; c_index < num_expansions; c_index += num_threads) 
    {
        const long* next_rep_row_location = thrust::upper_bound(thrust::seq, rep_row_offsets, rep_row_offsets + num_rep_edges + 1, c_index);
        const long rep_edge_index = thrust::distance(rep_row_offsets, next_rep_row_location) - 1;
        assert(rep_edge_index < num_rep_edges && rep_edge_index >= 0);
        const long local_offset = c_index - rep_row_offsets[rep_edge_index];
        assert(local_offset >= 0);
        const int v1 = rep_row_ids[rep_edge_index];
        const int v2 = rep_col_ids[rep_edge_index];
        const int v1_v1_n1_edge_index = mst_row_offsets[v1] + local_offset;
        const int v1_n1 = mst_col_ids[v1_v1_n1_edge_index];
        int v1_n1_mid_edge_index = mst_row_offsets[v1_n1];
        int v2_mid_edge_index = mst_row_offsets[v2];

        const int mid = compute_lowest_common_neighbour(v1_n1, v2, mst_row_offsets, mst_col_ids, mst_data, v1_n1_mid_edge_index, v2_mid_edge_index);

        if (mid >= 0)
        {
            const float v1_v1_n1_data = mst_data[v1_v1_n1_edge_index];
            const float v1_mid_data = mst_data[v1_n1_mid_edge_index - 1];
            const float v2_mid_data = mst_data[v2_mid_edge_index - 1];
            if (v1_v1_n1_data < v1_mid_data && v1_v1_n1_data < v2_mid_data)
                mst_edge_valid[v1_v1_n1_edge_index] = false;
            else if (v1_mid_data < v2_mid_data)
                mst_edge_valid[v1_n1_mid_edge_index - 1] = false;
            else
                mst_edge_valid[v2_mid_edge_index - 1] = false;
        }
    }
}

__global__ void check_pentagons_cuda(const long num_expansions, const int num_rep_edges,
        const int* const __restrict__ rep_row_ids,
        const int* const __restrict__ rep_col_ids,
        const long* const __restrict__ rep_edge_offsets,
        const int* const __restrict__ mst_row_offsets,
        const int* const __restrict__ mst_col_ids,
        const float* const __restrict__ mst_data,
        bool* __restrict__ mst_edge_valid)
{
    const long start_index = blockIdx.x * blockDim.x + threadIdx.x;
    const long num_threads = blockDim.x * gridDim.x;
    for (long c_index = start_index; c_index < num_expansions; c_index += num_threads) 
    {
        const long* next_rep_edge_location = thrust::upper_bound(thrust::seq, rep_edge_offsets, rep_edge_offsets + num_rep_edges + 1, c_index);
        const long rep_edge_index = thrust::distance(rep_edge_offsets, next_rep_edge_location) - 1;
        assert(rep_edge_index < num_rep_edges);
        const long local_offset = c_index - rep_edge_offsets[rep_edge_index];
        assert(local_offset >= 0);
        const int v1 = rep_row_ids[rep_edge_index];
        const int v2 = rep_col_ids[rep_edge_index];
        const int v1_degree = mst_row_offsets[v1 + 1] - mst_row_offsets[v1];
        const int l1 = local_offset % v1_degree;
        const int l2 = local_offset / v1_degree;
        const int v1_v1_n1_edge_index = mst_row_offsets[v1] + l1;
        const int v1_n1 = mst_col_ids[v1_v1_n1_edge_index];
        const int v2_v2_n1_edge_index = mst_row_offsets[v2] + l2;
        const int v2_n1 = mst_col_ids[v2_v2_n1_edge_index];
        if (v1_n1 == v2_n1)
            continue;
        int v1_n1_mid_edge_index = mst_row_offsets[v1_n1];
        int v2_n1_mid_edge_index = mst_row_offsets[v2_n1];
        const int mid = compute_lowest_common_neighbour(v1_n1, v2_n1, mst_row_offsets, mst_col_ids, mst_data, v1_n1_mid_edge_index, v2_n1_mid_edge_index);
        if (mid >= 0)
        {
            const float v1_v1_n1_data = mst_data[v1_v1_n1_edge_index];
            const float v2_v2_n1_data = mst_data[v2_v2_n1_edge_index];
            const float v1_n1_mid_data = mst_data[v1_n1_mid_edge_index - 1];
            const float v2_n1_mid_data = mst_data[v2_n1_mid_edge_index - 1];
            if (v1_v1_n1_data < v2_v2_n1_data)
            {
                if (v1_v1_n1_data < v1_n1_mid_data && v1_v1_n1_data < v2_n1_mid_data)
                    mst_edge_valid[v1_v1_n1_edge_index] = false;
                else if (v1_n1_mid_data < v2_n1_mid_data)
                    mst_edge_valid[v1_n1_mid_edge_index - 1] = false;
                else
                    mst_edge_valid[v2_n1_mid_edge_index - 1] = false;
            }
            else
            {
                if (v2_v2_n1_data < v1_n1_mid_data && v2_v2_n1_data < v2_n1_mid_data)
                    mst_edge_valid[v2_v2_n1_edge_index] = false;
                else if (v1_n1_mid_data < v2_n1_mid_data)
                    mst_edge_valid[v1_n1_mid_edge_index - 1] = false;
                else
                    mst_edge_valid[v2_n1_mid_edge_index - 1] = false;
            }
        }
    }
}

bool edge_contractions_woc::check_triangles()
{
    const int num_rep_edges = rep_row_ids.size();
    if(num_rep_edges == 0)
        return false;

    thrust::device_vector<bool> mst_edge_valid(mst_data.size(), true);
    const int threadCount = 256;
    const int blockCount = ceil(num_rep_edges / (float) threadCount);

    check_triangles_cuda<<<blockCount, threadCount>>>(num_rep_edges, 
        thrust::raw_pointer_cast(rep_row_ids.data()), 
        thrust::raw_pointer_cast(rep_col_ids.data()), 
        thrust::raw_pointer_cast(mst_row_offsets.data()),
        thrust::raw_pointer_cast(mst_col_ids.data()),
        thrust::raw_pointer_cast(mst_data.data()),
        thrust::raw_pointer_cast(mst_edge_valid.data()));

    bool any_removed = remove_mst_by_mask(mst_edge_valid);
    return any_removed;
}

bool edge_contractions_woc::check_quadrangles()
{
    const int num_rep_edges = rep_row_ids.size();
    if(num_rep_edges == 0)
        return false;

    thrust::device_vector<bool> mst_edge_valid(mst_data.size(), true);
    thrust::device_vector<long> rep_row_offsets(num_rep_edges + 1);
    {
        const thrust::device_vector<int> vertex_degrees = offsets_to_degrees(mst_row_offsets);
        thrust::gather(rep_row_ids.begin(), rep_row_ids.end(), vertex_degrees.begin(), rep_row_offsets.begin());

        rep_row_offsets.back() = 0;
        thrust::exclusive_scan(rep_row_offsets.begin(), rep_row_offsets.end(), rep_row_offsets.begin());
    }
    const long num_expansions = rep_row_offsets.back();
    const int threadCount = 256;
    const int blockCount = ceil(num_expansions / (float) threadCount);

    check_quadrangles_cuda<<<blockCount, threadCount>>>(num_expansions, num_rep_edges, 
        thrust::raw_pointer_cast(rep_row_ids.data()), 
        thrust::raw_pointer_cast(rep_col_ids.data()), 
        thrust::raw_pointer_cast(rep_row_offsets.data()),         
        thrust::raw_pointer_cast(mst_row_offsets.data()),
        thrust::raw_pointer_cast(mst_col_ids.data()),
        thrust::raw_pointer_cast(mst_data.data()),
        thrust::raw_pointer_cast(mst_edge_valid.data()));

    bool any_removed = remove_mst_by_mask(mst_edge_valid);
    return any_removed;
}

bool edge_contractions_woc::check_pentagons()
{
    const int num_rep_edges = rep_row_ids.size();
    if(num_rep_edges == 0)
        return false;

    thrust::device_vector<bool> mst_edge_valid(mst_data.size(), true);
    thrust::device_vector<long> rep_edge_offsets(num_rep_edges + 1);
    {
        const thrust::device_vector<int> vertex_degrees = offsets_to_degrees(mst_row_offsets);
        thrust::device_vector<int> row_ids_degrees(num_rep_edges);
        thrust::gather(rep_row_ids.begin(), rep_row_ids.end(), vertex_degrees.begin(), row_ids_degrees.begin());
        thrust::device_vector<int> col_ids_degrees(num_rep_edges);
        thrust::gather(rep_col_ids.begin(), rep_col_ids.end(), vertex_degrees.begin(), col_ids_degrees.begin());

        thrust::transform(row_ids_degrees.begin(), row_ids_degrees.end(), col_ids_degrees.begin(), rep_edge_offsets.begin(), thrust::multiplies<int>());
        rep_edge_offsets.back() = 0;
        thrust::exclusive_scan(rep_edge_offsets.begin(), rep_edge_offsets.end(), rep_edge_offsets.begin());
    }
    const long num_expansions = rep_edge_offsets.back();
    const int threadCount = 256;
    const int blockCount = ceil(num_expansions / (float) threadCount);

    check_pentagons_cuda<<<blockCount, threadCount>>>(num_expansions, num_rep_edges,
        thrust::raw_pointer_cast(rep_row_ids.data()),
        thrust::raw_pointer_cast(rep_col_ids.data()),
        thrust::raw_pointer_cast(rep_edge_offsets.data()), 
        thrust::raw_pointer_cast(mst_row_offsets.data()),
        thrust::raw_pointer_cast(mst_col_ids.data()),
        thrust::raw_pointer_cast(mst_data.data()),
        thrust::raw_pointer_cast(mst_edge_valid.data()));

    bool any_removed = remove_mst_by_mask(mst_edge_valid);
    return any_removed;
}

struct is_positive_edge
{
    __host__ __device__
        bool operator()(const thrust::tuple<int,int,float> t)
        {
            return thrust::get<2>(t) > 0;
        }
};

struct edge_index_in_same_cc_func
{
    const int* cc_labels;
    const int* row_ids;
    const int* col_ids;
    __host__ __device__
        bool operator()(const int edge_index)
        {
            const int i = row_ids[edge_index];
            const int j = col_ids[edge_index];
            return cc_labels[i] == cc_labels[j];
        }
};

struct edge_in_diff_cc_func
{
    const int* cc_labels;
    __host__ __device__
        bool operator()(const thrust::tuple<int, int>& t)
        {
            const int i = thrust::get<0>(t);
            const int j = thrust::get<1>(t);
            return cc_labels[i] != cc_labels[j];
        }
};

void edge_contractions_woc::filter_by_cc()
{
    assert(cc_labels.size() == num_nodes);
    computeCC_gpu(num_nodes, mst_row_ids.size(),
            thrust::raw_pointer_cast(mst_row_offsets.data()),
            thrust::raw_pointer_cast(mst_col_ids.data()),
            thrust::raw_pointer_cast(cc_labels.data()), 
            get_cuda_device());
    
    edge_in_diff_cc_func edge_in_diff_cc({thrust::raw_pointer_cast(cc_labels.data())});

    auto first_rep = thrust::make_zip_iterator(thrust::make_tuple(rep_row_ids.begin(), rep_col_ids.begin()));
    auto last_rep = thrust::make_zip_iterator(thrust::make_tuple(rep_row_ids.end(), rep_col_ids.end()));

    auto last_invalid = thrust::remove_if(first_rep, last_rep, edge_in_diff_cc);
    int num_rep_edges = std::distance(first_rep, last_invalid);

    rep_row_ids.resize(num_rep_edges);
    rep_col_ids.resize(num_rep_edges);
}

struct is_invalid_edge_func
{
    __host__ __device__
        bool operator()(const thrust::tuple<int,int,float,bool> t)
        {
            return !thrust::get<3>(t);
        }
};

bool edge_contractions_woc::remove_mst_by_mask(thrust::device_vector<bool>& edge_valid_mask)
{
    const int mst_size = mst_data.size();

    thrust::device_vector<int> mst_i_to_remove(mst_size);
    thrust::device_vector<int> mst_j_to_remove(mst_size);
    {
        auto first = thrust::make_zip_iterator(thrust::make_tuple(mst_row_ids.begin(), mst_col_ids.begin(), mst_data.begin(), edge_valid_mask.begin()));
        auto last = thrust::make_zip_iterator(thrust::make_tuple(mst_row_ids.end(), mst_col_ids.end(), mst_data.end(), edge_valid_mask.end()));
    
        auto first_to_remove = thrust::make_zip_iterator(thrust::make_tuple(mst_i_to_remove.begin(), mst_j_to_remove.begin(), thrust::make_discard_iterator(), thrust::make_discard_iterator()));

        auto new_last = thrust::copy_if(first, last, first_to_remove, is_invalid_edge_func());
        const int num_mst_to_remove = std::distance(first_to_remove, new_last);

        if (num_mst_to_remove == 0)
            return false;

        mst_i_to_remove.resize(num_mst_to_remove);
        mst_j_to_remove.resize(num_mst_to_remove);
        
        std::tie(mst_i_to_remove, mst_j_to_remove) = to_undirected(mst_i_to_remove, mst_j_to_remove);
        coo_sorting(mst_i_to_remove, mst_j_to_remove);
    }

    auto first_mst = thrust::make_zip_iterator(thrust::make_tuple(mst_row_ids.begin(), mst_col_ids.begin()));
    auto last_mst = thrust::make_zip_iterator(thrust::make_tuple(mst_row_ids.end(), mst_col_ids.end()));
    
    auto first_mst_to_remove = thrust::make_zip_iterator(thrust::make_tuple(mst_i_to_remove.begin(), mst_j_to_remove.begin()));
    auto last_mst_to_remove = thrust::make_zip_iterator(thrust::make_tuple(mst_i_to_remove.end(), mst_j_to_remove.end()));

    thrust::device_vector<int> mst_row_ids_valid(mst_row_ids.size());
    thrust::device_vector<int> mst_col_ids_valid(mst_col_ids.size());
    thrust::device_vector<float> mst_data_valid(mst_data.size());

    auto first_mst_valid_key = thrust::make_zip_iterator(thrust::make_tuple(mst_row_ids_valid.begin(), mst_col_ids_valid.begin()));

    auto last_to_keep = thrust::set_difference_by_key(first_mst, last_mst, first_mst_to_remove, last_mst_to_remove, 
                                                    mst_data.begin(), thrust::make_constant_iterator<float>(0),
                                                    first_mst_valid_key, mst_data_valid.begin());

    int num_valid_mst_edges = std::distance(first_mst_valid_key, last_to_keep.first);
    mst_row_ids_valid.resize(num_valid_mst_edges);
    mst_col_ids_valid.resize(num_valid_mst_edges);
    mst_data_valid.resize(num_valid_mst_edges);

    thrust::swap(mst_row_ids_valid, mst_row_ids);
    thrust::swap(mst_col_ids_valid, mst_col_ids);
    thrust::swap(mst_data_valid, mst_data);

    mst_row_offsets = compute_offsets(mst_row_ids, num_nodes - 1);
    return true;
}

edge_contractions_woc::edge_contractions_woc(const dCOO& A) : num_nodes(A.max_dim())
{
    cc_labels = thrust::device_vector<int>(num_nodes);

    // 1. Parition into positive and negative edges:
    assert(A.is_directed());
    const thrust::device_vector<int> row_ids = A.get_row_ids();
    const thrust::device_vector<int> col_ids = A.get_col_ids();
    const thrust::device_vector<float> data = A.get_data();

    auto first = thrust::make_zip_iterator(thrust::make_tuple(row_ids.begin(), col_ids.begin(), data.begin()));
    auto last = thrust::make_zip_iterator(thrust::make_tuple(row_ids.end(), col_ids.end(), data.end()));

    thrust::device_vector<int> pos_row_ids(row_ids.size());
    thrust::device_vector<int> pos_col_ids(col_ids.size());
    thrust::device_vector<float> pos_data(data.size());
    auto first_pos = thrust::make_zip_iterator(thrust::make_tuple(pos_row_ids.begin(), pos_col_ids.begin(), pos_data.begin()));

    rep_row_ids = thrust::device_vector<int>(row_ids.size());
    rep_col_ids = thrust::device_vector<int>(col_ids.size());
    auto first_rep = thrust::make_zip_iterator(thrust::make_tuple(rep_row_ids.begin(), rep_col_ids.begin(), thrust::make_discard_iterator()));

    auto ends = thrust::partition_copy(first, last, first_pos, first_rep, is_positive_edge());

    const int num_positive = std::distance(first_pos, ends.first);
    if (num_positive == 0)
        return; 

    pos_row_ids.resize(num_positive);
    pos_col_ids.resize(num_positive);
    pos_data.resize(num_positive);

    const int num_negative = std::distance(first_rep, ends.second);
    rep_row_ids.resize(num_negative);
    rep_col_ids.resize(num_negative);

    // 2. Compute maximum spanning tree in attractive edges.
    std::tie(mst_row_ids, mst_col_ids, mst_data) = MST_boruvka::maximum_spanning_tree(pos_row_ids, pos_col_ids, pos_data);
    std::tie(mst_row_ids, mst_col_ids, mst_data) = to_undirected(mst_row_ids, mst_col_ids, mst_data);
    coo_sorting(mst_row_ids, mst_col_ids, mst_data);

    mst_row_offsets = compute_offsets(mst_row_ids, num_nodes - 1);
}

// Uses specialized implementation for finding 3,4,5 cycles which uses less memory than thrust based one.
// For larger cycles it uses the general thrust based implementation.
std::tuple<thrust::device_vector<int>, int> edge_contractions_woc::find_contraction_mapping()
{
    MEASURE_CUMULATIVE_FUNCTION_EXECUTION_TIME;

    if (mst_row_ids.size() == 0)
        return {thrust::device_vector<int>(0), 0};

    std::cout<<"# MST edges "<<mst_row_ids.size()<<", # Repulsive edges "<<rep_row_ids.size()<<"\n";
    
    filter_by_cc();

    bool any_removed = check_triangles();
    if (any_removed)
        filter_by_cc();

    std::cout<<"Conflicted 3-cycle removal, # MST edges "<<mst_row_ids.size()<<", # Repulsive edges  "<<rep_row_ids.size()<<"\n";

    any_removed = check_quadrangles();
    if (any_removed)
        filter_by_cc();

    std::cout<<"Conflicted 4-cycle removal, # MST edges "<<mst_row_ids.size()<<", # Repulsive edges  "<<rep_row_ids.size()<<"\n";

    any_removed = check_pentagons();
    // if (any_removed) // would be done by next thrust based mapper.
    //     filter_by_cc();

    std::cout<<"Conflicted 5-cycle removal, # MST edges "<<mst_row_ids.size()<<", # Repulsive edges  "<<rep_row_ids.size()<<"\n";

    edge_contractions_woc_thrust c_mapper_full(num_nodes, std::move(mst_row_ids), std::move(mst_col_ids), std::move(mst_data), 
                                                std::move(rep_row_ids), std::move(rep_col_ids), std::move(cc_labels));
    return c_mapper_full.find_contraction_mapping();
}