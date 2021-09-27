#include "edge_contractions_woc_thrust.h"
#include <gpuMST.h>
#include "ECLgraph.h"
#include "rama_utils.h"

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

void frontier::restrict_to_indices(const thrust::device_vector<int>& indices_to_keep)
{
    int num_to_keep = indices_to_keep.size();

    thrust::device_vector<int> new_nodes(num_to_keep);
    thrust::device_vector<int> new_parent_nodes(num_to_keep);
    thrust::device_vector<int> new_rep_edges(num_to_keep);
    thrust::device_vector<int> new_bottleneck_indices(num_to_keep);
    thrust::device_vector<float> new_bottleneck_values(num_to_keep);
    
    auto first_input = thrust::make_zip_iterator(thrust::make_tuple(nodes.begin(), parent_nodes.begin(), rep_edges.begin(), bottleneck_indices.begin(), bottleneck_values.begin()));
    auto first_output = thrust::make_zip_iterator(thrust::make_tuple(new_nodes.begin(), new_parent_nodes.begin(), new_rep_edges.begin(), new_bottleneck_indices.begin(), new_bottleneck_values.begin()));

    thrust::gather(indices_to_keep.begin(), indices_to_keep.begin() + num_to_keep, first_input, first_output);
    thrust::swap(new_nodes, nodes);
    thrust::swap(new_parent_nodes, parent_nodes);
    thrust::swap(new_rep_edges, rep_edges);
    thrust::swap(new_bottleneck_indices, bottleneck_indices);
    thrust::swap(new_bottleneck_values, bottleneck_values);
}

void frontier::filter_by_rep_edges(const thrust::device_vector<int>& rep_edges_to_remove)
{
    assert(thrust::is_sorted(rep_edges_to_remove.begin(), rep_edges_to_remove.end()));

    thrust::device_vector<int> rep_edges_sorted = rep_edges;
    thrust::device_vector<int> rep_edges_sorting_order(rep_edges_sorted.size());
    thrust::sequence(rep_edges_sorting_order.begin(), rep_edges_sorting_order.end());
    thrust::sort_by_key(rep_edges_sorted.begin(), rep_edges_sorted.end(), rep_edges_sorting_order.begin());

    thrust::device_vector<int> indices_to_keep(nodes.size());
    auto last_to_keep = thrust::set_difference_by_key(rep_edges_sorted.begin(), rep_edges_sorted.end(), rep_edges_to_remove.begin(), rep_edges_to_remove.end(),
                                                    rep_edges_sorting_order.begin(), thrust::make_constant_iterator<int>(0),
                                                    thrust::make_discard_iterator(), indices_to_keep.begin());
    indices_to_keep.resize(std::distance(indices_to_keep.begin(), last_to_keep.second));
    restrict_to_indices(indices_to_keep);
}

void frontier::filter_by_mst_edges(const thrust::device_vector<int>& mst_edges_to_keep)
{
    assert(thrust::is_sorted(mst_edges_to_keep.begin(), mst_edges_to_keep.end()));

    thrust::device_vector<int> bottleneck_indices_sorted = bottleneck_indices;
    thrust::device_vector<int> bottleneck_indices_sorting_order(bottleneck_indices_sorted.size());
    thrust::sequence(bottleneck_indices_sorting_order.begin(), bottleneck_indices_sorting_order.end());
    thrust::sort_by_key(bottleneck_indices_sorted.begin(), bottleneck_indices_sorted.end(), bottleneck_indices_sorting_order.begin());

    thrust::device_vector<int> indices_to_keep(bottleneck_indices_sorted.size());
    auto last_to_keep = thrust::set_intersection_by_key(bottleneck_indices_sorted.begin(), bottleneck_indices_sorted.end(), 
                                                    mst_edges_to_keep.begin(), mst_edges_to_keep.end(),
                                                    bottleneck_indices_sorting_order.begin(),
                                                    thrust::make_discard_iterator(), indices_to_keep.begin());
    indices_to_keep.resize(std::distance(indices_to_keep.begin(), last_to_keep.second));
    restrict_to_indices(indices_to_keep);
}

void frontier::reassign_mst_indices(const thrust::device_vector<int>& valid_mst_indices, const int prev_mst_size)
{
    map_old_values_consec(bottleneck_indices, valid_mst_indices, prev_mst_size);
}

int edge_contractions_woc_thrust::filter_by_cc()
{
    assert(cc_labels.size() == num_nodes);
    thrust::device_vector<int> mst_row_offsets = compute_offsets(mst_row_ids, num_nodes - 1);
    computeCC_gpu(num_nodes, mst_row_ids.size(),
            thrust::raw_pointer_cast(mst_row_offsets.data()),
            thrust::raw_pointer_cast(mst_col_ids.data()),
            thrust::raw_pointer_cast(cc_labels.data()), 
            get_cuda_device());
    
    edge_in_diff_cc_func edge_in_diff_cc({thrust::raw_pointer_cast(cc_labels.data())});

    // thrust::device_vector<int> rep_invalid_indices(rep_row_ids.size());
    // thrust::sequence(rep_invalid_indices.begin(), rep_invalid_indices.end());
    // auto last_invalid = thrust::remove_if(rep_invalid_indices.begin(), rep_invalid_indices.end(), edge_in_same_cc);
    // const int num_invalid_rep = std::distance(rep_invalid_indices.begin(), last_invalid);
    // rep_invalid_indices.resize(num_invalid_rep);

    // row_frontier.filter_by_rep_edges(rep_invalid_indices);
    // col_frontier.filter_by_rep_edges(rep_invalid_indices);

    // return rep_row_ids.size() - num_invalid_rep;

    auto first_rep = thrust::make_zip_iterator(thrust::make_tuple(rep_row_ids.begin(), rep_col_ids.begin()));
    auto last_rep = thrust::make_zip_iterator(thrust::make_tuple(rep_row_ids.end(), rep_col_ids.end()));

    auto last_invalid = thrust::remove_if(first_rep, last_rep, edge_in_diff_cc);
    int num_rep_edges = std::distance(first_rep, last_invalid);

    rep_row_ids.resize(num_rep_edges);
    rep_col_ids.resize(num_rep_edges);

    // Re-initialize the frontiers. Another (probably faster) possibility is to only filter-out the found conflicted cycles and keep going, however
    // then we need to keep track of all predecessors which requires more memory. 
    row_frontier = frontier(rep_row_ids);
    col_frontier = frontier(rep_col_ids);

    return num_rep_edges;
}

struct recompute_degree_func
{
    __host__ __device__
        void operator()(thrust::tuple<const int, int&> t)
        {
            const int parent_node = thrust::get<0>(t);
            if (parent_node != -1)
            {
                int& degree = thrust::get<1>(t);
                degree--;
            }
        }
};

__global__ void expand_cuda(const int num_vertices,
    const int* const __restrict__ row_offsets,
    const int* const __restrict__ col_ids,
    const float* const __restrict__ costs,
    const int* const __restrict__ v_frontier,
    const int* const __restrict__ v_frontier_offsets,
    const int* const __restrict__ v_rep_edges,
    const int* const __restrict__ v_parent_nodes,
    const int* const __restrict__ v_bottleneck_edge_index,
    const float* const __restrict__ v_bottleneck_edge_value,
    int* __restrict__ expanded_frontier,
    int* __restrict__ expanded_rep_edges,
    int* __restrict__ expanded_parent_nodes,
    int* __restrict__ expanded_bottleneck_edge_index,
    float* __restrict__ expanded_bottleneck_edge_value)
{
    const int start_index = blockIdx.x * blockDim.x + threadIdx.x;
    const int num_threads = blockDim.x * gridDim.x;

    for (int idx = start_index; idx < num_vertices; idx += num_threads) 
    {
        const int src = v_frontier[idx];
        const int src_parent = v_parent_nodes[idx];
        const int src_rep_edge = v_rep_edges[idx];
        const int prev_bottleneck_index = v_bottleneck_edge_index[idx];
        const float prev_bottleneck_value = v_bottleneck_edge_value[idx];
        int output_offset = v_frontier_offsets[idx];
        for (int input_offset = row_offsets[src]; input_offset != row_offsets[src + 1]; ++input_offset)
        {
            const int dst = col_ids[input_offset];
            if (dst != src_parent)
            {
                expanded_frontier[output_offset] = dst;
                expanded_rep_edges[output_offset] = src_rep_edge;
                expanded_parent_nodes[output_offset] = src;
                const float cost = costs[input_offset];
                if (cost < prev_bottleneck_value)
                {
                    expanded_bottleneck_edge_index[output_offset] = input_offset;
                    expanded_bottleneck_edge_value[output_offset] = cost;
                }
                else
                {
                    expanded_bottleneck_edge_index[output_offset] = prev_bottleneck_index;
                    expanded_bottleneck_edge_value[output_offset] = prev_bottleneck_value;
                }
                ++output_offset;
            }
        }
    }
}

struct reduce_intersecting_paths
{
    __host__ __device__
        thrust::tuple<int, float, int>
        operator()(const thrust::tuple<int, float, int>& t1, 
                const thrust::tuple<int, float, int>& t2)
        {
            const float val1 = thrust::get<1>(t1);
            const float val2 = thrust::get<1>(t2);
            const int count = thrust::get<2>(t1) + thrust::get<2>(t2);
            if (val1 < val2)
                return thrust::make_tuple(thrust::get<0>(t1), val1, count);
            else
                return thrust::make_tuple(thrust::get<0>(t2), val2, count);
        }
};

struct single_occurence
{
    __host__ __device__
    bool operator()(const thrust::tuple<int, int, int>& t)
        {
            return thrust::get<2>(t) == 1;
        }
};

struct is_row_unique_frontier
{
    __host__ __device__
    bool operator()(const thrust::tuple<int, int, int, int, float, int, bool>& t)
        {
            return thrust::get<5>(t) == 1 && thrust::get<6>(t);
        }
};

struct is_col_unique_frontier
{
    __host__ __device__
    bool operator()(const thrust::tuple<int, int, int, int, float, int, bool>& t)
        {
            return thrust::get<5>(t) == 1 && !thrust::get<6>(t);
        }
};

// void remove_frontiers(thrust::device_vector<int>& v_frontier, thrust::device_vector<int>& v_rep_edges, 
//     thrust::device_vector<int>& v_parent_nodes, thrust::device_vector<int>& v_bottleneck_edge_index, thrust::device_vector<float>& v_bottleneck_edge_value,
//     const thrust::device_vector<int>& rep_edges_to_remove)
// {
//     assert(thrust::is_sorted(rep_edges_to_remove.begin(), rep_edges_to_remove.end()));
//     auto first_val_wo_rep = thrust::make_zip_iterator(thrust::make_tuple(v_frontier.begin(), v_parent_nodes.begin(), v_bottleneck_edge_index.begin(), v_bottleneck_edge_value.begin()));
//     thrust::sort_by_key(v_rep_edges.begin(), v_rep_edges.end(), first_val_wo_rep);

//     auto second_val_dummy = thrust::make_zip_iterator(thrust::make_tuple(thrust::make_constant_iterator<int>(0), thrust::make_constant_iterator<int>(0), thrust::make_constant_iterator<int>(0), thrust::make_constant_iterator<int>(0)));
    
//     thrust::device_vector<int> v_frontier_valid(v_frontier.size);
//     thrust::device_vector<int> v_rep_edges_valid(v_frontier.size);
//     thrust::device_vector<int> v_parent_nodes_valid(v_frontier.size);
//     thrust::device_vector<int> v_bottleneck_edge_index_valid(v_frontier.size);
//     thrust::device_vector<float> v_bottleneck_edge_value_valid(v_frontier.size);

//     auto first_val_valid = thrust::make_zip_iterator(thrust::make_tuple(v_frontier_valid.begin(), v_parent_nodes_valid.begin(), v_bottleneck_edge_index_valid.begin(), v_bottleneck_edge_value_valid.begin()));

//     auto last_valid = thrust::set_difference_by_key(v_rep_edges.begin(), v_rep_edges.end(), rep_edges_to_remove.begin(), rep_edges_to_remove.end(), 
//                                 first_val_wo_rep, second_val_dummy, v_rep_edges_valid.begin(), first_val_valid);

//     thrust::swap(v_frontier_valid, v_frontier);
//     thrust::swap(v_rep_edges_valid, v_rep_edges);
//     thrust::swap(v_parent_nodes_valid, v_parent_nodes);
//     thrust::swap(v_bottleneck_edge_index_valid, v_bottleneck_edge_index);
//     thrust::swap(v_bottleneck_edge_value_valid, v_bottleneck_edge_value);
// }

bool edge_contractions_woc_thrust::filter_cycles()
{
    // The elements in (v_frontier_row, v_rep_edges_row) which match (v_frontier_col, v_rep_edges_col) correspond to conflicted cycles
    // for these elements find the best bottleneck edge index by comparing  v_bottleneck_edge_index_row with v_bottleneck_edge_index_col based 
    // on corresponding values and remove that attractive edge from mst.
    thrust::device_vector<int>& v_frontier_row = row_frontier.get_nodes();
    thrust::device_vector<int>& v_parent_nodes_row = row_frontier.get_parent_nodes();
    thrust::device_vector<int>& v_rep_edges_row = row_frontier.get_rep_edges();
    thrust::device_vector<int>& v_bottleneck_edge_index_row = row_frontier.get_bottleneck_indices();
    thrust::device_vector<float>& v_bottleneck_edge_value_row = row_frontier.get_bottleneck_values();

    // Prepare by sorting both arrays:
    auto first_row_key = thrust::make_zip_iterator(thrust::make_tuple(v_frontier_row.begin(), v_rep_edges_row.begin()));
    auto last_row_key = thrust::make_zip_iterator(thrust::make_tuple(v_frontier_row.end(), v_rep_edges_row.end()));

    auto first_row_val = thrust::make_zip_iterator(thrust::make_tuple(v_parent_nodes_row.begin(), v_bottleneck_edge_index_row.begin(), v_bottleneck_edge_value_row.begin()));
    thrust::sort_by_key(first_row_key, last_row_key, first_row_val);
    
    thrust::device_vector<int>& v_frontier_col = col_frontier.get_nodes();
    thrust::device_vector<int>& v_parent_nodes_col = col_frontier.get_parent_nodes();
    thrust::device_vector<int>& v_rep_edges_col = col_frontier.get_rep_edges();
    thrust::device_vector<int>& v_bottleneck_edge_index_col = col_frontier.get_bottleneck_indices();
    thrust::device_vector<float>& v_bottleneck_edge_value_col = col_frontier.get_bottleneck_values();

    auto first_col_key = thrust::make_zip_iterator(thrust::make_tuple(v_frontier_col.begin(), v_rep_edges_col.begin()));
    auto last_col_key = thrust::make_zip_iterator(thrust::make_tuple(v_frontier_col.end(), v_rep_edges_col.end()));

    auto first_col_val = thrust::make_zip_iterator(thrust::make_tuple(v_parent_nodes_col.begin(), v_bottleneck_edge_index_col.begin(), v_bottleneck_edge_value_col.begin()));
    thrust::sort_by_key(first_col_key, last_col_key, first_col_val);

    // Merge and search for duplicates
    thrust::device_vector<int> v_frontier_merged(v_frontier_row.size() + v_frontier_col.size());
    thrust::device_vector<int> v_rep_edges_merged(v_frontier_row.size() + v_frontier_col.size());
    thrust::device_vector<int> v_bottleneck_index_merged(v_frontier_row.size() + v_frontier_col.size());
    thrust::device_vector<float> v_bottleneck_value_merged(v_frontier_row.size() + v_frontier_col.size());

    auto first_row_val_merge = thrust::make_zip_iterator(thrust::make_tuple(v_bottleneck_edge_index_row.begin(), v_bottleneck_edge_value_row.begin()));
    auto first_col_val_merge = thrust::make_zip_iterator(thrust::make_tuple(v_bottleneck_edge_index_col.begin(), v_bottleneck_edge_value_col.begin()));

    auto first_merged_key = thrust::make_zip_iterator(thrust::make_tuple(v_frontier_merged.begin(), v_rep_edges_merged.begin()));
    auto first_merged_val = thrust::make_zip_iterator(thrust::make_tuple(v_bottleneck_index_merged.begin(), v_bottleneck_value_merged.begin()));

    auto last_merged = thrust::merge_by_key(first_row_key, last_row_key, first_col_key, last_col_key,
                            first_row_val_merge, first_col_val_merge, first_merged_key, first_merged_val);
    
    assert(std::distance(first_merged_key, last_merged.first) == v_frontier_merged.size());

    auto first_merged_val_with_count = thrust::make_zip_iterator(thrust::make_tuple(v_bottleneck_index_merged.begin(), v_bottleneck_value_merged.begin(), thrust::make_constant_iterator<int>(1)));

    thrust::device_vector<int> v_rep_edges_reduced(v_frontier_row.size() + v_frontier_col.size());
    thrust::device_vector<int> v_bottleneck_index_reduced(v_frontier_row.size() + v_frontier_col.size());
    thrust::device_vector<int> num_occ(v_frontier_row.size() + v_frontier_col.size());

    auto reduced_key_first = thrust::make_zip_iterator(thrust::make_tuple(thrust::make_discard_iterator(), v_rep_edges_reduced.begin()));
    auto reduced_val_first = thrust::make_zip_iterator(thrust::make_tuple(v_bottleneck_index_reduced.begin(), thrust::make_discard_iterator(), num_occ.begin()));

    thrust::equal_to<thrust::tuple<int, int>> binary_pred_comp;
    auto last_reduce = thrust::reduce_by_key(first_merged_key, last_merged.first, first_merged_val_with_count, reduced_key_first, reduced_val_first, binary_pred_comp, reduce_intersecting_paths());
    int num_reduced = std::distance(reduced_key_first, last_reduce.first);
    v_rep_edges_reduced.resize(num_reduced);
    v_bottleneck_index_reduced.resize(num_reduced);
    num_occ.resize(num_reduced);

    // Find bottleneck edges and repulsive edges to remove.
    thrust::device_vector<int> mst_edges_to_remove = v_bottleneck_index_reduced;
    thrust::device_vector<int> rep_edges_to_remove = v_rep_edges_reduced;
    auto first_mst_remove = thrust::make_zip_iterator(thrust::make_tuple(mst_edges_to_remove.begin(), v_rep_edges_reduced.begin(), num_occ.begin()));
    auto last_mst_remove = thrust::make_zip_iterator(thrust::make_tuple(mst_edges_to_remove.end(), v_rep_edges_reduced.begin(), num_occ.end()));

    auto last_mst_remove_valid = thrust::remove_if(first_mst_remove, last_mst_remove, single_occurence());
    int num_directed_edges_to_remove = std::distance(first_mst_remove, last_mst_remove_valid);

    if (num_directed_edges_to_remove == 0)
        return false;

    mst_edges_to_remove.resize(num_directed_edges_to_remove);
    rep_edges_to_remove.resize(num_directed_edges_to_remove);

    thrust::sort(mst_edges_to_remove.begin(), mst_edges_to_remove.end());
    thrust::sort(rep_edges_to_remove.begin(), rep_edges_to_remove.end());

    auto last_mst_unique = thrust::unique(mst_edges_to_remove.begin(), mst_edges_to_remove.end());
    mst_edges_to_remove.resize(std::distance(mst_edges_to_remove.begin(), last_mst_unique));

    // Now remove the bottleneck edges (in both directions) from mst.

    // For this first find the edges which need to be removed and make them undirected.
    thrust::device_vector<int> mst_i_to_remove(mst_edges_to_remove.size());
    thrust::device_vector<int> mst_j_to_remove(mst_edges_to_remove.size());
    thrust::gather(mst_edges_to_remove.begin(), mst_edges_to_remove.end(), mst_row_ids.begin(), mst_i_to_remove.begin());
    thrust::gather(mst_edges_to_remove.begin(), mst_edges_to_remove.end(), mst_col_ids.begin(), mst_j_to_remove.begin());
    std::tie(mst_i_to_remove, mst_j_to_remove) = to_undirected(mst_i_to_remove, mst_j_to_remove);
    coo_sorting(mst_i_to_remove, mst_j_to_remove);

    auto first_mst = thrust::make_zip_iterator(thrust::make_tuple(mst_row_ids.begin(), mst_col_ids.begin()));
    auto last_mst = thrust::make_zip_iterator(thrust::make_tuple(mst_row_ids.end(), mst_col_ids.end()));
    
    auto first_mst_val = thrust::make_zip_iterator(thrust::make_tuple(mst_data.begin(), thrust::make_counting_iterator<int>(0)));
    auto val2_dummy = thrust::make_zip_iterator(thrust::make_tuple(thrust::make_constant_iterator<float>(0), thrust::make_counting_iterator<int>(0)));

    auto first_mst_to_remove = thrust::make_zip_iterator(thrust::make_tuple(mst_i_to_remove.begin(), mst_j_to_remove.begin()));
    auto last_mst_to_remove = thrust::make_zip_iterator(thrust::make_tuple(mst_i_to_remove.end(), mst_j_to_remove.end()));

    thrust::device_vector<int> mst_row_ids_valid(mst_row_ids.size());
    thrust::device_vector<int> mst_col_ids_valid(mst_col_ids.size());
    thrust::device_vector<float> mst_data_valid(mst_data.size());
    thrust::device_vector<int> mst_valid_indices(mst_row_ids.size());

    auto first_mst_valid_key = thrust::make_zip_iterator(thrust::make_tuple(mst_row_ids_valid.begin(), mst_col_ids_valid.begin()));
    auto first_mst_valid_val = thrust::make_zip_iterator(thrust::make_tuple(mst_data_valid.begin(), mst_valid_indices.begin()));

    auto last_to_keep = thrust::set_difference_by_key(first_mst, last_mst, first_mst_to_remove, last_mst_to_remove, 
                                                    first_mst_val, val2_dummy,
                                                    first_mst_valid_key, first_mst_valid_val);

    int num_valid_mst_edges = std::distance(first_mst_valid_key, last_to_keep.first);
    mst_row_ids_valid.resize(num_valid_mst_edges);
    mst_col_ids_valid.resize(num_valid_mst_edges);
    mst_data_valid.resize(num_valid_mst_edges);
    mst_valid_indices.resize(num_valid_mst_edges);

    // Since MST has changed, map old mst indices to new ones present in bottleneck_edge_index and remove invalid.
    // Since we re-initialize the frontiers anyway therefore no need to filter out. 
    // row_frontier.filter_by_mst_edges(mst_valid_indices);
    // col_frontier.filter_by_mst_edges(mst_valid_indices);
    // row_frontier.reassign_mst_indices(mst_valid_indices, mst_row_ids.size());
    // col_frontier.reassign_mst_indices(mst_valid_indices, mst_row_ids.size());

    thrust::swap(mst_row_ids_valid, mst_row_ids);
    thrust::swap(mst_col_ids_valid, mst_col_ids);
    thrust::swap(mst_data_valid, mst_data);

    return true;

    // Remove the intersected path from frontier (would be done by CC):
    // remove_frontiers(v_frontier_row, v_rep_edges_row, v_parent_nodes_row, v_bottleneck_edge_index_row, v_bottleneck_edge_value_row, rep_edges_to_remove);
    // remove_frontiers(v_frontier_col, v_rep_edges_col, v_parent_nodes_col, v_bottleneck_edge_index_col, v_bottleneck_edge_value_col, rep_edges_to_remove);
}

void edge_contractions_woc_thrust::expand_frontier(frontier& f)
{
    const thrust::device_vector<int>& v_frontier = f.get_nodes();
    const thrust::device_vector<int>& v_parent_nodes = f.get_parent_nodes();
    const thrust::device_vector<int>& v_rep_edges = f.get_rep_edges();
    const thrust::device_vector<int>& v_bottleneck_edge_index = f.get_bottleneck_indices();
    const thrust::device_vector<float>& v_bottleneck_edge_value = f.get_bottleneck_values();

    assert(v_frontier.size() == v_rep_edges.size());
    assert(v_frontier.size() == v_parent_nodes.size()); //parent node = -1 corresponds to seeds.
    assert(v_frontier.size() == v_bottleneck_edge_index.size());
    assert(v_frontier.size() == v_bottleneck_edge_value.size());

    const thrust::device_vector<int> mst_row_offsets = compute_offsets(mst_row_ids, num_nodes - 1);
    const thrust::device_vector<int> mst_node_degrees = offsets_to_degrees(mst_row_offsets);

    assert(mst_node_degrees.size() == num_nodes);
    thrust::device_vector<int> v_frontier_num_neighbours(v_frontier.size());
    thrust::gather(v_frontier.begin(), v_frontier.end(), mst_node_degrees.begin(), v_frontier_num_neighbours.begin());

    auto first = thrust::make_zip_iterator(thrust::make_tuple(v_parent_nodes.begin(), v_frontier_num_neighbours.begin()));
    auto last = thrust::make_zip_iterator(thrust::make_tuple(v_parent_nodes.end(), v_frontier_num_neighbours.end()));

    thrust::for_each(first, last, recompute_degree_func());

    thrust::device_vector<int> v_frontier_offsets = degrees_to_offsets(v_frontier_num_neighbours);

    const int num_expansions = v_frontier_offsets[v_frontier_offsets.size() - 1];
    thrust::device_vector<int> expanded_frontier(num_expansions);
    thrust::device_vector<int> expanded_rep_edges(num_expansions);
    thrust::device_vector<int> expanded_parent_nodes(num_expansions);
    thrust::device_vector<int> expanded_bottleneck_edge_index(num_expansions);
    thrust::device_vector<float> expanded_bottleneck_edge_value(num_expansions);

    const int threadCount = 256;
    const int blockCount = ceil(v_frontier.size() / (float) threadCount);
    expand_cuda<<<blockCount, threadCount>>>(v_frontier.size(),
                        thrust::raw_pointer_cast(mst_row_offsets.data()),
                        thrust::raw_pointer_cast(mst_col_ids.data()),
                        thrust::raw_pointer_cast(mst_data.data()),
                        thrust::raw_pointer_cast(v_frontier.data()),
                        thrust::raw_pointer_cast(v_frontier_offsets.data()),
                        thrust::raw_pointer_cast(v_rep_edges.data()),
                        thrust::raw_pointer_cast(v_parent_nodes.data()),
                        thrust::raw_pointer_cast(v_bottleneck_edge_index.data()),
                        thrust::raw_pointer_cast(v_bottleneck_edge_value.data()),
                        thrust::raw_pointer_cast(expanded_frontier.data()),
                        thrust::raw_pointer_cast(expanded_rep_edges.data()),
                        thrust::raw_pointer_cast(expanded_parent_nodes.data()),
                        thrust::raw_pointer_cast(expanded_bottleneck_edge_index.data()),
                        thrust::raw_pointer_cast(expanded_bottleneck_edge_value.data()));

    f = frontier(std::move(expanded_frontier), 
                std::move(expanded_parent_nodes), 
                std::move(expanded_rep_edges),
                std::move(expanded_bottleneck_edge_index), 
                std::move(expanded_bottleneck_edge_value));
}

edge_contractions_woc_thrust::edge_contractions_woc_thrust(const dCOO& A) : num_nodes(A.max_dim())
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
}

struct is_below_thresh_func
{
    const float min_thresh;
    __host__ __device__ bool operator()(const thrust::tuple<int,int,float>& t)
    {
        return thrust::get<2>(t) < min_thresh;
    }
};

bool edge_contractions_woc_thrust::filter_by_thresholding(const float& mean_multiplier = 0.9f)
{
    auto first = thrust::make_zip_iterator(thrust::make_tuple(thrust::constant_iterator<int>(1), mst_data.begin()));
    auto last = thrust::make_zip_iterator(thrust::make_tuple(thrust::constant_iterator<int>(1) + mst_data.size(), mst_data.end()));
    // compute average of positive edge costs:
    auto red = thrust::transform_reduce(first, last, pos_part(), thrust::make_tuple(0, 0.0f), tuple_sum());
    const float min_thresh = mean_multiplier * thrust::get<1>(red) / thrust::get<0>(red);

    // Remove all mst edges below min_thresh:
    auto first_mst = thrust::make_zip_iterator(thrust::make_tuple(mst_row_ids.begin(), mst_col_ids.begin(), mst_data.begin()));
    auto last_mst = thrust::make_zip_iterator(thrust::make_tuple(mst_row_ids.end(), mst_col_ids.end(), mst_data.end()));

    auto last_mst_valid = thrust::remove_if(first_mst, last_mst, is_below_thresh_func({min_thresh}));
    const int new_size = std::distance(first_mst, last_mst_valid);
    if(new_size == mst_row_ids.size())
        return false;
    mst_row_ids.resize(new_size);
    mst_col_ids.resize(new_size);
    mst_data.resize(new_size);
    return true;
}

std::tuple<thrust::device_vector<int>, int> edge_contractions_woc_thrust::find_contraction_mapping()
{
    MEASURE_CUMULATIVE_FUNCTION_EXECUTION_TIME;

    if (mst_row_ids.size() == 0)
        return {thrust::device_vector<int>(0), 0};

    std::cout<<"# MST edges "<<mst_row_ids.size()<<", # Repulsive edges "<<rep_row_ids.size()<<"\n";

    row_frontier = frontier(rep_row_ids);
    col_frontier = frontier(rep_col_ids);
    
    int num_rep_valid = filter_by_cc();

    int itr = 0;
    while(num_rep_valid > 0 && mst_row_ids.size() > 0)
    {
        if (itr % 5 == 0)
            if(filter_by_thresholding())
                num_rep_valid = filter_by_cc();

        if (num_rep_valid == 0 || mst_row_ids.size() == 0)
            break;
    
        // filter odd length conflicted cycles:
        std::cout<<"Conflicted cycle removal MST, Iteration: "<<itr<<", # MST edges "<<mst_row_ids.size()<<", # Repulsive edges  "<<num_rep_valid<<"\n";
        expand_frontier(row_frontier);
        bool any_removed = filter_cycles();
        if (any_removed)
            num_rep_valid = filter_by_cc();

        if (num_rep_valid == 0 || mst_row_ids.size() == 0)
            break;

        // filter even length conflicted cycles:
        std::cout<<"Conflicted cycle removal MST, Iteration: "<<itr<<", # MST edges "<<mst_row_ids.size()<<", # Repulsive edges  "<<num_rep_valid<<"\n";
        expand_frontier(col_frontier);
        any_removed = filter_cycles();
        if (any_removed)
            num_rep_valid = filter_by_cc();

        itr++;
    }

    thrust::device_vector<int> node_mapping = compress_label_sequence(cc_labels, cc_labels.size() - 1);
    int nr_ccs = *thrust::max_element(node_mapping.begin(), node_mapping.end()) + 1;
    std::cout<<"Found conflict-free contraction mapping with: "<<nr_ccs<<" connected components\n";

    assert(nr_ccs < num_nodes);

    return {node_mapping, mst_row_ids.size()};
}