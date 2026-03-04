#pragma once

#include <tuple>
#include <cassert>
#include <iostream>
#include <limits>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sort.h>
#include <thrust/merge.h>
#include <thrust/reduce.h>
#include <thrust/remove.h>
#include <thrust/copy.h>
#include <thrust/sequence.h>
#include <thrust/gather.h>
#include <thrust/scatter.h>
#include <thrust/partition.h>
#include <thrust/set_operations.h>
#include <thrust/for_each.h>
#include <thrust/transform.h>
#include <thrust/transform_reduce.h>
#include <thrust/scan.h>
#include <thrust/extrema.h>
#include <thrust/unique.h>
#include <thrust/count.h>
#include <thrust/swap.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/tuple.h>
#include <thrust/functional.h>

#include "graph.h"
#include "mst_boruvka.h"
#include "connected_components.h"
#include "rama_utils.h"
#include "time_measure_util.h"

#ifdef __CUDACC__
#define EC_HOST_DEVICE __host__ __device__
#else
#define EC_HOST_DEVICE
#endif

namespace ec_detail {

template<template<typename> class VectorType>
struct frontier {
    VectorType<int> nodes;
    VectorType<int> parent_nodes;
    VectorType<int> rep_edges;
    VectorType<int> bottleneck_indices;
    VectorType<float> bottleneck_values;

    frontier() {}

    frontier(const VectorType<int>& seeds)
        : nodes(seeds),
          parent_nodes(seeds.size(), -1),
          rep_edges(seeds.size()),
          bottleneck_indices(seeds.size(), 0),
          bottleneck_values(seeds.size(), std::numeric_limits<float>::max())
    {
        thrust::sequence(rep_edges.begin(), rep_edges.end());
    }

    frontier(VectorType<int>&& _nodes, VectorType<int>&& _parent_nodes,
             VectorType<int>&& _rep_edges, VectorType<int>&& _bottleneck_indices,
             VectorType<float>&& _bottleneck_values)
        : nodes(std::move(_nodes)),
          parent_nodes(std::move(_parent_nodes)),
          rep_edges(std::move(_rep_edges)),
          bottleneck_indices(std::move(_bottleneck_indices)),
          bottleneck_values(std::move(_bottleneck_values))
    {}

    size_t size() const { return nodes.size(); }
};

// Functors

struct reduce_intersecting_paths {
    EC_HOST_DEVICE
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

struct single_occurence {
    EC_HOST_DEVICE
    bool operator()(const thrust::tuple<int, int, int>& t)
    {
        return thrust::get<2>(t) == 1;
    }
};

struct is_below_thresh_func {
    const float min_thresh;
    EC_HOST_DEVICE bool operator()(const thrust::tuple<int, int, float>& t)
    {
        return thrust::get<2>(t) < min_thresh;
    }
};

struct is_reverse_edge {
    EC_HOST_DEVICE
    bool operator()(const thrust::tuple<int, int>& t)
    {
        return thrust::get<0>(t) >= thrust::get<1>(t);
    }
};

// BFS expansion of frontier along MST adjacency
template<template<typename> class VectorType>
void expand_frontier(
    frontier<VectorType>& f,
    const VectorType<int>& mst_row_ids,
    const VectorType<int>& mst_col_ids,
    const VectorType<float>& mst_data,
    const int num_nodes)
{
    if (f.size() == 0)
        return;

    const VectorType<int> mst_row_offsets = compute_offsets<VectorType>(mst_row_ids, num_nodes - 1);
    const VectorType<int> mst_node_degrees = offsets_to_degrees<VectorType>(mst_row_offsets);
    assert(mst_node_degrees.size() == (size_t)num_nodes);

    // Get degree for each frontier node
    VectorType<int> v_frontier_num_neighbours(f.size());
    thrust::gather(f.nodes.begin(), f.nodes.end(), mst_node_degrees.begin(), v_frontier_num_neighbours.begin());

    // Subtract 1 for parent (except for seeds where parent == -1)
    {
        const int* parent_ptr = thrust::raw_pointer_cast(f.parent_nodes.data());
        int* degree_ptr = thrust::raw_pointer_cast(v_frontier_num_neighbours.data());
        const int n = f.size();
        thrust::for_each(
            thrust::make_counting_iterator(0),
            thrust::make_counting_iterator(n),
            [parent_ptr, degree_ptr] EC_HOST_DEVICE (int idx) {
                if (parent_ptr[idx] != -1)
                    degree_ptr[idx]--;
            });
    }

    VectorType<int> v_frontier_offsets = degrees_to_offsets<VectorType>(v_frontier_num_neighbours);
    const int num_expansions = v_frontier_offsets[v_frontier_offsets.size() - 1];

    VectorType<int> expanded_frontier(num_expansions);
    VectorType<int> expanded_rep_edges(num_expansions);
    VectorType<int> expanded_parent_nodes(num_expansions);
    VectorType<int> expanded_bottleneck_edge_index(num_expansions);
    VectorType<float> expanded_bottleneck_edge_value(num_expansions);

    // Fill expanded arrays using for_each over frontier indices
    {
        const int* row_offsets_ptr = thrust::raw_pointer_cast(mst_row_offsets.data());
        const int* col_ids_ptr = thrust::raw_pointer_cast(mst_col_ids.data());
        const float* costs_ptr = thrust::raw_pointer_cast(mst_data.data());
        const int* frontier_ptr = thrust::raw_pointer_cast(f.nodes.data());
        const int* offsets_ptr = thrust::raw_pointer_cast(v_frontier_offsets.data());
        const int* rep_edges_ptr = thrust::raw_pointer_cast(f.rep_edges.data());
        const int* parent_ptr = thrust::raw_pointer_cast(f.parent_nodes.data());
        const int* bn_index_ptr = thrust::raw_pointer_cast(f.bottleneck_indices.data());
        const float* bn_value_ptr = thrust::raw_pointer_cast(f.bottleneck_values.data());
        int* exp_frontier_ptr = thrust::raw_pointer_cast(expanded_frontier.data());
        int* exp_rep_ptr = thrust::raw_pointer_cast(expanded_rep_edges.data());
        int* exp_parent_ptr = thrust::raw_pointer_cast(expanded_parent_nodes.data());
        int* exp_bn_index_ptr = thrust::raw_pointer_cast(expanded_bottleneck_edge_index.data());
        float* exp_bn_value_ptr = thrust::raw_pointer_cast(expanded_bottleneck_edge_value.data());

        const int n = f.size();
        thrust::for_each(
            thrust::make_counting_iterator(0),
            thrust::make_counting_iterator(n),
            [row_offsets_ptr, col_ids_ptr, costs_ptr,
             frontier_ptr, offsets_ptr, rep_edges_ptr, parent_ptr,
             bn_index_ptr, bn_value_ptr,
             exp_frontier_ptr, exp_rep_ptr, exp_parent_ptr,
             exp_bn_index_ptr, exp_bn_value_ptr] EC_HOST_DEVICE (int idx) {
                const int src = frontier_ptr[idx];
                const int src_parent = parent_ptr[idx];
                const int src_rep_edge = rep_edges_ptr[idx];
                const int prev_bn_index = bn_index_ptr[idx];
                const float prev_bn_value = bn_value_ptr[idx];
                int output_offset = offsets_ptr[idx];
                for (int input_offset = row_offsets_ptr[src];
                     input_offset != row_offsets_ptr[src + 1]; ++input_offset)
                {
                    const int dst = col_ids_ptr[input_offset];
                    if (dst != src_parent)
                    {
                        exp_frontier_ptr[output_offset] = dst;
                        exp_rep_ptr[output_offset] = src_rep_edge;
                        exp_parent_ptr[output_offset] = src;
                        const float cost = costs_ptr[input_offset];
                        if (cost < prev_bn_value)
                        {
                            exp_bn_index_ptr[output_offset] = input_offset;
                            exp_bn_value_ptr[output_offset] = cost;
                        }
                        else
                        {
                            exp_bn_index_ptr[output_offset] = prev_bn_index;
                            exp_bn_value_ptr[output_offset] = prev_bn_value;
                        }
                        ++output_offset;
                    }
                }
            });
    }

    f = frontier<VectorType>(
        std::move(expanded_frontier),
        std::move(expanded_parent_nodes),
        std::move(expanded_rep_edges),
        std::move(expanded_bottleneck_edge_index),
        std::move(expanded_bottleneck_edge_value));
}

// Find intersecting BFS paths and remove bottleneck MST edges
template<template<typename> class VectorType>
bool filter_cycles(
    frontier<VectorType>& row_frontier,
    frontier<VectorType>& col_frontier,
    VectorType<int>& mst_row_ids,
    VectorType<int>& mst_col_ids,
    VectorType<float>& mst_data)
{
    // Sort both frontiers by (node, rep_edge)
    auto first_row_key = thrust::make_zip_iterator(thrust::make_tuple(
        row_frontier.nodes.begin(), row_frontier.rep_edges.begin()));
    auto last_row_key = thrust::make_zip_iterator(thrust::make_tuple(
        row_frontier.nodes.end(), row_frontier.rep_edges.end()));
    auto first_row_val = thrust::make_zip_iterator(thrust::make_tuple(
        row_frontier.parent_nodes.begin(),
        row_frontier.bottleneck_indices.begin(),
        row_frontier.bottleneck_values.begin()));
    thrust::sort_by_key(first_row_key, last_row_key, first_row_val);

    auto first_col_key = thrust::make_zip_iterator(thrust::make_tuple(
        col_frontier.nodes.begin(), col_frontier.rep_edges.begin()));
    auto last_col_key = thrust::make_zip_iterator(thrust::make_tuple(
        col_frontier.nodes.end(), col_frontier.rep_edges.end()));
    auto first_col_val = thrust::make_zip_iterator(thrust::make_tuple(
        col_frontier.parent_nodes.begin(),
        col_frontier.bottleneck_indices.begin(),
        col_frontier.bottleneck_values.begin()));
    thrust::sort_by_key(first_col_key, last_col_key, first_col_val);

    // Merge and search for duplicates
    const size_t total_size = row_frontier.size() + col_frontier.size();
    VectorType<int> v_frontier_merged(total_size);
    VectorType<int> v_rep_edges_merged(total_size);
    VectorType<int> v_bottleneck_index_merged(total_size);
    VectorType<float> v_bottleneck_value_merged(total_size);

    auto first_row_val_merge = thrust::make_zip_iterator(thrust::make_tuple(
        row_frontier.bottleneck_indices.begin(), row_frontier.bottleneck_values.begin()));
    auto first_col_val_merge = thrust::make_zip_iterator(thrust::make_tuple(
        col_frontier.bottleneck_indices.begin(), col_frontier.bottleneck_values.begin()));

    auto first_merged_key = thrust::make_zip_iterator(thrust::make_tuple(
        v_frontier_merged.begin(), v_rep_edges_merged.begin()));
    auto first_merged_val = thrust::make_zip_iterator(thrust::make_tuple(
        v_bottleneck_index_merged.begin(), v_bottleneck_value_merged.begin()));

    auto last_merged = thrust::merge_by_key(
        first_row_key, last_row_key, first_col_key, last_col_key,
        first_row_val_merge, first_col_val_merge, first_merged_key, first_merged_val);

    assert(std::distance(first_merged_key, last_merged.first) == (int)total_size);

    auto first_merged_val_with_count = thrust::make_zip_iterator(thrust::make_tuple(
        v_bottleneck_index_merged.begin(), v_bottleneck_value_merged.begin(),
        thrust::make_constant_iterator<int>(1)));

    VectorType<int> v_rep_edges_reduced(total_size);
    VectorType<int> v_bottleneck_index_reduced(total_size);
    VectorType<int> num_occ(total_size);

    auto reduced_key_first = thrust::make_zip_iterator(thrust::make_tuple(
        thrust::make_discard_iterator(), v_rep_edges_reduced.begin()));
    auto reduced_val_first = thrust::make_zip_iterator(thrust::make_tuple(
        v_bottleneck_index_reduced.begin(), thrust::make_discard_iterator(), num_occ.begin()));

    thrust::equal_to<thrust::tuple<int, int>> binary_pred_comp;
    auto last_reduce = thrust::reduce_by_key(
        first_merged_key, last_merged.first, first_merged_val_with_count,
        reduced_key_first, reduced_val_first, binary_pred_comp, reduce_intersecting_paths());
    int num_reduced = std::distance(reduced_key_first, last_reduce.first);
    v_rep_edges_reduced.resize(num_reduced);
    v_bottleneck_index_reduced.resize(num_reduced);
    num_occ.resize(num_reduced);

    // Find bottleneck edges and repulsive edges to remove
    VectorType<int> mst_edges_to_remove = v_bottleneck_index_reduced;
    auto first_mst_remove = thrust::make_zip_iterator(thrust::make_tuple(
        mst_edges_to_remove.begin(), v_rep_edges_reduced.begin(), num_occ.begin()));
    auto last_mst_remove = thrust::make_zip_iterator(thrust::make_tuple(
        mst_edges_to_remove.end(), v_rep_edges_reduced.end(), num_occ.end()));

    auto last_mst_remove_valid = thrust::remove_if(first_mst_remove, last_mst_remove, single_occurence());
    int num_directed_edges_to_remove = std::distance(first_mst_remove, last_mst_remove_valid);

    if (num_directed_edges_to_remove == 0)
        return false;

    mst_edges_to_remove.resize(num_directed_edges_to_remove);

    thrust::sort(mst_edges_to_remove.begin(), mst_edges_to_remove.end());
    auto last_mst_unique = thrust::unique(mst_edges_to_remove.begin(), mst_edges_to_remove.end());
    mst_edges_to_remove.resize(std::distance(mst_edges_to_remove.begin(), last_mst_unique));

    // Remove bottleneck edges (in both directions) from MST
    VectorType<int> mst_i_to_remove(mst_edges_to_remove.size());
    VectorType<int> mst_j_to_remove(mst_edges_to_remove.size());
    thrust::gather(mst_edges_to_remove.begin(), mst_edges_to_remove.end(),
                   mst_row_ids.begin(), mst_i_to_remove.begin());
    thrust::gather(mst_edges_to_remove.begin(), mst_edges_to_remove.end(),
                   mst_col_ids.begin(), mst_j_to_remove.begin());
    std::tie(mst_i_to_remove, mst_j_to_remove) = to_undirected<VectorType>(mst_i_to_remove, mst_j_to_remove);
    coo_sorting<VectorType>(mst_i_to_remove, mst_j_to_remove);

    auto first_mst = thrust::make_zip_iterator(thrust::make_tuple(mst_row_ids.begin(), mst_col_ids.begin()));
    auto last_mst = thrust::make_zip_iterator(thrust::make_tuple(mst_row_ids.end(), mst_col_ids.end()));

    auto first_mst_val = thrust::make_zip_iterator(thrust::make_tuple(
        mst_data.begin(), thrust::make_counting_iterator<int>(0)));
    auto val2_dummy = thrust::make_zip_iterator(thrust::make_tuple(
        thrust::make_constant_iterator<float>(0), thrust::make_counting_iterator<int>(0)));

    auto first_mst_to_remove = thrust::make_zip_iterator(thrust::make_tuple(
        mst_i_to_remove.begin(), mst_j_to_remove.begin()));
    auto last_mst_to_remove = thrust::make_zip_iterator(thrust::make_tuple(
        mst_i_to_remove.end(), mst_j_to_remove.end()));

    VectorType<int> mst_row_ids_valid(mst_row_ids.size());
    VectorType<int> mst_col_ids_valid(mst_col_ids.size());
    VectorType<float> mst_data_valid(mst_data.size());
    VectorType<int> mst_valid_indices(mst_row_ids.size());

    auto first_mst_valid_key = thrust::make_zip_iterator(thrust::make_tuple(
        mst_row_ids_valid.begin(), mst_col_ids_valid.begin()));
    auto first_mst_valid_val = thrust::make_zip_iterator(thrust::make_tuple(
        mst_data_valid.begin(), mst_valid_indices.begin()));

    auto last_to_keep = thrust::set_difference_by_key(
        first_mst, last_mst, first_mst_to_remove, last_mst_to_remove,
        first_mst_val, val2_dummy, first_mst_valid_key, first_mst_valid_val);

    int num_valid_mst_edges = std::distance(first_mst_valid_key, last_to_keep.first);
    mst_row_ids_valid.resize(num_valid_mst_edges);
    mst_col_ids_valid.resize(num_valid_mst_edges);
    mst_data_valid.resize(num_valid_mst_edges);

    thrust::swap(mst_row_ids_valid, mst_row_ids);
    thrust::swap(mst_col_ids_valid, mst_col_ids);
    thrust::swap(mst_data_valid, mst_data);

    return true;
}

// Compute connected components and remove repulsive edges crossing different CCs
template<template<typename> class VectorType>
int filter_by_cc(
    VectorType<int>& cc_labels,
    const VectorType<int>& mst_row_ids,
    const VectorType<int>& mst_col_ids,
    VectorType<int>& rep_row_ids,
    VectorType<int>& rep_col_ids,
    frontier<VectorType>& row_frontier,
    frontier<VectorType>& col_frontier,
    const int num_nodes)
{
    cc_labels = connected_components::compute_cc<VectorType>(num_nodes, mst_row_ids, mst_col_ids);

    const int* cc_ptr = thrust::raw_pointer_cast(cc_labels.data());

    auto first_rep = thrust::make_zip_iterator(thrust::make_tuple(rep_row_ids.begin(), rep_col_ids.begin()));
    auto last_rep = thrust::make_zip_iterator(thrust::make_tuple(rep_row_ids.end(), rep_col_ids.end()));

    auto last_valid = thrust::remove_if(first_rep, last_rep,
        [cc_ptr] EC_HOST_DEVICE (const thrust::tuple<int, int>& t) {
            return cc_ptr[thrust::get<0>(t)] != cc_ptr[thrust::get<1>(t)];
        });
    int num_rep_edges = std::distance(first_rep, last_valid);
    rep_row_ids.resize(num_rep_edges);
    rep_col_ids.resize(num_rep_edges);

    // Re-initialize frontiers
    row_frontier = frontier<VectorType>(rep_row_ids);
    col_frontier = frontier<VectorType>(rep_col_ids);

    return num_rep_edges;
}

// Remove MST edges below average cost threshold
template<template<typename> class VectorType>
bool filter_by_thresholding(
    VectorType<int>& mst_row_ids,
    VectorType<int>& mst_col_ids,
    VectorType<float>& mst_data,
    const float mean_multiplier = 0.9f)
{
    auto first = thrust::make_zip_iterator(thrust::make_tuple(
        thrust::constant_iterator<int>(1), mst_data.begin()));
    auto last = thrust::make_zip_iterator(thrust::make_tuple(
        thrust::constant_iterator<int>(1) + mst_data.size(), mst_data.end()));
    auto red = thrust::transform_reduce(first, last, pos_part(), thrust::make_tuple(0, 0.0f), tuple_sum());
    const float min_thresh = mean_multiplier * thrust::get<1>(red) / thrust::get<0>(red);

    auto first_mst = thrust::make_zip_iterator(thrust::make_tuple(
        mst_row_ids.begin(), mst_col_ids.begin(), mst_data.begin()));
    auto last_mst = thrust::make_zip_iterator(thrust::make_tuple(
        mst_row_ids.end(), mst_col_ids.end(), mst_data.end()));

    auto last_mst_valid = thrust::remove_if(first_mst, last_mst, is_below_thresh_func({min_thresh}));
    const int new_size = std::distance(first_mst, last_mst_valid);
    if (new_size == (int)mst_row_ids.size())
        return false;
    mst_row_ids.resize(new_size);
    mst_col_ids.resize(new_size);
    mst_data.resize(new_size);
    return true;
}

} // namespace ec_detail

// Find contraction mapping using BFS frontier approach on MST.
// Returns (node_mapping, num_remaining_mst_edges).
// lifted_G: if provided, negative lifted edges are also treated as repulsive seeds.
template<template<typename> class VectorType>
std::tuple<VectorType<int>, int> find_contraction_mapping(
    const Graph<VectorType>& G, const Graph<VectorType>& lifted_G, bool verbose = true)
{
    MEASURE_CUMULATIVE_FUNCTION_EXECUTION_TIME;

    const int num_nodes = G.num_nodes();

    // 1. Partition symmetric edges into positive and repulsive
    const VectorType<int>& all_tails = G.get_tails();
    const VectorType<int>& all_heads = G.get_heads();
    const VectorType<float>& all_costs = G.get_costs();

    const size_t total = all_costs.size();
    VectorType<int> pos_tails(total), pos_heads(total);
    VectorType<float> pos_costs(total);
    VectorType<int> rep_tails(total), rep_heads(total);

    auto first_all = thrust::make_zip_iterator(thrust::make_tuple(
        all_tails.begin(), all_heads.begin(), all_costs.begin()));
    auto last_all = thrust::make_zip_iterator(thrust::make_tuple(
        all_tails.end(), all_heads.end(), all_costs.end()));
    auto first_pos = thrust::make_zip_iterator(thrust::make_tuple(
        pos_tails.begin(), pos_heads.begin(), pos_costs.begin()));
    auto first_rep = thrust::make_zip_iterator(thrust::make_tuple(
        rep_tails.begin(), rep_heads.begin(), thrust::make_discard_iterator()));

    auto ends = thrust::partition_copy(first_all, last_all, first_pos, first_rep,
        [] EC_HOST_DEVICE (const thrust::tuple<int,int,float>& t) {
            return thrust::get<2>(t) >= 0.0f;
        });

    const int num_positive = std::distance(first_pos, ends.first);
    if (num_positive == 0)
        return {VectorType<int>(0), 0};

    pos_tails.resize(num_positive);
    pos_heads.resize(num_positive);
    pos_costs.resize(num_positive);

    const int num_repulsive = std::distance(first_rep, ends.second);
    rep_tails.resize(num_repulsive);
    rep_heads.resize(num_repulsive);

    // Extract single-direction repulsive edges (tail < head) for frontier seeds
    {
        auto first_rep_zip = thrust::make_zip_iterator(thrust::make_tuple(
            rep_tails.begin(), rep_heads.begin()));
        auto last_rep_zip = thrust::make_zip_iterator(thrust::make_tuple(
            rep_tails.end(), rep_heads.end()));
        auto last_valid = thrust::remove_if(first_rep_zip, last_rep_zip, ec_detail::is_reverse_edge());
        int num_single = std::distance(first_rep_zip, last_valid);
        rep_tails.resize(num_single);
        rep_heads.resize(num_single);
    }

    // Append negative lifted edges (single-direction, tail < head) as additional repulsive seeds
    if (lifted_G.num_directed_edges() > 0)
    {
        const VectorType<int>& l_tails = lifted_G.get_tails();
        const VectorType<int>& l_heads = lifted_G.get_heads();
        const VectorType<float>& l_costs = lifted_G.get_costs();

        // Count negative, single-direction lifted edges
        const size_t l_total = l_costs.size();
        VectorType<int> neg_l_tails(l_total), neg_l_heads(l_total);

        auto first_l = thrust::make_zip_iterator(thrust::make_tuple(
            l_tails.begin(), l_heads.begin(), l_costs.begin()));
        auto last_l = thrust::make_zip_iterator(thrust::make_tuple(
            l_tails.end(), l_heads.end(), l_costs.end()));
        auto first_neg_l = thrust::make_zip_iterator(thrust::make_tuple(
            neg_l_tails.begin(), neg_l_heads.begin(), thrust::make_discard_iterator()));

        auto last_neg_l = thrust::copy_if(first_l, last_l, first_neg_l,
            [] EC_HOST_DEVICE (const thrust::tuple<int,int,float>& t) {
                return thrust::get<2>(t) < 0.0f && thrust::get<0>(t) < thrust::get<1>(t);
            });
        int num_neg_lifted = std::distance(first_neg_l, last_neg_l);
        neg_l_tails.resize(num_neg_lifted);
        neg_l_heads.resize(num_neg_lifted);

        if (num_neg_lifted > 0)
        {
            const size_t old_size = rep_tails.size();
            rep_tails.resize(old_size + num_neg_lifted);
            rep_heads.resize(old_size + num_neg_lifted);
            thrust::copy(neg_l_tails.begin(), neg_l_tails.end(), rep_tails.begin() + old_size);
            thrust::copy(neg_l_heads.begin(), neg_l_heads.end(), rep_heads.begin() + old_size);
        }
    }

    // 2. Compute maximum spanning tree (takes symmetric input, returns single-direction)
    VectorType<int> mst_tails, mst_heads;
    VectorType<float> mst_costs;
    std::tie(mst_tails, mst_heads, mst_costs) =
        MST_boruvka::maximum_spanning_tree<VectorType>(pos_tails, pos_heads, pos_costs);

    if (mst_tails.size() == 0)
        return {VectorType<int>(0), 0};

    // 3. Symmetrize and sort MST for BFS adjacency
    std::tie(mst_tails, mst_heads, mst_costs) = to_undirected<VectorType>(mst_tails, mst_heads, mst_costs);
    coo_sorting<VectorType>(mst_tails, mst_heads, mst_costs);

    if (verbose)
        std::cout << "# MST edges " << mst_tails.size() << ", # Repulsive edges " << rep_tails.size() << "\n";

    // 4. Initialize frontiers from repulsive edge endpoints
    ec_detail::frontier<VectorType> row_frontier(rep_tails);
    ec_detail::frontier<VectorType> col_frontier(rep_heads);

    // 5. Initial CC filtering
    VectorType<int> cc_labels(num_nodes);
    int num_rep_valid = ec_detail::filter_by_cc<VectorType>(
        cc_labels, mst_tails, mst_heads, rep_tails, rep_heads,
        row_frontier, col_frontier, num_nodes);

    // 6. BFS loop
    int itr = 0;
    while (num_rep_valid > 0 && mst_tails.size() > 0)
    {
        if (itr % 5 == 0)
            if (ec_detail::filter_by_thresholding<VectorType>(mst_tails, mst_heads, mst_costs))
                num_rep_valid = ec_detail::filter_by_cc<VectorType>(
                    cc_labels, mst_tails, mst_heads, rep_tails, rep_heads,
                    row_frontier, col_frontier, num_nodes);

        if (num_rep_valid == 0 || mst_tails.size() == 0)
            break;

        if (verbose)
            std::cout << "Conflicted cycle removal MST, Iteration: " << itr
                      << ", # MST edges " << mst_tails.size()
                      << ", # Repulsive edges  " << num_rep_valid << "\n";
        ec_detail::expand_frontier<VectorType>(row_frontier, mst_tails, mst_heads, mst_costs, num_nodes);
        bool any_removed = ec_detail::filter_cycles<VectorType>(
            row_frontier, col_frontier, mst_tails, mst_heads, mst_costs);
        if (any_removed)
            num_rep_valid = ec_detail::filter_by_cc<VectorType>(
                cc_labels, mst_tails, mst_heads, rep_tails, rep_heads,
                row_frontier, col_frontier, num_nodes);

        if (num_rep_valid == 0 || mst_tails.size() == 0)
            break;

        if (verbose)
            std::cout << "Conflicted cycle removal MST, Iteration: " << itr
                      << ", # MST edges " << mst_tails.size()
                      << ", # Repulsive edges  " << num_rep_valid << "\n";
        ec_detail::expand_frontier<VectorType>(col_frontier, mst_tails, mst_heads, mst_costs, num_nodes);
        any_removed = ec_detail::filter_cycles<VectorType>(
            row_frontier, col_frontier, mst_tails, mst_heads, mst_costs);
        if (any_removed)
            num_rep_valid = ec_detail::filter_by_cc<VectorType>(
                cc_labels, mst_tails, mst_heads, rep_tails, rep_heads,
                row_frontier, col_frontier, num_nodes);

        itr++;
    }

    // 7. Return contraction mapping
    VectorType<int> node_mapping = compress_label_sequence<VectorType>(cc_labels, cc_labels.size() - 1);
    int nr_ccs = *thrust::max_element(node_mapping.begin(), node_mapping.end()) + 1;
    if (verbose)
        std::cout << "Found conflict-free contraction mapping with: " << nr_ccs << " connected components\n";

    assert(nr_ccs <= num_nodes);

    return {node_mapping, (int)mst_tails.size()};
}

// Backward-compatible single-argument overload (no lifted graph).
template<template<typename> class VectorType>
std::tuple<VectorType<int>, int> find_contraction_mapping(
    const Graph<VectorType>& G, bool verbose = true)
{
    return find_contraction_mapping<VectorType>(G, Graph<VectorType>(), verbose);
}

// Explicit instantiation declarations.
extern template
std::tuple<thrust::host_vector<int>, int>
find_contraction_mapping<thrust::host_vector>(const Graph<thrust::host_vector>&, const Graph<thrust::host_vector>&, bool);

extern template
std::tuple<thrust::device_vector<int>, int>
find_contraction_mapping<thrust::device_vector>(const Graph<thrust::device_vector>&, const Graph<thrust::device_vector>&, bool);
