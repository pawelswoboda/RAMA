#pragma once

#include <cassert>
#include <iostream>
#include <tuple>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/for_each.h>
#include <thrust/transform_reduce.h>
#include <thrust/reduce.h>
#include <thrust/sequence.h>
#include <thrust/fill.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/tuple.h>
#include "graph.h"

#ifdef __CUDACC__
#define MM_HOST_DEVICE __host__ __device__
#else
#define MM_HOST_DEVICE
#endif

namespace mm_detail {

// Compute minimum edge cost for matching: mean_multiplier_mm * mean(positive edge costs).
// Returns -1 if no positive edges exist.
template<template<typename> class VectorType>
inline float determine_matching_threshold(const Graph<VectorType>& A, const float mean_multiplier_mm)
{
    const auto& costs = A.get_costs();
    auto first = thrust::make_zip_iterator(thrust::make_tuple(
        thrust::constant_iterator<int>(1), costs.begin()));
    auto last = thrust::make_zip_iterator(thrust::make_tuple(
        thrust::constant_iterator<int>(1) + costs.size(), costs.end()));

    auto pos_part = [] MM_HOST_DEVICE (const thrust::tuple<int, float>& t) {
        if (thrust::get<1>(t) >= 0.0f)
            return t;
        return thrust::make_tuple(0, 0.0f);
    };

    auto tuple_sum = [] MM_HOST_DEVICE (const thrust::tuple<int, float>& t1,
                                        const thrust::tuple<int, float>& t2) {
        return thrust::make_tuple(
            thrust::get<0>(t1) + thrust::get<0>(t2),
            thrust::get<1>(t1) + thrust::get<1>(t2));
    };

    auto red = thrust::transform_reduce(first, last,
        pos_part, thrust::make_tuple(0, 0.0f), tuple_sum);
    if (thrust::get<0>(red) == 0)
        return -1.0f;
    return mean_multiplier_mm * thrust::get<1>(red) / thrust::get<0>(red);
}

} // namespace mm_detail

// Greedy maximum matching via iterative mutual best-neighbour pairing.
// Each round, every unmatched vertex picks its strongest unmatched neighbour
// above threshold; mutual pairs are matched. Repeats until convergence.
// Returns (node_mapping, nr_matched_vertices) where node_mapping[v] = u
// means v is matched to u (with u <= v), and unmatched vertices map to themselves.
template<template<typename> class VectorType>
std::tuple<VectorType<int>, int> filter_edges_by_matching(
    const Graph<VectorType>& A,
    const float mean_multiplier_mm = 0.0,
    const bool verbose = true)
{
    const int num_nodes = A.num_nodes();

    VectorType<int> node_mapping(num_nodes);
    thrust::sequence(node_mapping.begin(), node_mapping.end(), 0);

    const float min_edge_weight_to_match = mm_detail::determine_matching_threshold(A, mean_multiplier_mm);
    if (verbose)
        std::cout << "min_edge_weight_to_match: " << min_edge_weight_to_match << "\n";
    if (min_edge_weight_to_match < 0)
        return {node_mapping, 0};

    VectorType<int> v_matched(num_nodes, 0);
    VectorType<int> v_best_neighbours(num_nodes, -1);
    VectorType<int> still_running(1);

    const VectorType<int> node_offsets = A.compute_node_offsets();
    const int* offsets_ptr = thrust::raw_pointer_cast(node_offsets.data());
    const int* heads_ptr = A.get_heads_ptr();
    const float* costs_ptr = A.get_costs_ptr();

    int prev_num_matched = 0;
    for (int t = 0; t < 10; t++)
    {
        thrust::fill(still_running.begin(), still_running.end(), 0);

        // Phase 1: pick best neighbour for each unmatched vertex
        {
            thrust::fill(v_best_neighbours.begin(), v_best_neighbours.end(), -1);
            const int* matched_ptr = thrust::raw_pointer_cast(v_matched.data());
            int* best_ptr = thrust::raw_pointer_cast(v_best_neighbours.data());
            const float thresh = min_edge_weight_to_match;

            thrust::for_each(
                thrust::make_counting_iterator(0),
                thrust::make_counting_iterator(num_nodes),
                [offsets_ptr, heads_ptr, costs_ptr, matched_ptr, best_ptr, thresh]
                MM_HOST_DEVICE (const int v) {
                    if (matched_ptr[v])
                        return;
                    int best_head = -1;
                    float best_cost = thresh;
                    for (int e = offsets_ptr[v]; e < offsets_ptr[v + 1]; ++e) {
                        const int h = heads_ptr[e];
                        const float c = costs_ptr[e];
                        if (h == v || matched_ptr[h] || c < best_cost)
                            continue;
                        best_cost = c;
                        best_head = h;
                    }
                    best_ptr[v] = best_head;
                });
        }

        // Phase 2: match mutual best neighbours
        {
            int* matched_ptr = thrust::raw_pointer_cast(v_matched.data());
            int* mapping_ptr = thrust::raw_pointer_cast(node_mapping.data());
            const int* best_ptr = thrust::raw_pointer_cast(v_best_neighbours.data());
            int* still_running_ptr = thrust::raw_pointer_cast(still_running.data());

            thrust::for_each(
                thrust::make_counting_iterator(0),
                thrust::make_counting_iterator(num_nodes),
                [best_ptr, matched_ptr, mapping_ptr, still_running_ptr]
                MM_HOST_DEVICE (const int v) {
                    if (matched_ptr[v])
                        return;
                    const int v_best = best_ptr[v];
                    if (v_best == -1 || matched_ptr[v_best])
                        return;
                    if (best_ptr[v_best] == v) {
                        const int lo = (v < v_best) ? v : v_best;
                        const int hi = (v < v_best) ? v_best : v;
                        mapping_ptr[hi] = lo;
                        matched_ptr[v] = 1;
                        matched_ptr[v_best] = 1;
                        *still_running_ptr = 1;
                    }
                });
        }

        int current_num_matched = thrust::reduce(v_matched.begin(), v_matched.end(), 0);
        float rel_increase = (current_num_matched - prev_num_matched) / (prev_num_matched + 1.0f);
        if (verbose)
            std::cout << "matched sum: " << current_num_matched
                      << ", rel_increase: " << rel_increase << "\n";
        prev_num_matched = current_num_matched;

        if (!still_running[0] || rel_increase < 0.1f)
            break;
    }

    if (verbose) {
        std::cout << "# vertices = " << num_nodes << "\n";
        std::cout << "# matched edges = " << prev_num_matched / 2
                  << " / " << A.num_edges() << "\n";
    }

    return {node_mapping, prev_num_matched};
}

// Explicit instantiation declarations.
extern template
std::tuple<thrust::host_vector<int>, int>
filter_edges_by_matching<thrust::host_vector>(const Graph<thrust::host_vector>&, float, bool);

extern template
std::tuple<thrust::device_vector<int>, int>
filter_edges_by_matching<thrust::device_vector>(const Graph<thrust::device_vector>&, float, bool);