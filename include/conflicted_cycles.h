#pragma once

#include <tuple>
#include <iostream>
#include <cfloat>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/copy.h>
#include <thrust/count.h>
#include <thrust/sort.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include "graph.h"
#include "find_triangles.h"
#include "find_quadrangles.h"
#include "find_pentagons.h"
#include "find_cycles_bfs.h"
#include "find_cycles_mst.h"

// Find all conflicted cycles up to max_cycle_length (3-5) in the graph,
// decompose them into triangles, and return deduplicated sorted triangles.
//
// A conflicted cycle contains at least one repulsive (negative cost) edge
// and connects through positive-cost edges. The function:
// 1. Extracts repulsive edges (single orientation, cost < thresh)
// 2. Builds a positive subgraph (cost >= -thresh) in CSR format
// 3. Finds conflicted triangles, quadrangles, and pentagons as requested
// 4. Concatenates and deduplicates all resulting triangles
// 5. If triangle_budget_ratio > 0, keeps only top triangles by strength
//
// Returns three vectors of equal length: (v1, v2, v3) with v1 < v2 < v3.
template<template<typename> class VectorType>
std::tuple<VectorType<int>, VectorType<int>, VectorType<int>>
conflicted_cycles(const Graph<VectorType>& G, int max_cycle_length,
                  float tol_ratio = 1e-4, bool verbose = true,
                  const std::string& long_cycle_method = "bfs",
                  float triangle_budget_ratio = 0)
{
    if (max_cycle_length < 3)
        return {VectorType<int>(), VectorType<int>(), VectorType<int>()};

    // Separate thresholds for repulsive and attractive edges.
    const float neg_thresh = tol_ratio * G.min();  // negative (G.min() < 0)
    const float pos_thresh = tol_ratio * G.max();  // positive (G.max() > 0)

    // No repulsive or no attractive edges → no conflicted cycles.
    if (neg_thresh >= 0 || pos_thresh <= 0)
        return {VectorType<int>(), VectorType<int>(), VectorType<int>()};

    // Extract repulsive edges (single orientation: tail < head) with costs.
    const int num_directed = G.num_directed_edges();
    const int* tails_ptr = G.get_tails_ptr();
    const int* heads_ptr = G.get_heads_ptr();
    const float* costs_ptr = G.get_costs_ptr();

    const int num_rep = thrust::count_if(
        thrust::make_counting_iterator(0),
        thrust::make_counting_iterator(num_directed),
        [tails_ptr, heads_ptr, costs_ptr, neg_thresh]
        CC_HOST_DEVICE (const int i) {
            return costs_ptr[i] < neg_thresh && tails_ptr[i] < heads_ptr[i];
        });

    if (num_rep == 0)
        return {VectorType<int>(), VectorType<int>(), VectorType<int>()};

    VectorType<int> rep_tails(num_rep);
    VectorType<int> rep_heads(num_rep);
    VectorType<float> rep_costs(num_rep);

    // Use copy_if with a zip iterator to extract tail/head/cost triples.
    auto src_first = thrust::make_zip_iterator(
        thrust::make_tuple(G.get_tails().begin(), G.get_heads().begin(), G.get_costs().begin()));
    auto src_last = thrust::make_zip_iterator(
        thrust::make_tuple(G.get_tails().end(), G.get_heads().end(), G.get_costs().end()));
    auto dst_first = thrust::make_zip_iterator(
        thrust::make_tuple(rep_tails.begin(), rep_heads.begin(), rep_costs.begin()));

    thrust::copy_if(
        src_first, src_last,
        thrust::make_counting_iterator(0),
        dst_first,
        [tails_ptr, heads_ptr, costs_ptr, neg_thresh]
        CC_HOST_DEVICE (const int i) {
            return costs_ptr[i] < neg_thresh && tails_ptr[i] < heads_ptr[i];
        });

    // Build positive subgraph in CSR format.
    Graph<VectorType> pos_graph = G.filter(pos_thresh, FLT_MAX);
    if (pos_graph.num_directed_edges() == 0)
        return {VectorType<int>(), VectorType<int>(), VectorType<int>()};

    VectorType<int> pos_offsets = pos_graph.compute_node_offsets();
    const VectorType<int>& pos_heads = pos_graph.get_heads();
    const VectorType<float>& pos_costs = pos_graph.get_costs();

    // Find triangles from 3-cycles.
    VectorType<int> all_v1, all_v2, all_v3;
    VectorType<float> all_strength;

    if (max_cycle_length >= 3)
    {
        auto [v1, v2, v3, str] = find_triangles<VectorType>(
            rep_tails, rep_heads, pos_offsets, pos_heads, pos_costs, rep_costs);
        if (verbose)
            std::cout << "3-cycles: found " << v1.size() << " triangles\n";
        all_v1 = std::move(v1);
        all_v2 = std::move(v2);
        all_v3 = std::move(v3);
        all_strength = std::move(str);
    }

    auto append = [](VectorType<int>& dst, const VectorType<int>& src) {
        if (src.size() == 0) return;
        const size_t old = dst.size();
        dst.resize(old + src.size());
        thrust::copy(src.begin(), src.end(), dst.begin() + old);
    };
    auto append_f = [](VectorType<float>& dst, const VectorType<float>& src) {
        if (src.size() == 0) return;
        const size_t old = dst.size();
        dst.resize(old + src.size());
        thrust::copy(src.begin(), src.end(), dst.begin() + old);
    };

    // Find triangles from 4-cycles.
    if (max_cycle_length >= 4)
    {
        auto [v1, v2, v3, str] = find_quadrangles<VectorType>(
            rep_tails, rep_heads, pos_offsets, pos_heads, pos_costs, rep_costs);
        if (verbose)
            std::cout << "4-cycles: found " << v1.size() << " triangles\n";
        append(all_v1, v1); append(all_v2, v2); append(all_v3, v3);
        append_f(all_strength, str);
    }

    // Find triangles from 5-cycles.
    if (max_cycle_length >= 5)
    {
        auto [v1, v2, v3, str] = find_pentagons<VectorType>(
            rep_tails, rep_heads, pos_offsets, pos_heads, pos_costs, rep_costs);
        if (verbose)
            std::cout << "5-cycles: found " << v1.size() << " triangles\n";
        append(all_v1, v1); append(all_v2, v2); append(all_v3, v3);
        append_f(all_strength, str);
    }

    // Find triangles from 6+-cycles via BFS or MST.
    if (max_cycle_length > 5)
    {
        VectorType<int> v1, v2, v3;
        VectorType<float> str;
        if (long_cycle_method == "mst")
        {
            std::tie(v1, v2, v3, str) = find_conflicted_cycles_mst<VectorType>(
                rep_tails, rep_heads, rep_costs,
                pos_graph.get_tails(), pos_graph.get_heads(), pos_costs,
                G.num_nodes(), max_cycle_length, verbose);
        }
        else
        {
            std::tie(v1, v2, v3, str) = find_conflicted_cycles_bfs<VectorType>(
                rep_tails, rep_heads, pos_offsets, pos_heads,
                pos_costs, rep_costs,
                G.num_nodes(), max_cycle_length, verbose);
        }
        if (verbose)
            std::cout << "6+-cycles (" << long_cycle_method << "): found "
                      << v1.size() << " triangles\n";
        append(all_v1, v1); append(all_v2, v2); append(all_v3, v3);
        append_f(all_strength, str);
    }

    // Deduplicate combined result (keeping max strength per triangle).
    deduplicate_triangles<VectorType>(all_v1, all_v2, all_v3, all_strength);

    // Budget filtering: keep only top triangles by strength.
    if (triangle_budget_ratio > 0 && all_v1.size() > 0)
    {
        const int cap = (int)(triangle_budget_ratio * G.num_edges());
        if (cap > 0 && (int)all_v1.size() > cap)
        {
            const int total_before = (int)all_v1.size();

            // Sort by strength descending (negate, sort ascending, negate back).
            thrust::transform(all_strength.begin(), all_strength.end(), all_strength.begin(),
                [] CC_HOST_DEVICE (float s) { return -s; });

            auto first = thrust::make_zip_iterator(thrust::make_tuple(
                all_strength.begin(), all_v1.begin(), all_v2.begin(), all_v3.begin()));
            auto last = thrust::make_zip_iterator(thrust::make_tuple(
                all_strength.end(), all_v1.end(), all_v2.end(), all_v3.end()));
            thrust::sort(first, last);

            all_v1.resize(cap);
            all_v2.resize(cap);
            all_v3.resize(cap);
            all_strength.resize(cap);

            if (verbose)
                std::cout << "Triangle budget: kept " << cap << " of "
                          << total_before << " triangles\n";
        }
    }

    return {std::move(all_v1), std::move(all_v2), std::move(all_v3)};
}

// Explicit instantiation declarations.
extern template
std::tuple<thrust::host_vector<int>, thrust::host_vector<int>, thrust::host_vector<int>>
conflicted_cycles<thrust::host_vector>(
    const Graph<thrust::host_vector>&, int, float, bool, const std::string&, float);

extern template
std::tuple<thrust::device_vector<int>, thrust::device_vector<int>, thrust::device_vector<int>>
conflicted_cycles<thrust::device_vector>(
    const Graph<thrust::device_vector>&, int, float, bool, const std::string&, float);
