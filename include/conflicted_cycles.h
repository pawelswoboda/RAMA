#pragma once

#include <tuple>
#include <iostream>
#include <cfloat>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/copy.h>
#include <thrust/count.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include "graph.h"
#include "find_triangles.h"
#include "find_quadrangles.h"
#include "find_pentagons.h"

// Find all conflicted cycles up to max_cycle_length (3-5) in the graph,
// decompose them into triangles, and return deduplicated sorted triangles.
//
// A conflicted cycle contains at least one repulsive (negative cost) edge
// and connects through positive-cost edges. The function:
// 1. Extracts repulsive edges (single orientation, cost < thresh)
// 2. Builds a positive subgraph (cost >= -thresh) in CSR format
// 3. Finds conflicted triangles, quadrangles, and pentagons as requested
// 4. Concatenates and deduplicates all resulting triangles
//
// Returns three vectors of equal length: (v1, v2, v3) with v1 < v2 < v3.
template<template<typename> class VectorType>
std::tuple<VectorType<int>, VectorType<int>, VectorType<int>>
conflicted_cycles(const Graph<VectorType>& G, int max_cycle_length,
                  float tol_ratio = 1e-4, bool verbose = true)
{
    if (max_cycle_length > 5)
        throw std::runtime_error("max_cycle_length should be <= 5. Received: " + std::to_string(max_cycle_length));

    if (max_cycle_length < 3)
        return {VectorType<int>(), VectorType<int>(), VectorType<int>()};

    // Separate thresholds for repulsive and attractive edges.
    const float neg_thresh = tol_ratio * G.min();  // negative (G.min() < 0)
    const float pos_thresh = tol_ratio * G.max();  // positive (G.max() > 0)

    // No repulsive or no attractive edges → no conflicted cycles.
    if (neg_thresh >= 0 || pos_thresh <= 0)
        return {VectorType<int>(), VectorType<int>(), VectorType<int>()};

    // Extract repulsive edges (single orientation: tail < head).
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

    // Use copy_if with a zip iterator to extract tail/head pairs.
    auto src_first = thrust::make_zip_iterator(
        thrust::make_tuple(G.get_tails().begin(), G.get_heads().begin()));
    auto src_last = thrust::make_zip_iterator(
        thrust::make_tuple(G.get_tails().end(), G.get_heads().end()));
    auto dst_first = thrust::make_zip_iterator(
        thrust::make_tuple(rep_tails.begin(), rep_heads.begin()));

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

    // Find triangles from 3-cycles.
    VectorType<int> all_v1, all_v2, all_v3;

    if (max_cycle_length >= 3)
    {
        auto [v1, v2, v3] = find_triangles<VectorType>(
            rep_tails, rep_heads, pos_offsets, pos_heads);
        if (verbose)
            std::cout << "3-cycles: found " << v1.size() << " triangles\n";
        all_v1 = std::move(v1);
        all_v2 = std::move(v2);
        all_v3 = std::move(v3);
    }

    // Find triangles from 4-cycles.
    if (max_cycle_length >= 4)
    {
        auto [v1, v2, v3] = find_quadrangles<VectorType>(
            rep_tails, rep_heads, pos_offsets, pos_heads);
        if (verbose)
            std::cout << "4-cycles: found " << v1.size() << " triangles\n";
        if (v1.size() > 0)
        {
            const size_t old_size = all_v1.size();
            all_v1.resize(old_size + v1.size());
            all_v2.resize(old_size + v2.size());
            all_v3.resize(old_size + v3.size());
            thrust::copy(v1.begin(), v1.end(), all_v1.begin() + old_size);
            thrust::copy(v2.begin(), v2.end(), all_v2.begin() + old_size);
            thrust::copy(v3.begin(), v3.end(), all_v3.begin() + old_size);
        }
    }

    // Find triangles from 5-cycles.
    if (max_cycle_length >= 5)
    {
        auto [v1, v2, v3] = find_pentagons<VectorType>(
            rep_tails, rep_heads, pos_offsets, pos_heads);
        if (verbose)
            std::cout << "5-cycles: found " << v1.size() << " triangles\n";
        if (v1.size() > 0)
        {
            const size_t old_size = all_v1.size();
            all_v1.resize(old_size + v1.size());
            all_v2.resize(old_size + v2.size());
            all_v3.resize(old_size + v3.size());
            thrust::copy(v1.begin(), v1.end(), all_v1.begin() + old_size);
            thrust::copy(v2.begin(), v2.end(), all_v2.begin() + old_size);
            thrust::copy(v3.begin(), v3.end(), all_v3.begin() + old_size);
        }
    }

    // Deduplicate combined result (quadrangles/pentagons may share triangles
    // with each other or with the 3-cycle triangles).
    deduplicate_triangles<VectorType>(all_v1, all_v2, all_v3);

    return {std::move(all_v1), std::move(all_v2), std::move(all_v3)};
}
