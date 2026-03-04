#pragma once

#include <cassert>
#include <tuple>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/for_each.h>
#include <thrust/transform.h>
#include <thrust/scan.h>
#include <thrust/reduce.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/extrema.h>

#ifdef __CUDACC__
#define CC_HOST_DEVICE __host__ __device__
#else
#define CC_HOST_DEVICE
#endif

namespace cc_detail {

// Count common neighbours of v1 and v2 in a symmetric CSR graph.
// offsets: CSR row offsets (size num_nodes + 1), col_ids: CSR column indices.
// Adjacency lists must be sorted by column id.
CC_HOST_DEVICE
inline int count_common_neighbours(
    const int v1, const int v2,
    const int* const __restrict__ offsets,
    const int* const __restrict__ col_ids)
{
    int v1_idx = offsets[v1];
    int v2_idx = offsets[v2];
    int count = 0;
    while (v1_idx < offsets[v1 + 1] && v2_idx < offsets[v2 + 1])
    {
        const int v1_n = col_ids[v1_idx];
        const int v2_n = col_ids[v2_idx];
        if (v1_n == v2_n)
        {
            ++count;
            ++v1_idx;
            ++v2_idx;
        }
        else if (v1_n < v2_n)
            ++v1_idx;
        else
            ++v2_idx;
    }
    return count;
}

// Write common-neighbour triangles for edge (v1, v2) into output arrays.
// Each triangle is stored with sorted vertices (min, mid, max).
// write_offset: starting index in the output arrays for this edge.
// costs: CSR edge costs (parallel to col_ids), rep_cost: cost of the repulsive edge.
// strength: output array for per-triangle strength = min(|rep_cost|, |cost_v1_w|, |cost_v2_w|).
CC_HOST_DEVICE
inline void fill_triangles_for_edge(
    const int v1, const int v2,
    const int* const __restrict__ offsets,
    const int* const __restrict__ col_ids,
    const float* const __restrict__ costs,
    const float rep_cost,
    int* const __restrict__ tri_v1,
    int* const __restrict__ tri_v2,
    int* const __restrict__ tri_v3,
    float* const __restrict__ strength,
    const int write_offset)
{
    int v1_idx = offsets[v1];
    int v2_idx = offsets[v2];
    int local_idx = 0;
    const float abs_rep = rep_cost < 0 ? -rep_cost : rep_cost;
    while (v1_idx < offsets[v1 + 1] && v2_idx < offsets[v2 + 1])
    {
        const int v1_n = col_ids[v1_idx];
        const int v2_n = col_ids[v2_idx];
        if (v1_n == v2_n)
        {
            const int min_v = thrust::min(v1, thrust::min(v2, v1_n));
            const int max_v = thrust::max(v1, thrust::max(v2, v1_n));
            const int mid_v = v1 + v2 + v1_n - min_v - max_v;
            const int idx = write_offset + local_idx;
            tri_v1[idx] = min_v;
            tri_v2[idx] = mid_v;
            tri_v3[idx] = max_v;
            const float c1 = costs[v1_idx] < 0 ? -costs[v1_idx] : costs[v1_idx];
            const float c2 = costs[v2_idx] < 0 ? -costs[v2_idx] : costs[v2_idx];
            strength[idx] = thrust::min(abs_rep, thrust::min(c1, c2));
            ++local_idx;
            ++v1_idx;
            ++v2_idx;
        }
        else if (v1_n < v2_n)
            ++v1_idx;
        else
            ++v2_idx;
    }
}

} // namespace cc_detail

// Find all triangles formed by repulsive edges and a symmetric positive graph.
// For each repulsive edge (tail, head), finds all common neighbours in the
// positive graph and emits one triangle per common neighbour.
// Triangles are returned with sorted vertices (v1 < v2 < v3) and are
// unique (no deduplication needed).
//
// rep_edge_tails, rep_edge_heads: endpoints of repulsive (negative) edges.
// pos_graph_offsets: CSR row offsets of the symmetric positive graph (size >= max_node + 2).
// pos_graph_heads: CSR column indices of the symmetric positive graph.
// pos_graph_costs: CSR edge costs (parallel to pos_graph_heads).
// rep_edge_costs: costs of repulsive edges (parallel to rep_edge_tails/heads).
//
// Returns (v1, v2, v3, strength) where strength = min |cost| over all cycle edges.
template<template<typename> class VectorType>
std::tuple<VectorType<int>, VectorType<int>, VectorType<int>, VectorType<float>>
find_triangles(
    const VectorType<int>& rep_edge_tails,
    const VectorType<int>& rep_edge_heads,
    const VectorType<int>& pos_graph_offsets,
    const VectorType<int>& pos_graph_heads,
    const VectorType<float>& pos_graph_costs,
    const VectorType<float>& rep_edge_costs)
{
    const int num_rep_edges = rep_edge_tails.size();
    assert(num_rep_edges == (int)rep_edge_heads.size());

    if (num_rep_edges == 0)
        return {VectorType<int>(), VectorType<int>(), VectorType<int>(), VectorType<float>()};

    const int* tails_ptr = thrust::raw_pointer_cast(rep_edge_tails.data());
    const int* heads_ptr = thrust::raw_pointer_cast(rep_edge_heads.data());
    const int* pos_offsets_ptr = thrust::raw_pointer_cast(pos_graph_offsets.data());
    const int* pos_heads_ptr = thrust::raw_pointer_cast(pos_graph_heads.data());
    const float* pos_costs_ptr = thrust::raw_pointer_cast(pos_graph_costs.data());
    const float* rep_costs_ptr = thrust::raw_pointer_cast(rep_edge_costs.data());

    // Pass 1: count triangles per repulsive edge.
    VectorType<int> counts(num_rep_edges);
    thrust::transform(
        thrust::make_counting_iterator(0),
        thrust::make_counting_iterator(num_rep_edges),
        counts.begin(),
        [tails_ptr, heads_ptr, pos_offsets_ptr, pos_heads_ptr]
        CC_HOST_DEVICE (const int edge_idx) {
            return cc_detail::count_common_neighbours(
                tails_ptr[edge_idx], heads_ptr[edge_idx],
                pos_offsets_ptr, pos_heads_ptr);
        });

    const int total_triangles = thrust::reduce(counts.begin(), counts.end());
    if (total_triangles == 0)
        return {VectorType<int>(), VectorType<int>(), VectorType<int>(), VectorType<float>()};

    // Compute write offsets via exclusive scan.
    VectorType<int> offsets(num_rep_edges);
    thrust::exclusive_scan(counts.begin(), counts.end(), offsets.begin());

    // Allocate output.
    VectorType<int> tri_v1(total_triangles);
    VectorType<int> tri_v2(total_triangles);
    VectorType<int> tri_v3(total_triangles);
    VectorType<float> tri_strength(total_triangles);

    int* tri_v1_ptr = thrust::raw_pointer_cast(tri_v1.data());
    int* tri_v2_ptr = thrust::raw_pointer_cast(tri_v2.data());
    int* tri_v3_ptr = thrust::raw_pointer_cast(tri_v3.data());
    float* tri_str_ptr = thrust::raw_pointer_cast(tri_strength.data());
    const int* offsets_ptr = thrust::raw_pointer_cast(offsets.data());

    // Pass 2: fill triangles at computed offsets.
    thrust::for_each(
        thrust::make_counting_iterator(0),
        thrust::make_counting_iterator(num_rep_edges),
        [tails_ptr, heads_ptr, pos_offsets_ptr, pos_heads_ptr, pos_costs_ptr,
         rep_costs_ptr, tri_v1_ptr, tri_v2_ptr, tri_v3_ptr, tri_str_ptr, offsets_ptr]
        CC_HOST_DEVICE (const int edge_idx) {
            cc_detail::fill_triangles_for_edge(
                tails_ptr[edge_idx], heads_ptr[edge_idx],
                pos_offsets_ptr, pos_heads_ptr,
                pos_costs_ptr, rep_costs_ptr[edge_idx],
                tri_v1_ptr, tri_v2_ptr, tri_v3_ptr, tri_str_ptr,
                offsets_ptr[edge_idx]);
        });

    return {std::move(tri_v1), std::move(tri_v2), std::move(tri_v3), std::move(tri_strength)};
}

// Explicit instantiation declarations.
extern template
std::tuple<thrust::host_vector<int>, thrust::host_vector<int>, thrust::host_vector<int>, thrust::host_vector<float>>
find_triangles<thrust::host_vector>(
    const thrust::host_vector<int>&, const thrust::host_vector<int>&,
    const thrust::host_vector<int>&, const thrust::host_vector<int>&,
    const thrust::host_vector<float>&, const thrust::host_vector<float>&);

extern template
std::tuple<thrust::device_vector<int>, thrust::device_vector<int>, thrust::device_vector<int>, thrust::device_vector<float>>
find_triangles<thrust::device_vector>(
    const thrust::device_vector<int>&, const thrust::device_vector<int>&,
    const thrust::device_vector<int>&, const thrust::device_vector<int>&,
    const thrust::device_vector<float>&, const thrust::device_vector<float>&);