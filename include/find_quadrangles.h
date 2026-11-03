#pragma once

#include <cassert>
#include <tuple>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/for_each.h>
#include <thrust/transform.h>
#include <thrust/scan.h>
#include <thrust/reduce.h>
#include <thrust/sort.h>
#include <thrust/unique.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/extrema.h>
#include "find_triangles.h" // for cc_detail::count_common_neighbours

namespace qd_detail {

// Write a triangle with sorted vertices (min, mid, max) at the given index.
CC_HOST_DEVICE
inline void write_sorted_triangle(
    const int a, const int b, const int c,
    int* const __restrict__ tri_v1,
    int* const __restrict__ tri_v2,
    int* const __restrict__ tri_v3,
    const long long idx)
{
    const int min_v = thrust::min(a, thrust::min(b, c));
    const int max_v = thrust::max(a, thrust::max(b, c));
    const int mid_v = a + b + c - min_v - max_v;
    tri_v1[idx] = min_v;
    tri_v2[idx] = mid_v;
    tri_v3[idx] = max_v;
}

// Count triangles produced by quadrangles for repulsive edge (v1, v2).
// For each neighbor w of v1, count common neighbours of (w, v2).
// Each common neighbour yields one quadrangle = 2 triangles.
CC_HOST_DEVICE
inline long long count_quadrangle_triangles(
    const int v1, const int v2,
    const int* const __restrict__ offsets,
    const int* const __restrict__ col_ids)
{
    long long count = 0;
    for (int i = offsets[v1]; i < offsets[v1 + 1]; ++i)
    {
        const int w = col_ids[i];
        count += cc_detail::count_common_neighbours(w, v2, offsets, col_ids);
    }
    return 2 * count;
}

// Fill triangles for quadrangles of repulsive edge (v1, v2).
// For each neighbor w of v1 and each common neighbour m of (w, v2),
// writes two sorted triangles: (v1, v2, w) and (v2, w, m).
// costs: CSR edge costs (parallel to col_ids), rep_cost: cost of the repulsive edge.
// strength: output per-triangle strength = min(|rep_cost|, |v1-w|, |w-m|, |m-v2|).
CC_HOST_DEVICE
inline void fill_quadrangle_triangles(
    const int v1, const int v2,
    const int* const __restrict__ offsets,
    const int* const __restrict__ col_ids,
    const float* const __restrict__ costs,
    const float rep_cost,
    int* const __restrict__ tri_v1,
    int* const __restrict__ tri_v2,
    int* const __restrict__ tri_v3,
    float* const __restrict__ strength,
    const long long write_offset)
{
    long long local_idx = 0;
    const float abs_rep = rep_cost < 0 ? -rep_cost : rep_cost;
    for (int i = offsets[v1]; i < offsets[v1 + 1]; ++i)
    {
        const int w = col_ids[i];
        const float abs_v1w = costs[i] < 0 ? -costs[i] : costs[i];
        int w_idx = offsets[w];
        int v2_idx = offsets[v2];
        while (w_idx < offsets[w + 1] && v2_idx < offsets[v2 + 1])
        {
            const int w_n = col_ids[w_idx];
            const int v2_n = col_ids[v2_idx];
            if (w_n == v2_n)
            {
                const int m = w_n;
                const float abs_wm = costs[w_idx] < 0 ? -costs[w_idx] : costs[w_idx];
                const float abs_mv2 = costs[v2_idx] < 0 ? -costs[v2_idx] : costs[v2_idx];
                const float s = thrust::min(abs_rep,
                    thrust::min(abs_v1w, thrust::min(abs_wm, abs_mv2)));
                const long long idx0 = write_offset + local_idx;
                write_sorted_triangle(v1, v2, w,
                    tri_v1, tri_v2, tri_v3, idx0);
                strength[idx0] = s;
                ++local_idx;
                const long long idx1 = write_offset + local_idx;
                write_sorted_triangle(v2, w, m,
                    tri_v1, tri_v2, tri_v3, idx1);
                strength[idx1] = s;
                ++local_idx;
                ++w_idx;
                ++v2_idx;
            }
            else if (w_n < v2_n)
                ++w_idx;
            else
                ++v2_idx;
        }
    }
}

} // namespace qd_detail

// Remove duplicate triangles from three parallel vectors.
// Sorts lexicographically by (v1, v2, v3) then removes consecutive duplicates.
// Vectors are resized to the deduplicated count.
template<template<typename> class VectorType>
void deduplicate_triangles(
    VectorType<int>& tri_v1,
    VectorType<int>& tri_v2,
    VectorType<int>& tri_v3)
{
    if (tri_v1.size() == 0)
        return;

    auto first = thrust::make_zip_iterator(
        thrust::make_tuple(tri_v1.begin(), tri_v2.begin(), tri_v3.begin()));
    auto last = thrust::make_zip_iterator(
        thrust::make_tuple(tri_v1.end(), tri_v2.end(), tri_v3.end()));

    thrust::sort(first, last);
    auto new_last = thrust::unique(first, last);
    const int new_size = new_last - first;

    tri_v1.resize(new_size);
    tri_v2.resize(new_size);
    tri_v3.resize(new_size);
}

// Remove duplicate triangles, keeping the one with maximum strength.
// Sorts by (v1, v2, v3, -strength), then unique on (v1, v2, v3) keeps first (max strength).
template<template<typename> class VectorType>
void deduplicate_triangles(
    VectorType<int>& tri_v1,
    VectorType<int>& tri_v2,
    VectorType<int>& tri_v3,
    VectorType<float>& strength)
{
    if (tri_v1.size() == 0)
        return;

    // Negate strength so that sorting ascending gives highest strength first.
    thrust::transform(strength.begin(), strength.end(), strength.begin(),
        [] CC_HOST_DEVICE (float s) { return -s; });

    auto first = thrust::make_zip_iterator(
        thrust::make_tuple(tri_v1.begin(), tri_v2.begin(), tri_v3.begin(), strength.begin()));
    auto last = thrust::make_zip_iterator(
        thrust::make_tuple(tri_v1.end(), tri_v2.end(), tri_v3.end(), strength.end()));

    thrust::sort(first, last);

    // Unique on (v1, v2, v3), ignoring strength — first occurrence (max strength) wins.
    auto new_last = thrust::unique(first, last,
        [] CC_HOST_DEVICE (const thrust::tuple<int,int,int,float>& a,
                           const thrust::tuple<int,int,int,float>& b) {
            return thrust::get<0>(a) == thrust::get<0>(b) &&
                   thrust::get<1>(a) == thrust::get<1>(b) &&
                   thrust::get<2>(a) == thrust::get<2>(b);
        });
    const int new_size = new_last - first;

    tri_v1.resize(new_size);
    tri_v2.resize(new_size);
    tri_v3.resize(new_size);
    strength.resize(new_size);

    // Restore positive strength.
    thrust::transform(strength.begin(), strength.end(), strength.begin(),
        [] CC_HOST_DEVICE (float s) { return -s; });
}

// Find all quadrangles (4-cycles) formed by repulsive edges and a symmetric
// positive graph, and decompose each into two triangles.
//
// For each repulsive edge (v1, v2), for each neighbor w of v1 in the positive
// graph, finds all common neighbours m of (w, v2). Each such path
// v1-w-m-v2 is a 4-cycle, decomposed into triangles (v1,v2,w) and (v2,w,m).
//
// Returns deduplicated triangles with sorted vertices (v1 < v2 < v3) and strength.
//
// rep_edge_tails, rep_edge_heads: endpoints of repulsive (negative) edges.
// pos_graph_offsets: CSR row offsets of the symmetric positive graph.
// pos_graph_heads: CSR column indices of the symmetric positive graph.
// pos_graph_costs: CSR edge costs (parallel to pos_graph_heads).
// rep_edge_costs: costs of repulsive edges (parallel to rep_edge_tails/heads).
// max_triangles: cap on the number of triangles emitted (0 = unlimited).
template<template<typename> class VectorType>
std::tuple<VectorType<int>, VectorType<int>, VectorType<int>, VectorType<float>>
find_quadrangles(
    const VectorType<int>& rep_edge_tails,
    const VectorType<int>& rep_edge_heads,
    const VectorType<int>& pos_graph_offsets,
    const VectorType<int>& pos_graph_heads,
    const VectorType<float>& pos_graph_costs,
    const VectorType<float>& rep_edge_costs,
    const long long max_triangles = 0)
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
    VectorType<long long> counts(num_rep_edges);
    thrust::transform(
        thrust::make_counting_iterator(0),
        thrust::make_counting_iterator(num_rep_edges),
        counts.begin(),
        [tails_ptr, heads_ptr, pos_offsets_ptr, pos_heads_ptr]
        CC_HOST_DEVICE (const int edge_idx) -> long long {
            return qd_detail::count_quadrangle_triangles(
                tails_ptr[edge_idx], heads_ptr[edge_idx],
                pos_offsets_ptr, pos_heads_ptr);
        });

    // Compute write offsets via exclusive scan (64-bit, see helper).
    VectorType<long long> offsets(num_rep_edges);
    const long long total_triangles =
        cc_detail::compute_write_offsets<VectorType>(counts, offsets, max_triangles);
    if (total_triangles == 0)
        return {VectorType<int>(), VectorType<int>(), VectorType<int>(), VectorType<float>()};

    // Allocate output.
    VectorType<int> tri_v1(total_triangles);
    VectorType<int> tri_v2(total_triangles);
    VectorType<int> tri_v3(total_triangles);
    VectorType<float> tri_strength(total_triangles);

    int* tri_v1_ptr = thrust::raw_pointer_cast(tri_v1.data());
    int* tri_v2_ptr = thrust::raw_pointer_cast(tri_v2.data());
    int* tri_v3_ptr = thrust::raw_pointer_cast(tri_v3.data());
    float* tri_str_ptr = thrust::raw_pointer_cast(tri_strength.data());
    const long long* offsets_ptr = thrust::raw_pointer_cast(offsets.data());

    // Pass 2: fill triangles at computed offsets. Edges zeroed out by the
    // cap are skipped -- the fill helper re-enumerates the quadrangles and
    // would otherwise write past the end of the output.
    const long long* counts_ptr = thrust::raw_pointer_cast(counts.data());
    thrust::for_each(
        thrust::make_counting_iterator(0),
        thrust::make_counting_iterator(num_rep_edges),
        [tails_ptr, heads_ptr, pos_offsets_ptr, pos_heads_ptr, pos_costs_ptr,
         rep_costs_ptr, tri_v1_ptr, tri_v2_ptr, tri_v3_ptr, tri_str_ptr,
         offsets_ptr, counts_ptr]
        CC_HOST_DEVICE (const int edge_idx) {
            if (counts_ptr[edge_idx] == 0)
                return;
            qd_detail::fill_quadrangle_triangles(
                tails_ptr[edge_idx], heads_ptr[edge_idx],
                pos_offsets_ptr, pos_heads_ptr,
                pos_costs_ptr, rep_costs_ptr[edge_idx],
                tri_v1_ptr, tri_v2_ptr, tri_v3_ptr, tri_str_ptr,
                offsets_ptr[edge_idx]);
        });

    // Pass 3: deduplicate (keeping max strength).
    deduplicate_triangles<VectorType>(tri_v1, tri_v2, tri_v3, tri_strength);

    return {std::move(tri_v1), std::move(tri_v2), std::move(tri_v3), std::move(tri_strength)};
}

// Explicit instantiation declarations.
extern template
void deduplicate_triangles<thrust::host_vector>(
    thrust::host_vector<int>&, thrust::host_vector<int>&, thrust::host_vector<int>&);
extern template
void deduplicate_triangles<thrust::device_vector>(
    thrust::device_vector<int>&, thrust::device_vector<int>&, thrust::device_vector<int>&);

extern template
void deduplicate_triangles<thrust::host_vector>(
    thrust::host_vector<int>&, thrust::host_vector<int>&, thrust::host_vector<int>&,
    thrust::host_vector<float>&);
extern template
void deduplicate_triangles<thrust::device_vector>(
    thrust::device_vector<int>&, thrust::device_vector<int>&, thrust::device_vector<int>&,
    thrust::device_vector<float>&);

extern template
std::tuple<thrust::host_vector<int>, thrust::host_vector<int>, thrust::host_vector<int>, thrust::host_vector<float>>
find_quadrangles<thrust::host_vector>(
    const thrust::host_vector<int>&, const thrust::host_vector<int>&,
    const thrust::host_vector<int>&, const thrust::host_vector<int>&,
    const thrust::host_vector<float>&, const thrust::host_vector<float>&,
    long long);

extern template
std::tuple<thrust::device_vector<int>, thrust::device_vector<int>, thrust::device_vector<int>, thrust::device_vector<float>>
find_quadrangles<thrust::device_vector>(
    const thrust::device_vector<int>&, const thrust::device_vector<int>&,
    const thrust::device_vector<int>&, const thrust::device_vector<int>&,
    const thrust::device_vector<float>&, const thrust::device_vector<float>&,
    long long);