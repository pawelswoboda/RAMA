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
    const int idx)
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
inline int count_quadrangle_triangles(
    const int v1, const int v2,
    const int* const __restrict__ offsets,
    const int* const __restrict__ col_ids)
{
    int count = 0;
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
CC_HOST_DEVICE
inline void fill_quadrangle_triangles(
    const int v1, const int v2,
    const int* const __restrict__ offsets,
    const int* const __restrict__ col_ids,
    int* const __restrict__ tri_v1,
    int* const __restrict__ tri_v2,
    int* const __restrict__ tri_v3,
    const int write_offset)
{
    int local_idx = 0;
    for (int i = offsets[v1]; i < offsets[v1 + 1]; ++i)
    {
        const int w = col_ids[i];
        int w_idx = offsets[w];
        int v2_idx = offsets[v2];
        while (w_idx < offsets[w + 1] && v2_idx < offsets[v2 + 1])
        {
            const int w_n = col_ids[w_idx];
            const int v2_n = col_ids[v2_idx];
            if (w_n == v2_n)
            {
                const int m = w_n;
                write_sorted_triangle(v1, v2, w,
                    tri_v1, tri_v2, tri_v3, write_offset + local_idx);
                ++local_idx;
                write_sorted_triangle(v2, w, m,
                    tri_v1, tri_v2, tri_v3, write_offset + local_idx);
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
    const int new_size = thrust::distance(first, new_last);

    tri_v1.resize(new_size);
    tri_v2.resize(new_size);
    tri_v3.resize(new_size);
}

// Find all quadrangles (4-cycles) formed by repulsive edges and a symmetric
// positive graph, and decompose each into two triangles.
//
// For each repulsive edge (v1, v2), for each neighbor w of v1 in the positive
// graph, finds all common neighbours m of (w, v2). Each such path
// v1-w-m-v2 is a 4-cycle, decomposed into triangles (v1,v2,w) and (v2,w,m).
//
// Returns deduplicated triangles with sorted vertices (v1 < v2 < v3).
//
// rep_edge_tails, rep_edge_heads: endpoints of repulsive (negative) edges.
// pos_graph_offsets: CSR row offsets of the symmetric positive graph.
// pos_graph_heads: CSR column indices of the symmetric positive graph.
template<template<typename> class VectorType>
std::tuple<VectorType<int>, VectorType<int>, VectorType<int>>
find_quadrangles(
    const VectorType<int>& rep_edge_tails,
    const VectorType<int>& rep_edge_heads,
    const VectorType<int>& pos_graph_offsets,
    const VectorType<int>& pos_graph_heads)
{
    const int num_rep_edges = rep_edge_tails.size();
    assert(num_rep_edges == (int)rep_edge_heads.size());

    if (num_rep_edges == 0)
        return {VectorType<int>(), VectorType<int>(), VectorType<int>()};

    const int* tails_ptr = thrust::raw_pointer_cast(rep_edge_tails.data());
    const int* heads_ptr = thrust::raw_pointer_cast(rep_edge_heads.data());
    const int* pos_offsets_ptr = thrust::raw_pointer_cast(pos_graph_offsets.data());
    const int* pos_heads_ptr = thrust::raw_pointer_cast(pos_graph_heads.data());

    // Pass 1: count triangles per repulsive edge.
    VectorType<int> counts(num_rep_edges);
    thrust::transform(
        thrust::make_counting_iterator(0),
        thrust::make_counting_iterator(num_rep_edges),
        counts.begin(),
        [tails_ptr, heads_ptr, pos_offsets_ptr, pos_heads_ptr]
        CC_HOST_DEVICE (const int edge_idx) {
            return qd_detail::count_quadrangle_triangles(
                tails_ptr[edge_idx], heads_ptr[edge_idx],
                pos_offsets_ptr, pos_heads_ptr);
        });

    const int total_triangles = thrust::reduce(counts.begin(), counts.end());
    if (total_triangles == 0)
        return {VectorType<int>(), VectorType<int>(), VectorType<int>()};

    // Compute write offsets via exclusive scan.
    VectorType<int> offsets(num_rep_edges);
    thrust::exclusive_scan(counts.begin(), counts.end(), offsets.begin());

    // Allocate output.
    VectorType<int> tri_v1(total_triangles);
    VectorType<int> tri_v2(total_triangles);
    VectorType<int> tri_v3(total_triangles);

    int* tri_v1_ptr = thrust::raw_pointer_cast(tri_v1.data());
    int* tri_v2_ptr = thrust::raw_pointer_cast(tri_v2.data());
    int* tri_v3_ptr = thrust::raw_pointer_cast(tri_v3.data());
    const int* offsets_ptr = thrust::raw_pointer_cast(offsets.data());

    // Pass 2: fill triangles at computed offsets.
    thrust::for_each(
        thrust::make_counting_iterator(0),
        thrust::make_counting_iterator(num_rep_edges),
        [tails_ptr, heads_ptr, pos_offsets_ptr, pos_heads_ptr,
         tri_v1_ptr, tri_v2_ptr, tri_v3_ptr, offsets_ptr]
        CC_HOST_DEVICE (const int edge_idx) {
            qd_detail::fill_quadrangle_triangles(
                tails_ptr[edge_idx], heads_ptr[edge_idx],
                pos_offsets_ptr, pos_heads_ptr,
                tri_v1_ptr, tri_v2_ptr, tri_v3_ptr,
                offsets_ptr[edge_idx]);
        });

    // Pass 3: deduplicate.
    deduplicate_triangles<VectorType>(tri_v1, tri_v2, tri_v3);

    return {std::move(tri_v1), std::move(tri_v2), std::move(tri_v3)};
}