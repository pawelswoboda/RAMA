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
#include "find_quadrangles.h" // for deduplicate_triangles, qd_detail::write_sorted_triangle

namespace pent_detail {

// Count common neighbours of v1 and v2, excluding nodes excl1 and excl2.
// Adjacency lists must be sorted by column id.
CC_HOST_DEVICE
inline int count_common_neighbours_excluding(
    const int v1, const int v2,
    const int* const __restrict__ offsets,
    const int* const __restrict__ col_ids,
    const int excl1, const int excl2)
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
            if (v1_n != excl1 && v1_n != excl2)
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

// Count triangles produced by pentagons for repulsive edge (v1, v2).
// For each pair (v1_n1, v2_n1) of neighbours of v1 and v2 respectively,
// where v1_n1 != v2_n1, count common neighbours of (v1_n1, v2_n1) excluding
// v1 and v2. Each valid common neighbour yields one pentagon = 3 triangles.
CC_HOST_DEVICE
inline int count_pentagon_triangles(
    const int v1, const int v2,
    const int* const __restrict__ offsets,
    const int* const __restrict__ col_ids)
{
    int count = 0;
    for (int i = offsets[v1]; i < offsets[v1 + 1]; ++i)
    {
        const int v1_n1 = col_ids[i];
        for (int j = offsets[v2]; j < offsets[v2 + 1]; ++j)
        {
            const int v2_n1 = col_ids[j];
            if (v1_n1 == v2_n1)
                continue;
            count += count_common_neighbours_excluding(
                v1_n1, v2_n1, offsets, col_ids, v1, v2);
        }
    }
    return 3 * count;
}

// Fill triangles for pentagons of repulsive edge (v1, v2).
// For each pentagon v1 - v1_n1 - mid - v2_n1 - v2, writes three sorted
// triangles: (v1, v2, v1_n1), (v2, v1_n1, mid), (v2, mid, v2_n1).
CC_HOST_DEVICE
inline void fill_pentagon_triangles(
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
        const int v1_n1 = col_ids[i];
        for (int j = offsets[v2]; j < offsets[v2 + 1]; ++j)
        {
            const int v2_n1 = col_ids[j];
            if (v1_n1 == v2_n1)
                continue;
            // Two-pointer merge to find common neighbours excluding v1, v2
            int a_idx = offsets[v1_n1];
            int b_idx = offsets[v2_n1];
            while (a_idx < offsets[v1_n1 + 1] && b_idx < offsets[v2_n1 + 1])
            {
                const int a_n = col_ids[a_idx];
                const int b_n = col_ids[b_idx];
                if (a_n == b_n)
                {
                    const int mid = a_n;
                    if (mid != v1 && mid != v2)
                    {
                        qd_detail::write_sorted_triangle(v1, v2, v1_n1,
                            tri_v1, tri_v2, tri_v3, write_offset + local_idx);
                        ++local_idx;
                        qd_detail::write_sorted_triangle(v2, v1_n1, mid,
                            tri_v1, tri_v2, tri_v3, write_offset + local_idx);
                        ++local_idx;
                        qd_detail::write_sorted_triangle(v2, mid, v2_n1,
                            tri_v1, tri_v2, tri_v3, write_offset + local_idx);
                        ++local_idx;
                    }
                    ++a_idx;
                    ++b_idx;
                }
                else if (a_n < b_n)
                    ++a_idx;
                else
                    ++b_idx;
            }
        }
    }
}

} // namespace pent_detail

// Find all pentagons (5-cycles) formed by repulsive edges and a symmetric
// positive graph, and decompose each into three triangles.
//
// For each repulsive edge (v1, v2), for each pair of neighbours (v1_n1 of v1,
// v2_n1 of v2) where v1_n1 != v2_n1, finds all common neighbours mid of
// (v1_n1, v2_n1) excluding v1 and v2. Each such path v1-v1_n1-mid-v2_n1-v2
// is a 5-cycle, decomposed into triangles (v1,v2,v1_n1), (v2,v1_n1,mid),
// and (v2,mid,v2_n1).
//
// Returns deduplicated triangles with sorted vertices (v1 < v2 < v3).
//
// rep_edge_tails, rep_edge_heads: endpoints of repulsive (negative) edges.
// pos_graph_offsets: CSR row offsets of the symmetric positive graph.
// pos_graph_heads: CSR column indices of the symmetric positive graph.
template<template<typename> class VectorType>
std::tuple<VectorType<int>, VectorType<int>, VectorType<int>>
find_pentagons(
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
            return pent_detail::count_pentagon_triangles(
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
            pent_detail::fill_pentagon_triangles(
                tails_ptr[edge_idx], heads_ptr[edge_idx],
                pos_offsets_ptr, pos_heads_ptr,
                tri_v1_ptr, tri_v2_ptr, tri_v3_ptr,
                offsets_ptr[edge_idx]);
        });

    // Pass 3: deduplicate.
    deduplicate_triangles<VectorType>(tri_v1, tri_v2, tri_v3);

    return {std::move(tri_v1), std::move(tri_v2), std::move(tri_v3)};
}
