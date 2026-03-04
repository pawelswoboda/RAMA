#pragma once

#include <cassert>
#include <tuple>
#include <iostream>
#include <algorithm>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/for_each.h>
#include <thrust/transform.h>
#include <thrust/scan.h>
#include <thrust/reduce.h>
#include <thrust/copy.h>
#include <thrust/count.h>
#include <thrust/iterator/counting_iterator.h>
#include "find_triangles.h"    // CC_HOST_DEVICE
#include "find_quadrangles.h"  // qd_detail::write_sorted_triangle, deduplicate_triangles
#include "connected_components.h"

// Find conflicted cycles of length > 5 via BFS in the positive subgraph.
// For each repulsive edge (u,v), finds the shortest positive path from u to v.
// If found with path length >= 5 (cycle length >= 6), fan-triangulates from u.
//
// Uses batched multi-source BFS for parallelism, following the pattern in
// lifted_edge_contraction.h.
//
// rep_edge_tails, rep_edge_heads: repulsive edges (single orientation, tail < head).
// pos_graph_offsets: CSR row offsets (size num_nodes + 1) of symmetric positive graph.
// pos_graph_heads: CSR column indices of the symmetric positive graph.
// pos_graph_costs: CSR edge costs (parallel to pos_graph_heads).
// rep_edge_costs: costs of repulsive edges (parallel to rep_edge_tails/heads).
// num_nodes: total node count.
// max_cycle_length: upper bound on cycle length (0 = unlimited).
//
// Returns deduplicated sorted triangles (v1 < v2 < v3) with strength.
template<template<typename> class VectorType>
std::tuple<VectorType<int>, VectorType<int>, VectorType<int>, VectorType<float>>
find_conflicted_cycles_bfs(
    const VectorType<int>& rep_edge_tails,
    const VectorType<int>& rep_edge_heads,
    const VectorType<int>& pos_graph_offsets,
    const VectorType<int>& pos_graph_heads,
    const VectorType<float>& pos_graph_costs,
    const VectorType<float>& rep_edge_costs,
    const int num_nodes,
    const int max_cycle_length,
    const bool verbose = true)
{
    const int num_rep_edges = (int)rep_edge_tails.size();
    assert(num_rep_edges == (int)rep_edge_heads.size());

    if (num_rep_edges == 0 || num_nodes == 0 || pos_graph_heads.size() == 0)
        return {VectorType<int>(), VectorType<int>(), VectorType<int>(), VectorType<float>()};

    const int N = num_nodes;

    // Maximum BFS depth (path edges). Cycle length = depth + 1.
    const int max_depth = (max_cycle_length > 0) ? max_cycle_length - 1 : N;

    // Need depth >= 5 (cycle length >= 6) to produce any output.
    if (max_depth < 5)
        return {VectorType<int>(), VectorType<int>(), VectorType<int>(), VectorType<float>()};

    // --- CC prefilter: discard repulsive edges whose endpoints are in
    //     different positive-graph connected components (no path exists). ---
    const int num_pos_edges = (int)pos_graph_heads.size();
    VectorType<int> pos_tails(num_pos_edges);
    {
        const int* off_ptr = thrust::raw_pointer_cast(pos_graph_offsets.data());
        int* pt = thrust::raw_pointer_cast(pos_tails.data());
        thrust::for_each(
            thrust::make_counting_iterator(0),
            thrust::make_counting_iterator(N),
            [pt, off_ptr] CC_HOST_DEVICE (int node) {
                for (int e = off_ptr[node]; e < off_ptr[node + 1]; ++e)
                    pt[e] = node;
            });
    }

    VectorType<int> comp = connected_components::compute_cc<VectorType>(
        N, pos_tails, pos_graph_heads);

    const int* comp_ptr = thrust::raw_pointer_cast(comp.data());
    const int* rep_t = thrust::raw_pointer_cast(rep_edge_tails.data());
    const int* rep_h = thrust::raw_pointer_cast(rep_edge_heads.data());

    int S = thrust::count_if(
        thrust::make_counting_iterator(0),
        thrust::make_counting_iterator(num_rep_edges),
        [comp_ptr, rep_t, rep_h] CC_HOST_DEVICE (int i) {
            return comp_ptr[rep_t[i]] == comp_ptr[rep_h[i]];
        });

    if (S == 0)
        return {VectorType<int>(), VectorType<int>(), VectorType<int>(), VectorType<float>()};

    if (verbose)
        std::cout << "BFS cycles: " << S << "/" << num_rep_edges
                  << " repulsive edges reachable (CC prefilter)\n";

    auto is_reachable = [comp_ptr, rep_t, rep_h] CC_HOST_DEVICE (int i) {
        return comp_ptr[rep_t[i]] == comp_ptr[rep_h[i]];
    };

    VectorType<int> src_all(S), dst_all(S);
    VectorType<float> rep_costs_all(S);
    thrust::copy_if(rep_edge_tails.begin(), rep_edge_tails.end(),
                    thrust::make_counting_iterator(0), src_all.begin(), is_reachable);
    thrust::copy_if(rep_edge_heads.begin(), rep_edge_heads.end(),
                    thrust::make_counting_iterator(0), dst_all.begin(), is_reachable);
    thrust::copy_if(rep_edge_costs.begin(), rep_edge_costs.end(),
                    thrust::make_counting_iterator(0), rep_costs_all.begin(), is_reachable);

    const int* src_all_ptr = thrust::raw_pointer_cast(src_all.data());
    const int* dst_all_ptr = thrust::raw_pointer_cast(dst_all.data());
    const float* rep_costs_all_ptr = thrust::raw_pointer_cast(rep_costs_all.data());
    const int* pos_off = thrust::raw_pointer_cast(pos_graph_offsets.data());
    const int* pos_h = thrust::raw_pointer_cast(pos_graph_heads.data());
    const float* pos_c = thrust::raw_pointer_cast(pos_graph_costs.data());

    // Batch size: limit dist + parent arrays to ~128 MB total.
    const int max_batch = std::max(1,
        (int)std::min((long long)S,
                      16LL * 1024 * 1024 / std::max((long long)N, 1LL)));

    if (verbose)
        std::cout << "BFS cycles: " << S << " sources, batch " << max_batch
                  << ", depth limit " << max_depth << "\n";

    VectorType<int> all_v1, all_v2, all_v3;
    VectorType<float> all_strength;

    for (int b_start = 0; b_start < S; b_start += max_batch)
    {
        const int B = std::min(max_batch, S - b_start);
        const long long BN = (long long)B * N;
        const int* b_src = src_all_ptr + b_start;
        const int* b_dst = dst_all_ptr + b_start;
        const float* b_rep_costs = rep_costs_all_ptr + b_start;

        // dist[s * N + node], parent[s * N + node], min_cost[s * N + node]
        VectorType<int> dist((size_t)BN, N);
        VectorType<int> parent((size_t)BN, -1);
        VectorType<float> min_cost((size_t)BN, FLT_MAX);
        int* d = thrust::raw_pointer_cast(dist.data());
        int* p = thrust::raw_pointer_cast(parent.data());
        float* mc = thrust::raw_pointer_cast(min_cost.data());

        // Source init
        {
            const int n = N;
            thrust::for_each(
                thrust::make_counting_iterator(0),
                thrust::make_counting_iterator(B),
                [d, p, mc, b_src, n] CC_HOST_DEVICE (int s) {
                    const long long off = (long long)s * n + b_src[s];
                    d[off] = 0;
                    p[off] = b_src[s];
                    mc[off] = FLT_MAX;
                });
        }

        // Level-synchronous BFS
        const int depth_lim = std::min(max_depth, N);
        for (int level = 0; level < depth_lim; ++level)
        {
            const int n = N;
            thrust::for_each(
                thrust::make_counting_iterator<long long>(0),
                thrust::make_counting_iterator<long long>(BN),
                [d, p, mc, pos_off, pos_h, pos_c, n, level]
                CC_HOST_DEVICE (long long idx) {
                    const int s = (int)(idx / n);
                    const int node = (int)(idx % n);
                    const long long off = (long long)s * n;
                    if (d[off + node] != level) return;
                    for (int e = pos_off[node]; e < pos_off[node + 1]; ++e)
                    {
                        const int h = pos_h[e];
                        if (d[off + h] > level + 1)
                        {
                            d[off + h] = level + 1;
                            p[off + h] = node;
                            const float abs_c = pos_c[e] < 0 ? -pos_c[e] : pos_c[e];
                            mc[off + h] = (level == 0) ? abs_c
                                : thrust::min(mc[off + node], abs_c);
                        }
                    }
                });

            // Early exit: all targets reached?
            {
                const int n = N;
                int unreached = thrust::count_if(
                    thrust::make_counting_iterator(0),
                    thrust::make_counting_iterator(B),
                    [d, b_dst, n] CC_HOST_DEVICE (int s) {
                        return d[(long long)s * n + b_dst[s]] >= n;
                    });
                if (unreached == 0) break;
            }

            // Early exit: frontier empty?
            {
                const int next = level + 1;
                int frontier = thrust::count_if(
                    thrust::make_counting_iterator<long long>(0),
                    thrust::make_counting_iterator<long long>(BN),
                    [d, next] CC_HOST_DEVICE (long long idx) {
                        return d[idx] == next;
                    });
                if (frontier == 0) break;
            }
        }

        // Count triangles: dist - 1 fan triangles per valid path (dist >= 5).
        VectorType<int> counts(B);
        {
            const int n = N;
            int* cnt = thrust::raw_pointer_cast(counts.data());
            thrust::for_each(
                thrust::make_counting_iterator(0),
                thrust::make_counting_iterator(B),
                [cnt, d, b_dst, n] CC_HOST_DEVICE (int s) {
                    const int dv = d[(long long)s * n + b_dst[s]];
                    cnt[s] = (dv < n && dv >= 5) ? dv - 1 : 0;
                });
        }

        const int total_tri = thrust::reduce(counts.begin(), counts.end());
        if (total_tri == 0)
            continue;

        VectorType<int> write_off(B);
        thrust::exclusive_scan(counts.begin(), counts.end(), write_off.begin());

        VectorType<int> tri_v1(total_tri), tri_v2(total_tri), tri_v3(total_tri);
        VectorType<float> tri_str(total_tri);
        int* tv1 = thrust::raw_pointer_cast(tri_v1.data());
        int* tv2 = thrust::raw_pointer_cast(tri_v2.data());
        int* tv3 = thrust::raw_pointer_cast(tri_v3.data());
        float* tstr = thrust::raw_pointer_cast(tri_str.data());
        const int* cnt = thrust::raw_pointer_cast(counts.data());

        // Compute per-source strength: min(|rep_cost|, min_cost along path to dst).
        VectorType<float> src_strength(B);
        float* ss = thrust::raw_pointer_cast(src_strength.data());
        {
            const int n = N;
            thrust::for_each(
                thrust::make_counting_iterator(0),
                thrust::make_counting_iterator(B),
                [ss, mc, b_dst, b_rep_costs, n, cnt] CC_HOST_DEVICE (int s) {
                    if (cnt[s] == 0) { ss[s] = 0; return; }
                    const float abs_rep = b_rep_costs[s] < 0 ? -b_rep_costs[s] : b_rep_costs[s];
                    ss[s] = thrust::min(abs_rep, mc[(long long)s * n + b_dst[s]]);
                });
        }

        // Backward trace: emit fan triangles (src, prev, cur) for each step.
        VectorType<int> cur(B);
        VectorType<int> wi(B);
        int* cur_ptr = thrust::raw_pointer_cast(cur.data());
        int* wi_ptr = thrust::raw_pointer_cast(wi.data());

        thrust::copy(write_off.begin(), write_off.end(), wi.begin());
        {
            thrust::for_each(
                thrust::make_counting_iterator(0),
                thrust::make_counting_iterator(B),
                [cur_ptr, cnt, b_dst] CC_HOST_DEVICE (int s) {
                    cur_ptr[s] = (cnt[s] > 0) ? b_dst[s] : -1;
                });
        }

        for (int hop = 0; hop < depth_lim; ++hop)
        {
            int active = thrust::count_if(
                thrust::make_counting_iterator(0),
                thrust::make_counting_iterator(B),
                [cur_ptr] CC_HOST_DEVICE (int s) {
                    return cur_ptr[s] >= 0;
                });
            if (active == 0) break;

            const int n = N;
            thrust::for_each(
                thrust::make_counting_iterator(0),
                thrust::make_counting_iterator(B),
                [cur_ptr, p, tv1, tv2, tv3, tstr, ss, wi_ptr, b_src, n]
                CC_HOST_DEVICE (int s) {
                    const int c = cur_ptr[s];
                    if (c < 0) return;
                    const int prev = p[(long long)s * n + c];
                    if (prev == b_src[s])
                    {
                        cur_ptr[s] = -1;
                        return;
                    }
                    const int idx = wi_ptr[s];
                    qd_detail::write_sorted_triangle(
                        b_src[s], prev, c, tv1, tv2, tv3, idx);
                    tstr[idx] = ss[s];
                    ++wi_ptr[s];
                    cur_ptr[s] = prev;
                });
        }

        // Append batch results.
        const size_t old = all_v1.size();
        all_v1.resize(old + total_tri);
        all_v2.resize(old + total_tri);
        all_v3.resize(old + total_tri);
        all_strength.resize(old + total_tri);
        thrust::copy(tri_v1.begin(), tri_v1.end(), all_v1.begin() + old);
        thrust::copy(tri_v2.begin(), tri_v2.end(), all_v2.begin() + old);
        thrust::copy(tri_v3.begin(), tri_v3.end(), all_v3.begin() + old);
        thrust::copy(tri_str.begin(), tri_str.end(), all_strength.begin() + old);
    }

    if (all_v1.size() == 0)
        return {VectorType<int>(), VectorType<int>(), VectorType<int>(), VectorType<float>()};

    deduplicate_triangles<VectorType>(all_v1, all_v2, all_v3, all_strength);

    if (verbose)
        std::cout << "BFS cycles: " << all_v1.size()
                  << " triangles after deduplication\n";

    return {std::move(all_v1), std::move(all_v2), std::move(all_v3), std::move(all_strength)};
}

// Explicit instantiation declarations.
extern template
std::tuple<thrust::host_vector<int>, thrust::host_vector<int>, thrust::host_vector<int>, thrust::host_vector<float>>
find_conflicted_cycles_bfs<thrust::host_vector>(
    const thrust::host_vector<int>&, const thrust::host_vector<int>&,
    const thrust::host_vector<int>&, const thrust::host_vector<int>&,
    const thrust::host_vector<float>&, const thrust::host_vector<float>&,
    int, int, bool);

extern template
std::tuple<thrust::device_vector<int>, thrust::device_vector<int>, thrust::device_vector<int>, thrust::device_vector<float>>
find_conflicted_cycles_bfs<thrust::device_vector>(
    const thrust::device_vector<int>&, const thrust::device_vector<int>&,
    const thrust::device_vector<int>&, const thrust::device_vector<int>&,
    const thrust::device_vector<float>&, const thrust::device_vector<float>&,
    int, int, bool);
