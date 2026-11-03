#pragma once

#include <tuple>
#include <iostream>
#include <cfloat>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/for_each.h>
#include <thrust/count.h>
#include <thrust/copy.h>
#include <thrust/reduce.h>
#include <thrust/scan.h>
#include <thrust/iterator/counting_iterator.h>
#include "mst_boruvka.h"
#include "graph.h"
#include "connected_components.h"
#include "find_triangles.h"    // CC_HOST_DEVICE
#include "find_quadrangles.h"  // qd_detail::write_sorted_triangle, deduplicate_triangles

// Find conflicted cycles via MST fundamental cycles + LCA.
//
// For each repulsive edge (u,v), the unique tree path u->v in the maximum
// spanning tree forms a fundamental cycle. Uses LCA (Lowest Common Ancestor)
// walks to find paths — no dense B×N arrays, no batching, no dedup needed.
//
// Algorithm:
//   1. Compute maximum spanning tree of positive subgraph
//   2. Root the MST forest (BFS from component representatives)
//   3. For each repulsive edge, LCA walk to find path length and min-cost
//   4. Fan-triangulate paths of length >= 5 (cycle length >= 6)
//
// Complexity: O(E log V) for MST + O(N) for rooting + O(S * max_depth) for
// LCA walks. All S repulsive edges processed in parallel, no batching.
template<template<typename> class VectorType>
std::tuple<VectorType<int>, VectorType<int>, VectorType<int>, VectorType<float>>
find_conflicted_cycles_mst(
    const VectorType<int>& rep_edge_tails,
    const VectorType<int>& rep_edge_heads,
    const VectorType<float>& rep_edge_costs,
    const VectorType<int>& pos_edge_tails,
    const VectorType<int>& pos_edge_heads,
    const VectorType<float>& pos_edge_costs,
    const int num_nodes,
    const int max_cycle_length,
    const bool verbose = true)
{
    if (pos_edge_tails.size() == 0 || rep_edge_tails.size() == 0)
        return {VectorType<int>(), VectorType<int>(), VectorType<int>(), VectorType<float>()};

    const int N = num_nodes;
    const int max_depth = (max_cycle_length > 0) ? max_cycle_length - 1 : N;

    if (max_depth < 5)
        return {VectorType<int>(), VectorType<int>(), VectorType<int>(), VectorType<float>()};

    // 1. Compute maximum spanning tree of positive subgraph.
    auto [mst_tails, mst_heads, mst_costs] =
        MST_boruvka::maximum_spanning_tree<VectorType>(
            pos_edge_tails, pos_edge_heads, pos_edge_costs);

    if (mst_tails.size() == 0)
        return {VectorType<int>(), VectorType<int>(), VectorType<int>(), VectorType<float>()};

    if (verbose)
        std::cout << "MST cycles: " << mst_tails.size()
                  << " MST edges from " << pos_edge_tails.size() / 2
                  << " positive edges\n";

    // 2. Build symmetric CSR of MST.
    Graph<VectorType> mst_graph(N,
        std::move(mst_tails), std::move(mst_heads), std::move(mst_costs));
    VectorType<int> mst_offsets = mst_graph.compute_node_offsets();
    const int* mst_off = thrust::raw_pointer_cast(mst_offsets.data());
    const int* mst_h = thrust::raw_pointer_cast(mst_graph.get_heads().data());
    const float* mst_c = thrust::raw_pointer_cast(mst_graph.get_costs().data());

    // 3. Root the MST forest.
    // 3a. CC to find component representatives (= roots).
    VectorType<int> mst_comp = connected_components::compute_cc<VectorType>(
        N, mst_graph.get_tails(), mst_graph.get_heads());

    // 3b. Initialize roots: comp[node] == node means it's a root.
    VectorType<int> depth(N, -1);
    VectorType<int> par(N, -1);
    VectorType<float> mc_to_par(N, FLT_MAX);
    {
        int* dep = thrust::raw_pointer_cast(depth.data());
        int* par_p = thrust::raw_pointer_cast(par.data());
        const int* comp_p = thrust::raw_pointer_cast(mst_comp.data());
        thrust::for_each(
            thrust::make_counting_iterator(0),
            thrust::make_counting_iterator(N),
            [dep, par_p, comp_p] CC_HOST_DEVICE (int node) {
                if (comp_p[node] == node) {
                    dep[node] = 0;
                    par_p[node] = node;
                }
            });
    }

    // 3c. BFS from all roots to fill depth/parent/mc_to_par.
    // On a tree, each node has exactly one unvisited neighbor per level,
    // so no races and no dedup needed.
    {
        int* dep = thrust::raw_pointer_cast(depth.data());
        int* par_p = thrust::raw_pointer_cast(par.data());
        float* mcp = thrust::raw_pointer_cast(mc_to_par.data());

        for (int level = 0; level < N; ++level)
        {
            const int frontier = thrust::count(depth.begin(), depth.end(), level);
            if (frontier == 0) break;

            thrust::for_each(
                thrust::make_counting_iterator(0),
                thrust::make_counting_iterator(N),
                [dep, par_p, mcp, mst_off, mst_h, mst_c, level]
                CC_HOST_DEVICE (int node) {
                    if (dep[node] != level) return;
                    for (int e = mst_off[node]; e < mst_off[node + 1]; ++e)
                    {
                        const int nbr = mst_h[e];
                        if (dep[nbr] == -1)
                        {
                            dep[nbr] = level + 1;
                            par_p[nbr] = node;
                            const float c = mst_c[e];
                            mcp[nbr] = c < 0 ? -c : c;
                        }
                    }
                });
        }
    }

    // 4. CC prefilter: keep only repulsive edges within the same MST component.
    const int num_rep = (int)rep_edge_tails.size();
    const int* comp_ptr = thrust::raw_pointer_cast(mst_comp.data());
    const int* rep_t = thrust::raw_pointer_cast(rep_edge_tails.data());
    const int* rep_h_p = thrust::raw_pointer_cast(rep_edge_heads.data());

    auto is_reachable = [comp_ptr, rep_t, rep_h_p] CC_HOST_DEVICE (int i) {
        return comp_ptr[rep_t[i]] == comp_ptr[rep_h_p[i]];
    };

    const int S = thrust::count_if(
        thrust::make_counting_iterator(0),
        thrust::make_counting_iterator(num_rep),
        is_reachable);

    if (S == 0)
        return {VectorType<int>(), VectorType<int>(), VectorType<int>(), VectorType<float>()};

    VectorType<int> src_all(S), dst_all(S);
    VectorType<float> rep_costs_filt(S);
    thrust::copy_if(rep_edge_tails.begin(), rep_edge_tails.end(),
                    thrust::make_counting_iterator(0), src_all.begin(), is_reachable);
    thrust::copy_if(rep_edge_heads.begin(), rep_edge_heads.end(),
                    thrust::make_counting_iterator(0), dst_all.begin(), is_reachable);
    thrust::copy_if(rep_edge_costs.begin(), rep_edge_costs.end(),
                    thrust::make_counting_iterator(0), rep_costs_filt.begin(), is_reachable);

    if (verbose)
        std::cout << "MST cycles (LCA): " << S << "/" << num_rep
                  << " repulsive edges reachable, depth limit " << max_depth << "\n";

    // 5. Pass 1: Count triangles per repulsive edge via LCA walk.
    VectorType<int> tri_counts(S);
    VectorType<float> edge_strength(S);
    {
        const int* src_p = thrust::raw_pointer_cast(src_all.data());
        const int* dst_p = thrust::raw_pointer_cast(dst_all.data());
        const float* rc_p = thrust::raw_pointer_cast(rep_costs_filt.data());
        int* tc = thrust::raw_pointer_cast(tri_counts.data());
        float* es = thrust::raw_pointer_cast(edge_strength.data());
        const int* dep = thrust::raw_pointer_cast(depth.data());
        const int* par_p = thrust::raw_pointer_cast(par.data());
        const float* mcp = thrust::raw_pointer_cast(mc_to_par.data());
        const int md = max_depth;

        thrust::for_each(
            thrust::make_counting_iterator(0),
            thrust::make_counting_iterator(S),
            [src_p, dst_p, rc_p, tc, es, dep, par_p, mcp, md]
            CC_HOST_DEVICE (int i) {
                int a = src_p[i], b = dst_p[i];
                float path_min = FLT_MAX;
                int steps = 0;

                while (a != b)
                {
                    if (steps > md) break;
                    if (dep[a] > dep[b]) {
                        path_min = thrust::min(path_min, mcp[a]);
                        a = par_p[a];
                    } else {
                        path_min = thrust::min(path_min, mcp[b]);
                        b = par_p[b];
                    }
                    ++steps;
                }

                if (a != b || steps < 5 || steps > md) {
                    tc[i] = 0;
                    es[i] = 0;
                } else {
                    tc[i] = steps - 1;
                    const float abs_rep = rc_p[i] < 0 ? -rc_p[i] : rc_p[i];
                    es[i] = thrust::min(abs_rep, path_min);
                }
            });
    }

    const int total_tri = thrust::reduce(tri_counts.begin(), tri_counts.end());
    if (total_tri == 0)
        return {VectorType<int>(), VectorType<int>(), VectorType<int>(), VectorType<float>()};

    if (verbose)
        std::cout << "MST cycles (LCA): " << total_tri << " triangles from "
                  << S << " repulsive edges\n";

    // 6. Exclusive scan for write offsets.
    VectorType<int> write_off(S);
    thrust::exclusive_scan(tri_counts.begin(), tri_counts.end(), write_off.begin());

    // 7. Pass 2: Write fan triangles via LCA walk (re-walk to collect path).
    VectorType<int> all_v1(total_tri), all_v2(total_tri), all_v3(total_tri);
    VectorType<float> all_str(total_tri);
    {
        const int* src_p = thrust::raw_pointer_cast(src_all.data());
        const int* dst_p = thrust::raw_pointer_cast(dst_all.data());
        const int* tc = thrust::raw_pointer_cast(tri_counts.data());
        const float* es = thrust::raw_pointer_cast(edge_strength.data());
        const int* wo = thrust::raw_pointer_cast(write_off.data());
        const int* dep = thrust::raw_pointer_cast(depth.data());
        const int* par_p = thrust::raw_pointer_cast(par.data());
        int* tv1 = thrust::raw_pointer_cast(all_v1.data());
        int* tv2 = thrust::raw_pointer_cast(all_v2.data());
        int* tv3 = thrust::raw_pointer_cast(all_v3.data());
        float* ts = thrust::raw_pointer_cast(all_str.data());

        thrust::for_each(
            thrust::make_counting_iterator(0),
            thrust::make_counting_iterator(S),
            [src_p, dst_p, tc, es, wo, dep, par_p, tv1, tv2, tv3, ts]
            CC_HOST_DEVICE (int i) {
                if (tc[i] == 0) return;

                const int u = src_p[i];
                const int v = dst_p[i];

                // Find LCA
                int a = u, b = v;
                while (a != b) {
                    if (dep[a] > dep[b]) a = par_p[a];
                    else b = par_p[b];
                }
                const int lca = a;

                // Collect path u → ... → lca → ... → v
                // path_u: u, par(u), ..., lca
                // path_v: v, par(v), ..., lca (will be reversed)
                constexpr int MAX_HALF = 16;
                int path_u[MAX_HALF], path_v[MAX_HALF];
                int len_u = 0, len_v = 0;

                a = u;
                while (a != lca) { path_u[len_u++] = a; a = par_p[a]; }
                path_u[len_u++] = lca;

                b = v;
                while (b != lca) { path_v[len_v++] = b; b = par_p[b]; }
                // path_v does NOT include lca (already in path_u)

                // Full path: path_u[0..len_u-1], path_v[len_v-1..0]
                // Total nodes = len_u + len_v
                // Fan triangles from path[0]=u:
                //   (u, path[j], path[j+1]) for j = 1..total_nodes-2
                const float strength = es[i];
                int base = wo[i];

                // Triangles within path_u: (u, path_u[j], path_u[j+1])
                for (int j = 1; j + 1 < len_u; ++j) {
                    qd_detail::write_sorted_triangle(
                        u, path_u[j], path_u[j + 1], tv1, tv2, tv3, base);
                    ts[base++] = strength;
                }

                // Transition: (u, lca=path_u[len_u-1], path_v[len_v-1])
                if (len_v > 0 && len_u >= 2) {
                    qd_detail::write_sorted_triangle(
                        u, path_u[len_u - 1], path_v[len_v - 1],
                        tv1, tv2, tv3, base);
                    ts[base++] = strength;
                }
                // Triangles within reversed path_v:
                // (u, path_v[j], path_v[j-1]) for j = len_v-1 down to 1
                // When len_u == 1 (u == lca), no transition was emitted,
                // and this loop handles all path_v pairs.
                for (int j = len_v - 1; j >= 1; --j) {
                    qd_detail::write_sorted_triangle(
                        u, path_v[j], path_v[j - 1], tv1, tv2, tv3, base);
                    ts[base++] = strength;
                }
            });
    }

    // 8. Deduplicate triangles.
    deduplicate_triangles<VectorType>(all_v1, all_v2, all_v3, all_str);

    if (verbose)
        std::cout << "MST cycles (LCA): " << all_v1.size()
                  << " triangles after dedup\n";

    return {std::move(all_v1), std::move(all_v2), std::move(all_v3), std::move(all_str)};
}

// Explicit instantiation declarations.
extern template
std::tuple<thrust::host_vector<int>, thrust::host_vector<int>, thrust::host_vector<int>, thrust::host_vector<float>>
find_conflicted_cycles_mst<thrust::host_vector>(
    const thrust::host_vector<int>&, const thrust::host_vector<int>&, const thrust::host_vector<float>&,
    const thrust::host_vector<int>&, const thrust::host_vector<int>&, const thrust::host_vector<float>&,
    int, int, bool);

extern template
std::tuple<thrust::device_vector<int>, thrust::device_vector<int>, thrust::device_vector<int>, thrust::device_vector<float>>
find_conflicted_cycles_mst<thrust::device_vector>(
    const thrust::device_vector<int>&, const thrust::device_vector<int>&, const thrust::device_vector<float>&,
    const thrust::device_vector<int>&, const thrust::device_vector<int>&, const thrust::device_vector<float>&,
    int, int, bool);