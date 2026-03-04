#pragma once

#include "find_cycles_bfs.h"
#include "graph.h"
#include "test.h"
#include "random_graph.h"
#include <vector>
#include <tuple>
#include <set>
#include <algorithm>
#include <queue>
#include <random>

// Build a symmetric positive graph in CSR format from single-orientation edges.
// Returns (offsets, col_ids, costs).
template<template<typename> class VectorType>
std::tuple<VectorType<int>, VectorType<int>, VectorType<float>>
build_pos_csr(int num_nodes,
              const std::vector<int>& tails,
              const std::vector<int>& heads,
              const std::vector<float>& edge_costs = {})
{
    std::vector<float> costs_in = edge_costs;
    if (costs_in.empty())
        costs_in.assign(tails.size(), 1.0f);
    Graph<VectorType> g(num_nodes,
                        tails.begin(), tails.end(),
                        heads.begin(), heads.end(),
                        costs_in.begin(), costs_in.end());

    VectorType<int> offsets = g.compute_node_offsets();
    VectorType<int> col_ids = g.get_heads();
    VectorType<float> csr_costs(g.get_costs().begin(), g.get_costs().end());
    return {std::move(offsets), std::move(col_ids), std::move(csr_costs)};
}

// 6-node ring with one repulsive edge.
//
//     +1        +1        +1        +1        +1
// 0 -----> 5 -----> 4 -----> 3 -----> 2 -----> 1
// |                                             |
// +-------------------- -1 --------------------+
//
// Repulsive edge (0,1). Positive path: 0->5->4->3->2->1 (length 5).
// Cycle length 6. Fan from 0: (0,1,2), (0,2,3), (0,3,4), (0,4,5).
template<template<typename> class VectorType>
void test_bfs_6_cycle()
{
    // Positive edges (single orientation): 1-2, 2-3, 3-4, 4-5, 0-5
    auto [offsets, col_ids, costs] = build_pos_csr<VectorType>(
        6, {1, 2, 3, 4, 0}, {2, 3, 4, 5, 5});

    VectorType<int> rep_tails(1); rep_tails[0] = 0;
    VectorType<int> rep_heads(1); rep_heads[0] = 1;
    VectorType<float> rep_costs(1); rep_costs[0] = -1.0f;

    auto [v1, v2, v3, strength] = find_conflicted_cycles_bfs<VectorType>(
        rep_tails, rep_heads, offsets, col_ids, costs, rep_costs, 6, 0, false);

    test(v1.size() == 4, "6-cycle: should find 4 triangles, got " + std::to_string(v1.size()));

    // Copy to host for checking
    thrust::host_vector<int> h1(v1), h2(v2), h3(v3);
    std::vector<std::tuple<int,int,int>> found;
    for (size_t i = 0; i < h1.size(); ++i)
        found.push_back({h1[i], h2[i], h3[i]});
    std::sort(found.begin(), found.end());

    std::vector<std::tuple<int,int,int>> expected = {
        {0, 1, 2}, {0, 2, 3}, {0, 3, 4}, {0, 4, 5}};
    test(found == expected, "6-cycle: triangle content mismatch");
}

// 7-node ring with one repulsive edge.
// Repulsive (0,1). Positive path: 0->6->5->4->3->2->1 (length 6).
// Cycle length 7. Fan from 0: 5 triangles.
template<template<typename> class VectorType>
void test_bfs_7_cycle()
{
    // Positive edges: 1-2, 2-3, 3-4, 4-5, 5-6, 0-6
    auto [offsets, col_ids, costs] = build_pos_csr<VectorType>(
        7, {1, 2, 3, 4, 5, 0}, {2, 3, 4, 5, 6, 6});

    VectorType<int> rep_tails(1); rep_tails[0] = 0;
    VectorType<int> rep_heads(1); rep_heads[0] = 1;
    VectorType<float> rep_costs(1); rep_costs[0] = -1.0f;

    auto [v1, v2, v3, strength] = find_conflicted_cycles_bfs<VectorType>(
        rep_tails, rep_heads, offsets, col_ids, costs, rep_costs, 7, 0, false);

    test(v1.size() == 5, "7-cycle: should find 5 triangles, got " + std::to_string(v1.size()));

    thrust::host_vector<int> h1(v1), h2(v2), h3(v3);
    std::vector<std::tuple<int,int,int>> found;
    for (size_t i = 0; i < h1.size(); ++i)
        found.push_back({h1[i], h2[i], h3[i]});
    std::sort(found.begin(), found.end());

    std::vector<std::tuple<int,int,int>> expected = {
        {0, 1, 2}, {0, 2, 3}, {0, 3, 4}, {0, 4, 5}, {0, 5, 6}};
    test(found == expected, "7-cycle: triangle content mismatch");
}

// Repulsive edge endpoints in different positive-graph components.
//
// Component A: 0-1-2    Component B: 3-4-5
// Repulsive edge (0, 3) has no positive path. Returns 0 triangles.
template<template<typename> class VectorType>
void test_bfs_no_path()
{
    // Positive edges: 0-1, 1-2, 3-4, 4-5
    auto [offsets, col_ids, costs] = build_pos_csr<VectorType>(
        6, {0, 1, 3, 4}, {1, 2, 4, 5});

    VectorType<int> rep_tails(1); rep_tails[0] = 0;
    VectorType<int> rep_heads(1); rep_heads[0] = 3;
    VectorType<float> rep_costs(1); rep_costs[0] = -1.0f;

    auto [v1, v2, v3, strength] = find_conflicted_cycles_bfs<VectorType>(
        rep_tails, rep_heads, offsets, col_ids, costs, rep_costs, 6, 0, false);

    test(v1.size() == 0, "no path: should find 0 triangles");
}

// Repulsive edge with a common neighbour (triangle). BFS finds path of length 2
// (cycle length 3 <= 5), filtered out. Returns 0 BFS-specific triangles.
//
//       (+)
//   +-------------------------+
//   |                         v
// +---+  (-)   +---+  (+)   +---+
// | 0 | .....> | 1 | -----> | 2 |
// +---+        +---+        +---+
template<template<typename> class VectorType>
void test_bfs_short_path_filtered()
{
    auto [offsets, col_ids, costs] = build_pos_csr<VectorType>(
        3, {0, 1}, {2, 2});

    VectorType<int> rep_tails(1); rep_tails[0] = 0;
    VectorType<int> rep_heads(1); rep_heads[0] = 1;
    VectorType<float> rep_costs(1); rep_costs[0] = -1.0f;

    auto [v1, v2, v3, strength] = find_conflicted_cycles_bfs<VectorType>(
        rep_tails, rep_heads, offsets, col_ids, costs, rep_costs, 3, 0, false);

    test(v1.size() == 0, "short path: should find 0 triangles");
}

// Two repulsive edges, each forming a 6-cycle in separate rings.
// Ring 1: nodes 0-5, repulsive (0,1), positive (1-2, 2-3, 3-4, 4-5, 0-5).
// Ring 2: nodes 6-11, repulsive (6,7), positive (7-8, 8-9, 9-10, 10-11, 6-11).
// Expected: 4 + 4 = 8 triangles.
template<template<typename> class VectorType>
void test_bfs_multiple_edges()
{
    auto [offsets, col_ids, costs] = build_pos_csr<VectorType>(
        12,
        {1, 2, 3, 4, 0, 7, 8, 9, 10, 6},
        {2, 3, 4, 5, 5, 8, 9, 10, 11, 11});

    VectorType<int> rep_tails(2); rep_tails[0] = 0; rep_tails[1] = 6;
    VectorType<int> rep_heads(2); rep_heads[0] = 1; rep_heads[1] = 7;
    VectorType<float> rep_costs(2); rep_costs[0] = -1.0f; rep_costs[1] = -1.0f;

    auto [v1, v2, v3, strength] = find_conflicted_cycles_bfs<VectorType>(
        rep_tails, rep_heads, offsets, col_ids, costs, rep_costs, 12, 0, false);

    test(v1.size() == 8, "multiple edges: should find 8 triangles, got " + std::to_string(v1.size()));

    thrust::host_vector<int> h1(v1), h2(v2), h3(v3);
    std::vector<std::tuple<int,int,int>> found;
    for (size_t i = 0; i < h1.size(); ++i)
        found.push_back({h1[i], h2[i], h3[i]});
    std::sort(found.begin(), found.end());

    std::vector<std::tuple<int,int,int>> expected = {
        {0, 1, 2}, {0, 2, 3}, {0, 3, 4}, {0, 4, 5},
        {6, 7, 8}, {6, 8, 9}, {6, 9, 10}, {6, 10, 11}};
    test(found == expected, "multiple edges: triangle content mismatch");
}

// 10-node ring, but max_cycle_length=7. Path needs 9 edges (depth 9),
// but BFS bounded at depth 6. Target not reached. Returns 0 triangles.
template<template<typename> class VectorType>
void test_bfs_max_length_bound()
{
    // 10-node ring: repulsive (0,1), positive 1-2, 2-3, ..., 8-9, 0-9.
    auto [offsets, col_ids, costs] = build_pos_csr<VectorType>(
        10,
        {1, 2, 3, 4, 5, 6, 7, 8, 0},
        {2, 3, 4, 5, 6, 7, 8, 9, 9});

    VectorType<int> rep_tails(1); rep_tails[0] = 0;
    VectorType<int> rep_heads(1); rep_heads[0] = 1;
    VectorType<float> rep_costs(1); rep_costs[0] = -1.0f;

    // max_cycle_length=7 -> max_depth=6, but path is 9 edges
    auto [v1, v2, v3, strength] = find_conflicted_cycles_bfs<VectorType>(
        rep_tails, rep_heads, offsets, col_ids, costs, rep_costs, 10, 7, false);

    test(v1.size() == 0, "max length bound: should find 0 triangles");

    // Same ring with unlimited -> should find 8 triangles (dist=9, count=8)
    auto [u1, u2, u3, u_str] = find_conflicted_cycles_bfs<VectorType>(
        rep_tails, rep_heads, offsets, col_ids, costs, rep_costs, 10, 0, false);

    test(u1.size() == 8, "max length unlimited: should find 8 triangles, got " + std::to_string(u1.size()));
}

// Reference sequential BFS: for each repulsive edge (u,v), compute the
// shortest path distance from u to v in the positive subgraph.
// Returns one distance per repulsive edge (-1 if unreachable).
inline std::vector<int>
reference_bfs_distances(
    const std::vector<int>& neg_tails,
    const std::vector<int>& neg_heads,
    const std::vector<int>& csr_offsets,
    const std::vector<int>& csr_heads,
    const int num_nodes,
    const int max_cycle_length)
{
    const int max_depth = (max_cycle_length > 0) ? max_cycle_length - 1 : num_nodes;
    std::vector<int> result(neg_tails.size());

    for (size_t e = 0; e < neg_tails.size(); ++e)
    {
        const int src = neg_tails[e];
        const int dst = neg_heads[e];

        std::vector<int> dist(num_nodes, -1);
        std::queue<int> q;
        dist[src] = 0;
        q.push(src);

        while (!q.empty())
        {
            const int node = q.front();
            q.pop();
            if (dist[node] >= max_depth) continue;
            for (int i = csr_offsets[node]; i < csr_offsets[node + 1]; ++i)
            {
                const int nb = csr_heads[i];
                if (dist[nb] < 0)
                {
                    dist[nb] = dist[node] + 1;
                    q.push(nb);
                }
            }
        }

        result[e] = dist[dst];
    }

    return result;
}

// Random graphs: compare Thrust BFS triangle counts against reference distances.
template<template<typename> class VectorType>
void test_bfs_random_graphs()
{
    const int nodes_list[] = {8, 15, 25, 40};
    const double density_list[] = {0.15, 0.3, 0.5};
    const int num_seeds = 5;

    for (int num_nodes : nodes_list)
    for (double edge_prob : density_list)
    for (unsigned seed = 0; seed < (unsigned)num_seeds; ++seed)
    {
        RandomGraph rg = generate_random_graph(num_nodes, edge_prob, seed);

        // Separate positive and negative edges.
        std::vector<int> pos_tails, pos_heads, neg_tails, neg_heads;
        std::vector<float> pos_costs_vec, neg_costs_vec;
        for (size_t i = 0; i < rg.tails.size(); ++i)
        {
            if (rg.costs[i] < 0.0f)
            {
                neg_tails.push_back(rg.tails[i]);
                neg_heads.push_back(rg.heads[i]);
                neg_costs_vec.push_back(rg.costs[i]);
            }
            else
            {
                pos_tails.push_back(rg.tails[i]);
                pos_heads.push_back(rg.heads[i]);
                pos_costs_vec.push_back(rg.costs[i]);
            }
        }

        if (pos_tails.empty() || neg_tails.empty())
            continue;

        // Build CSR for positive subgraph.
        auto [offsets, col_ids, csr_costs] = build_pos_csr<VectorType>(
            num_nodes, pos_tails, pos_heads, pos_costs_vec);

        // Host-side CSR for the reference.
        thrust::host_vector<int> h_offsets(offsets);
        thrust::host_vector<int> h_col_ids(col_ids);
        std::vector<int> csr_off(h_offsets.begin(), h_offsets.end());
        std::vector<int> csr_col(h_col_ids.begin(), h_col_ids.end());

        const std::string label = "random n=" + std::to_string(num_nodes)
            + " p=" + std::to_string(edge_prob) + " seed=" + std::to_string(seed);

        // Test both unlimited and bounded cycle length.
        for (int max_cl : {0, 8})
        {
            const std::string cl_label = label
                + " max_cl=" + std::to_string(max_cl);

            // Reference: shortest path distances.
            auto ref_dists = reference_bfs_distances(
                neg_tails, neg_heads, csr_off, csr_col, num_nodes, max_cl);

            // Expected total triangle count: sum of (d-1) for each edge with d >= 5.
            int expected_total = 0;
            for (int d : ref_dists)
                if (d >= 5) expected_total += d - 1;

            // Thrust implementation.
            VectorType<int> rep_t(neg_tails.begin(), neg_tails.end());
            VectorType<int> rep_h(neg_heads.begin(), neg_heads.end());
            VectorType<float> rep_c(neg_costs_vec.begin(), neg_costs_vec.end());

            auto [v1, v2, v3, strength] = find_conflicted_cycles_bfs<VectorType>(
                rep_t, rep_h, offsets, col_ids, csr_costs, rep_c, num_nodes, max_cl, false);

            thrust::host_vector<int> h1(v1), h2(v2), h3(v3);

            // Well-formedness checks.
            for (size_t i = 0; i < h1.size(); ++i)
            {
                test(h1[i] >= 0 && h1[i] < num_nodes, cl_label + ": v1 in range");
                test(h2[i] >= 0 && h2[i] < num_nodes, cl_label + ": v2 in range");
                test(h3[i] >= 0 && h3[i] < num_nodes, cl_label + ": v3 in range");
                test(h1[i] < h2[i] && h2[i] < h3[i], cl_label + ": vertices sorted");
            }

            // No duplicates.
            std::set<std::tuple<int,int,int>> unique_tris;
            for (size_t i = 0; i < h1.size(); ++i)
                unique_tris.insert({h1[i], h2[i], h3[i]});
            test(unique_tris.size() == h1.size(),
                cl_label + ": duplicates in output ("
                + std::to_string(h1.size()) + " entries, "
                + std::to_string(unique_tris.size()) + " unique)");

            // Each triangle must contain at least one repulsive-edge endpoint as
            // the fan source. Specifically, in the fan from src, every triangle
            // has src as a vertex.
            std::set<int> rep_sources(neg_tails.begin(), neg_tails.end());
            for (int h : neg_heads) rep_sources.insert(h);
            for (size_t i = 0; i < h1.size(); ++i)
            {
                bool has_src = rep_sources.count(h1[i])
                            || rep_sources.count(h2[i])
                            || rep_sources.count(h3[i]);
                test(has_src, cl_label + ": triangle has no repulsive endpoint");
            }

            // Triangle count must match reference (before dedup, counts could
            // differ due to shared triangles across edges, but after dedup the
            // count must be <= expected_total and > 0 when expected_total > 0).
            if (expected_total == 0)
            {
                test(h1.size() == 0,
                    cl_label + ": expected 0 triangles, got "
                    + std::to_string(h1.size()));
            }
            else
            {
                test(h1.size() > 0,
                    cl_label + ": expected >0 triangles, got 0");
                test((int)h1.size() <= expected_total,
                    cl_label + ": too many triangles ("
                    + std::to_string(h1.size()) + " > "
                    + std::to_string(expected_total) + " pre-dedup)");
            }
        }
    }
}

template<template<typename> class VectorType>
void run_all_find_cycles_bfs_tests()
{
    test_bfs_6_cycle<VectorType>();
    test_bfs_7_cycle<VectorType>();
    test_bfs_no_path<VectorType>();
    test_bfs_short_path_filtered<VectorType>();
    test_bfs_multiple_edges<VectorType>();
    test_bfs_max_length_bound<VectorType>();
    test_bfs_random_graphs<VectorType>();
}
