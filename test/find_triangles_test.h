#pragma once

#include "find_triangles.h"
#include "graph.h"
#include "random_graph.h"
#include "test.h"
#include <vector>
#include <tuple>
#include <set>
#include <algorithm>

// Helper: build a symmetric positive graph in CSR format from single-orientation edges.
// Returns (offsets, col_ids) vectors suitable for find_triangles.
template<template<typename> class VectorType>
std::pair<VectorType<int>, VectorType<int>>
build_positive_csr(int num_nodes,
                   const std::vector<int>& tails,
                   const std::vector<int>& heads)
{
    std::vector<float> dummy_costs(tails.size(), 1.0f);
    Graph<VectorType> g(num_nodes,
                        tails.begin(), tails.end(),
                        heads.begin(), heads.end(),
                        dummy_costs.begin(), dummy_costs.end());

    VectorType<int> offsets = g.compute_node_offsets();
    VectorType<int> col_ids = g.get_heads();
    return {std::move(offsets), std::move(col_ids)};
}

//       (+)
//   +-------------------------+
//   |                         v
// +---+  (-)   +---+  (+)   +---+
// | 0 | .....> | 1 | -----> | 2 |
// +---+        +---+        +---+
//
// Repulsive edge 0-1, positive edges 0-2 and 1-2.
// Common neighbour 2 -> one triangle (0,1,2).
template<template<typename> class VectorType>
void test_single_triangle()
{
    // Positive graph: 0-2, 1-2
    auto [offsets, col_ids] = build_positive_csr<VectorType>(3, {0, 1}, {2, 2});

    // Repulsive edge: 0-1
    VectorType<int> rep_tails(1); rep_tails[0] = 0;
    VectorType<int> rep_heads(1); rep_heads[0] = 1;

    auto [v1, v2, v3] = find_triangles<VectorType>(rep_tails, rep_heads, offsets, col_ids);

    test(v1.size() == 1, "single triangle: should find 1 triangle");
    test(v1[0] == 0, "single triangle: v1 == 0");
    test(v2[0] == 1, "single triangle: v2 == 1");
    test(v3[0] == 2, "single triangle: v3 == 2");
}

// +------+  (-)   +---+  (+)   +---+
// |  0   | .....> | 1 | -----> | 3 |
// +------+        +---+        +---+
//   |
//   | (+)
//   v
// +------+
// |  2   |
// +------+
//
// Repulsive edge 0-1, but 0 and 1 share no neighbours. 0 triangles.
template<template<typename> class VectorType>
void test_no_common_neighbours()
{
    // Positive graph: 0-2, 1-3 (nodes 0 and 1 share no neighbours)
    auto [offsets, col_ids] = build_positive_csr<VectorType>(4, {0, 1}, {2, 3});

    VectorType<int> rep_tails(1); rep_tails[0] = 0;
    VectorType<int> rep_heads(1); rep_heads[0] = 1;

    auto [v1, v2, v3] = find_triangles<VectorType>(rep_tails, rep_heads, offsets, col_ids);

    test(v1.size() == 0, "no common neighbours: should find 0 triangles");
}

//          (+)
//   +----------------------------+
//   |                            v
// +------+  (-)   +---+  (+)   +---+
// |  0   | .....> | 1 | -----> | 2 |
// +------+        +---+        +---+
//   |               |
//   | (+)           | (+)
//   v               v
// +------+---------+
// |  3   |
// +------+
//
// Repulsive edge 0-1, common neighbours 2 and 3. Two triangles: (0,1,2) and (0,1,3).
template<template<typename> class VectorType>
void test_multiple_triangles_one_edge()
{
    // Positive graph: 0-2, 0-3, 1-2, 1-3
    auto [offsets, col_ids] = build_positive_csr<VectorType>(4, {0, 0, 1, 1}, {2, 3, 2, 3});

    VectorType<int> rep_tails(1); rep_tails[0] = 0;
    VectorType<int> rep_heads(1); rep_heads[0] = 1;

    auto [v1, v2, v3] = find_triangles<VectorType>(rep_tails, rep_heads, offsets, col_ids);

    test(v1.size() == 2, "multiple triangles one edge: should find 2 triangles");

    // Triangles should be (0,1,2) and (0,1,3), sorted output order depends on neighbour order
    // Both have min=0, mid=1, so they differ only in v3
    test(v1[0] == 0 && v2[0] == 1, "first triangle: (0,1,?)");
    test(v1[1] == 0 && v2[1] == 1, "second triangle: (0,1,?)");
    // v3 values should be 2 and 3 (in sorted adjacency order)
    test(v3[0] == 2, "first triangle v3 == 2");
    test(v3[1] == 3, "second triangle v3 == 3");
}

//               (+)
//        +----------------------------+
//        |                            v
//      +------+  (-)   +---+  (+)   +---+
//   +- |  0   | .....> | 1 | -----> | 2 |
//   |  +------+        +---+        +---+
//   |    :
//   |    : (-)
//   |    v
//   |  +------+  (+)   +---+
//   |  |  3   | -----> | 4 |
//   |  +------+        +---+
//   |   (+)              ^
//   +--------------------+
//
// Repulsive edges 0-1 and 0-3. Two triangles: (0,1,2) and (0,3,4).
template<template<typename> class VectorType>
void test_multiple_repulsive_edges()
{
    // 5-node graph: positive edges 0-2, 1-2, 3-4, 0-4
    // Repulsive edge 0-1 shares neighbour 2 → triangle (0,1,2)
    // Repulsive edge 0-3 shares neighbour 4 → triangle (0,3,4)
    auto [offsets, col_ids] = build_positive_csr<VectorType>(5, {0, 1, 3, 0}, {2, 2, 4, 4});

    VectorType<int> rep_tails(2); rep_tails[0] = 0; rep_tails[1] = 0;
    VectorType<int> rep_heads(2); rep_heads[0] = 1; rep_heads[1] = 3;

    auto [v1, v2, v3] = find_triangles<VectorType>(rep_tails, rep_heads, offsets, col_ids);

    test(v1.size() == 2, "multiple repulsive edges: should find 2 triangles");

    // First triangle from edge (0,1): common neighbour 2 → (0,1,2)
    test(v1[0] == 0 && v2[0] == 1 && v3[0] == 2,
         "first triangle should be (0,1,2)");

    // Second triangle from edge (0,3): common neighbour 4 → (0,3,4)
    test(v1[1] == 0 && v2[1] == 3 && v3[1] == 4,
         "second triangle should be (0,3,4)");
}

// +---+  (+)   +---+  (+)   +---+
// | 0 | -----> | 2 | <----- | 1 |
// +---+        +---+        +---+
//
// No repulsive edges -> no triangles.
template<template<typename> class VectorType>
void test_empty_input()
{
    // Build some positive graph (doesn't matter)
    auto [offsets, col_ids] = build_positive_csr<VectorType>(3, {0, 1}, {2, 2});

    VectorType<int> rep_tails;
    VectorType<int> rep_heads;

    auto [v1, v2, v3] = find_triangles<VectorType>(rep_tails, rep_heads, offsets, col_ids);

    test(v1.size() == 0, "empty input: should find 0 triangles");
    test(v2.size() == 0, "empty input: v2 empty");
    test(v3.size() == 0, "empty input: v3 empty");
}

// Brute-force find all triangles: for each repulsive edge (u,v), find all
// nodes w such that (u,w) and (v,w) are both positive edges.
// Returns sorted triples (v1 < v2 < v3), one per (repulsive edge, common neighbour)
// pair — i.e. duplicates arise when a triangle contains multiple repulsive edges.
inline std::vector<std::tuple<int,int,int>> brute_force_triangles(
    const int num_nodes,
    const std::vector<int>& rep_tails, const std::vector<int>& rep_heads,
    const std::vector<int>& pos_tails, const std::vector<int>& pos_heads)
{
    std::set<std::pair<int,int>> pos_adj;
    for (size_t i = 0; i < pos_tails.size(); ++i)
    {
        pos_adj.insert({pos_tails[i], pos_heads[i]});
        pos_adj.insert({pos_heads[i], pos_tails[i]});
    }

    std::vector<std::tuple<int,int,int>> result;
    for (size_t i = 0; i < rep_tails.size(); ++i)
    {
        const int u = rep_tails[i];
        const int v = rep_heads[i];
        for (int w = 0; w < num_nodes; ++w)
        {
            if (w == u || w == v) continue;
            if (pos_adj.count({u, w}) && pos_adj.count({v, w}))
            {
                const int min_v = std::min({u, v, w});
                const int max_v = std::max({u, v, w});
                const int mid_v = u + v + w - min_v - max_v;
                result.push_back({min_v, mid_v, max_v});
            }
        }
    }

    std::sort(result.begin(), result.end());
    return result;
}

// Random graphs: generate random edges with random costs, split into
// positive/repulsive by sign, run find_triangles, and verify against
// brute-force enumeration of all triangles.
template<template<typename> class VectorType>
void test_random_graphs()
{
    const int nodes_list[] = {5, 10, 20, 40};
    const double density_list[] = {0.1, 0.3, 0.5, 0.8};
    const int num_seeds = 3;
    unsigned seed = 0;

    for (int num_nodes : nodes_list)
    for (double edge_prob : density_list)
    for (int s = 0; s < num_seeds; ++s, ++seed)
    {
        auto rg = generate_random_graph(num_nodes, edge_prob, seed);

        // Split into positive (cost >= 0) and repulsive (cost < 0) edges
        std::vector<int> pos_tails, pos_heads, neg_tails, neg_heads;
        for (size_t i = 0; i < rg.tails.size(); ++i)
        {
            if (rg.costs[i] < 0.0f)
            {
                neg_tails.push_back(rg.tails[i]);
                neg_heads.push_back(rg.heads[i]);
            }
            else
            {
                pos_tails.push_back(rg.tails[i]);
                pos_heads.push_back(rg.heads[i]);
            }
        }

        if (pos_tails.empty() || neg_tails.empty())
            continue;

        // Build positive CSR via Graph
        auto [offsets, col_ids] = build_positive_csr<VectorType>(
            rg.num_nodes, pos_tails, pos_heads);

        VectorType<int> rep_tails(neg_tails.begin(), neg_tails.end());
        VectorType<int> rep_heads(neg_heads.begin(), neg_heads.end());

        auto [v1, v2, v3] = find_triangles<VectorType>(
            rep_tails, rep_heads, offsets, col_ids);

        // Copy results to host for comparison
        thrust::host_vector<int> h_v1(v1), h_v2(v2), h_v3(v3);

        std::vector<std::tuple<int,int,int>> found;
        for (size_t i = 0; i < h_v1.size(); ++i)
            found.push_back({h_v1[i], h_v2[i], h_v3[i]});
        std::sort(found.begin(), found.end());

        auto expected = brute_force_triangles(
            rg.num_nodes, neg_tails, neg_heads, pos_tails, pos_heads);

        const std::string label = "n=" + std::to_string(num_nodes)
            + " p=" + std::to_string(edge_prob) + " seed=" + std::to_string(seed);
        test(found.size() == expected.size(), label + ": triangle count mismatch");
        test(found == expected, label + ": triangle content mismatch");
    }
}

template<template<typename> class VectorType>
void run_all_find_triangles_tests()
{
    test_single_triangle<VectorType>();
    test_no_common_neighbours<VectorType>();
    test_multiple_triangles_one_edge<VectorType>();
    test_multiple_repulsive_edges<VectorType>();
    test_empty_input<VectorType>();
    test_random_graphs<VectorType>();
}
