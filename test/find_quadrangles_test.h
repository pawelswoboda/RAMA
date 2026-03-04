#pragma once

#include "find_quadrangles.h"
#include "find_triangles_test.h" // for build_positive_csr
#include "random_graph.h"
#include "test.h"
#include <vector>
#include <tuple>
#include <set>
#include <algorithm>

// Quadrangle: v1 - w - m - v2, where (v1,v2) is repulsive and
// (v1,w), (w,m), (m,v2) are positive edges.
// Decomposed into sorted triangles (v1,v2,w) and (v2,w,m).

//   +---+  (+)   +---+  (+)   +---+  (+)   +---+
//   | 0 | -----> | 1 | -----> | 2 | -----> | 3 |
//   +---+        +---+        +---+        +---+
//     |                                      ^
//     +...............(rep)..................+
//
// Repulsive edge 0-3, positive path 0-1-2-3.
// w=1: common neighbours of (1,3) = {2}. Quad 0-1-2-3, triangles (0,1,3) and (1,2,3).
template<template<typename> class VectorType>
void test_single_quadrangle()
{
    // Positive edges: 0-1, 1-2, 2-3
    auto [offsets, col_ids, costs] = build_positive_csr<VectorType>(4, {0, 1, 2}, {1, 2, 3});

    VectorType<int> rep_tails(1); rep_tails[0] = 0;
    VectorType<int> rep_heads(1); rep_heads[0] = 3;
    VectorType<float> rep_costs(1); rep_costs[0] = -1.0f;

    auto [v1, v2, v3, strength] = find_quadrangles<VectorType>(
        rep_tails, rep_heads, offsets, col_ids, costs, rep_costs);

    test(v1.size() == 2, "single quad: should find 2 triangles");
    // Sorted triangles from quad 0-1-2-3: (0,1,3) and (1,2,3)
    test(v1[0] == 0 && v2[0] == 1 && v3[0] == 3, "single quad: first triangle (0,1,3)");
    test(v1[1] == 1 && v2[1] == 2 && v3[1] == 3, "single quad: second triangle (1,2,3)");
}

//   +---+  (+)   +---+          +---+  (+)   +---+
//   | 0 | -----> | 1 |          | 2 | -----> | 3 |
//   +---+        +---+          +---+        +---+
//     |                                        ^
//     +................(rep)...................+
//
// Repulsive edge 0-3, positive edges 0-1 and 2-3 (disconnected).
// w=1: common neighbours of (1,3) = {}. No quadrangles.
template<template<typename> class VectorType>
void test_no_quadrangles()
{
    // Positive edges: 0-1, 2-3 (disconnected)
    auto [offsets, col_ids, costs] = build_positive_csr<VectorType>(4, {0, 2}, {1, 3});

    VectorType<int> rep_tails(1); rep_tails[0] = 0;
    VectorType<int> rep_heads(1); rep_heads[0] = 3;
    VectorType<float> rep_costs(1); rep_costs[0] = -1.0f;

    auto [v1, v2, v3, strength] = find_quadrangles<VectorType>(
        rep_tails, rep_heads, offsets, col_ids, costs, rep_costs);

    test(v1.size() == 0, "no quads: should find 0 triangles");
}

//                +-------------------------+
//                |                         |
//                |   (+)                   |
//   +------------+------------+            | (+)
//   |            |            v            v
// +---+  (+)   +---+  (+)   +---+  (+)   +------+
// | 0 | -----> | 1 | -----> | 2 | -----> |  3   |
// +---+        +---+        +---+        +------+
//   |   (rep)                              ^
//   +--------------------------------------+
//
// K4 minus edge 0-3 (all positive), repulsive edge 0-3.
// w=1: common neighbours of (1,3) = {2}. Quad 0-1-2-3, triangles (0,1,3),(1,2,3).
// w=2: common neighbours of (2,3) = {1}. Quad 0-2-1-3, triangles (0,2,3),(1,2,3).
// 3 unique triangles: (0,1,3), (0,2,3), (1,2,3).
template<template<typename> class VectorType>
void test_multiple_quadrangles_one_edge()
{
    // 4 nodes, positive edges: 0-1, 0-2, 1-2, 1-3, 2-3
    auto [offsets, col_ids, costs] = build_positive_csr<VectorType>(
        4, {0, 0, 1, 1, 2}, {1, 2, 2, 3, 3});

    VectorType<int> rep_tails(1); rep_tails[0] = 0;
    VectorType<int> rep_heads(1); rep_heads[0] = 3;
    VectorType<float> rep_costs(1); rep_costs[0] = -1.0f;

    auto [v1, v2, v3, strength] = find_quadrangles<VectorType>(
        rep_tails, rep_heads, offsets, col_ids, costs, rep_costs);

    // Two quadrangles: 0-1-2-3 and 0-2-1-3
    // Triangles (deduplicated): (0,1,3), (1,2,3), (0,2,3)
    test(v1.size() == 3, "multiple quads: should find 3 unique triangles");
    test(v1[0] == 0 && v2[0] == 1 && v3[0] == 3, "tri (0,1,3)");
    test(v1[1] == 0 && v2[1] == 2 && v3[1] == 3, "tri (0,2,3)");
    test(v1[2] == 1 && v2[2] == 2 && v3[2] == 3, "tri (1,2,3)");
}

// +---+  (+)   +---+  (+)   +---+
// | 0 | -----> | 2 | <----- | 1 |
// +---+        +---+        +---+
//
// No repulsive edges -> no quadrangles.
template<template<typename> class VectorType>
void test_quad_empty_input()
{
    auto [offsets, col_ids, costs] = build_positive_csr<VectorType>(3, {0, 1}, {2, 2});

    VectorType<int> rep_tails;
    VectorType<int> rep_heads;
    VectorType<float> rep_costs;

    auto [v1, v2, v3, strength] = find_quadrangles<VectorType>(
        rep_tails, rep_heads, offsets, col_ids, costs, rep_costs);

    test(v1.size() == 0, "empty input: should find 0 triangles");
}

// Brute-force: for each repulsive edge (u,v), for each neighbor w of u in the
// positive graph, for each common neighbor m of (w,v), emit sorted triangles
// (u,v,w) and (v,w,m). Deduplicate before returning.
inline std::vector<std::tuple<int,int,int>> brute_force_quadrangles(
    const int num_nodes,
    const std::vector<int>& rep_tails, const std::vector<int>& rep_heads,
    const std::vector<int>& pos_tails, const std::vector<int>& pos_heads)
{
    // Build adjacency set
    std::set<std::pair<int,int>> pos_adj;
    std::vector<std::vector<int>> adj(num_nodes);
    for (size_t i = 0; i < pos_tails.size(); ++i)
    {
        pos_adj.insert({pos_tails[i], pos_heads[i]});
        pos_adj.insert({pos_heads[i], pos_tails[i]});
        adj[pos_tails[i]].push_back(pos_heads[i]);
        adj[pos_heads[i]].push_back(pos_tails[i]);
    }

    auto sorted_tri = [](int a, int b, int c) {
        int min_v = std::min({a, b, c});
        int max_v = std::max({a, b, c});
        int mid_v = a + b + c - min_v - max_v;
        return std::make_tuple(min_v, mid_v, max_v);
    };

    std::set<std::tuple<int,int,int>> result_set;
    for (size_t i = 0; i < rep_tails.size(); ++i)
    {
        const int u = rep_tails[i];
        const int v = rep_heads[i];
        for (int w : adj[u])
        {
            for (int m : adj[w])
            {
                if (m == u || m == w) continue;
                if (!pos_adj.count({m, v})) continue;
                // Found quadrangle u-w-m-v
                result_set.insert(sorted_tri(u, v, w));
                result_set.insert(sorted_tri(v, w, m));
            }
        }
    }

    std::vector<std::tuple<int,int,int>> result(result_set.begin(), result_set.end());
    std::sort(result.begin(), result.end());
    return result;
}

template<template<typename> class VectorType>
void test_quad_random_graphs()
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

        auto [offsets, col_ids, csr_costs] = build_positive_csr<VectorType>(
            rg.num_nodes, pos_tails, pos_heads, pos_costs_vec);

        VectorType<int> rep_tails(neg_tails.begin(), neg_tails.end());
        VectorType<int> rep_heads(neg_heads.begin(), neg_heads.end());
        VectorType<float> rep_costs(neg_costs_vec.begin(), neg_costs_vec.end());

        auto [v1, v2, v3, strength] = find_quadrangles<VectorType>(
            rep_tails, rep_heads, offsets, col_ids, csr_costs, rep_costs);

        thrust::host_vector<int> h_v1(v1), h_v2(v2), h_v3(v3);

        std::vector<std::tuple<int,int,int>> found;
        for (size_t i = 0; i < h_v1.size(); ++i)
            found.push_back({h_v1[i], h_v2[i], h_v3[i]});
        std::sort(found.begin(), found.end());

        auto expected = brute_force_quadrangles(
            rg.num_nodes, neg_tails, neg_heads, pos_tails, pos_heads);

        const std::string label = "n=" + std::to_string(num_nodes)
            + " p=" + std::to_string(edge_prob) + " seed=" + std::to_string(seed);
        test(found.size() == expected.size(), label + ": quad triangle count mismatch ("
            + std::to_string(found.size()) + " vs " + std::to_string(expected.size()) + ")");
        test(found == expected, label + ": quad triangle content mismatch");
    }
}

template<template<typename> class VectorType>
void run_all_find_quadrangles_tests()
{
    test_single_quadrangle<VectorType>();
    test_no_quadrangles<VectorType>();
    test_multiple_quadrangles_one_edge<VectorType>();
    test_quad_empty_input<VectorType>();
    test_quad_random_graphs<VectorType>();
}
