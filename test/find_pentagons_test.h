#pragma once

#include "find_pentagons.h"
#include "find_triangles_test.h" // for build_positive_csr
#include "random_graph.h"
#include "test.h"
#include <vector>
#include <tuple>
#include <set>
#include <algorithm>

// Pentagon: v1 - v1_n1 - mid - v2_n1 - v2, where (v1,v2) is repulsive and
// all other edges are positive.
// Decomposed into sorted triangles (v1,v2,v1_n1), (v2,v1_n1,mid), (v2,mid,v2_n1).

//       (rep)
//   +---------------------------------------------------+
//   |                                                   v
// +---+  (+)   +---+  (+)   +---+  (+)   +---+  (+)   +---+
// | 0 | -----> | 1 | -----> | 2 | -----> | 3 | -----> | 4 |
// +---+        +---+        +---+        +---+        +---+
//
// Repulsive edge 0-4, positive path 0-1-2-3-4.
// v1=0, v2=4. v1_n1=1 (neighbour of 0), v2_n1=3 (neighbour of 4).
// Common neighbours of (1,3) excluding {0,4} = {2}.
// Pentagon 0-1-2-3-4, triangles: (0,1,4), (1,2,4), (2,3,4).
template<template<typename> class VectorType>
void test_single_pentagon()
{
    // Positive edges: 0-1, 1-2, 2-3, 3-4
    auto [offsets, col_ids, costs] = build_positive_csr<VectorType>(5, {0, 1, 2, 3}, {1, 2, 3, 4});

    VectorType<int> rep_tails(1); rep_tails[0] = 0;
    VectorType<int> rep_heads(1); rep_heads[0] = 4;
    VectorType<float> rep_costs(1); rep_costs[0] = -1.0f;

    auto [v1, v2, v3, strength] = find_pentagons<VectorType>(
        rep_tails, rep_heads, offsets, col_ids, costs, rep_costs);

    test(v1.size() == 3, "single pent: should find 3 triangles, found " + std::to_string(v1.size()));
    // Sorted triangles from pentagon 0-1-2-3-4: (0,1,4), (1,2,4), (2,3,4)
    test(v1[0] == 0 && v2[0] == 1 && v3[0] == 4, "single pent: first triangle (0,1,4)");
    test(v1[1] == 1 && v2[1] == 2 && v3[1] == 4, "single pent: second triangle (1,2,4)");
    test(v1[2] == 2 && v2[2] == 3 && v3[2] == 4, "single pent: third triangle (2,3,4)");
}

// +---+  (+)     +---+
// | 0 | -------> | 1 |
// +---+          +---+
//   |   (rep)
//   +--------------+
//                  v
// +---+  (+)     +---+
// | 3 | -------> | 4 |
// +---+          +---+
//
// Repulsive edge 0-4, positive edges 0-1 and 3-4 (disconnected path, no mid node).
// No pentagons.
template<template<typename> class VectorType>
void test_no_pentagons()
{
    // Positive edges: 0-1, 3-4 (disconnected)
    auto [offsets, col_ids, costs] = build_positive_csr<VectorType>(5, {0, 3}, {1, 4});

    VectorType<int> rep_tails(1); rep_tails[0] = 0;
    VectorType<int> rep_heads(1); rep_heads[0] = 4;
    VectorType<float> rep_costs(1); rep_costs[0] = -1.0f;

    auto [v1, v2, v3, strength] = find_pentagons<VectorType>(
        rep_tails, rep_heads, offsets, col_ids, costs, rep_costs);

    test(v1.size() == 0, "no pents: should find 0 triangles");
}

// No repulsive edges -> no pentagons.
template<template<typename> class VectorType>
void test_pent_empty_input()
{
    auto [offsets, col_ids, costs] = build_positive_csr<VectorType>(5, {0, 1, 2, 3}, {1, 2, 3, 4});

    VectorType<int> rep_tails;
    VectorType<int> rep_heads;
    VectorType<float> rep_costs;

    auto [v1, v2, v3, strength] = find_pentagons<VectorType>(
        rep_tails, rep_heads, offsets, col_ids, costs, rep_costs);

    test(v1.size() == 0, "empty input: should find 0 triangles");
}

// Two pentagons sharing the same repulsive edge (0,4):
//
//          (rep)
//   +------------------------------------------------------+
//   |                                                      v
// +------+  (+)   +---+  (+)   +---+  (+)   +---+  (+)   +---+
// |  0   | -----> | 1 | -----> | 2 | -----> | 3 | -----> | 4 |
// +------+        +---+        +---+        +---+        +---+
//   |                                                      ^
//   | (+)                                                  |
//   v                                                      |
// +------+  (+)   +---+  (+)   +---+  (+)                  |
// |  5   | -----> | 6 | -----> | 7 | ----------------------+
// +------+        +---+        +---+
//
// Pentagon 1: 0-1-2-3-4, triangles: (0,1,4), (1,2,4), (2,3,4)
// Pentagon 2: 0-5-6-7-4, triangles: (0,4,5), (4,5,6), (4,6,7)
// All 6 triangles are unique.
template<template<typename> class VectorType>
void test_two_pentagons_shared_repulsive_edge()
{
    // Positive edges: 0-1, 1-2, 2-3, 3-4, 0-5, 5-6, 6-7, 7-4
    auto [offsets, col_ids, costs] = build_positive_csr<VectorType>(
        8, {0, 1, 2, 3, 0, 5, 6, 7}, {1, 2, 3, 4, 5, 6, 7, 4});

    VectorType<int> rep_tails(1); rep_tails[0] = 0;
    VectorType<int> rep_heads(1); rep_heads[0] = 4;
    VectorType<float> rep_costs(1); rep_costs[0] = -1.0f;

    auto [v1, v2, v3, strength] = find_pentagons<VectorType>(
        rep_tails, rep_heads, offsets, col_ids, costs, rep_costs);

    test(v1.size() == 6, "two pents shared edge: should find 6 triangles, found "
        + std::to_string(v1.size()));
    // Sorted output: (0,1,4), (0,4,5), (1,2,4), (2,3,4), (4,5,6), (4,6,7)
    test(v1[0] == 0 && v2[0] == 1 && v3[0] == 4, "two pents: tri (0,1,4)");
    test(v1[1] == 0 && v2[1] == 4 && v3[1] == 5, "two pents: tri (0,4,5)");
    test(v1[2] == 1 && v2[2] == 2 && v3[2] == 4, "two pents: tri (1,2,4)");
    test(v1[3] == 2 && v2[3] == 3 && v3[3] == 4, "two pents: tri (2,3,4)");
    test(v1[4] == 4 && v2[4] == 5 && v3[4] == 6, "two pents: tri (4,5,6)");
    test(v1[5] == 4 && v2[5] == 6 && v3[5] == 7, "two pents: tri (4,6,7)");
}

// Brute-force: enumerate all 5-cycles v1-a-m-b-v2 where (v1,v2) is repulsive
// and (v1,a), (a,m), (m,b), (b,v2) are all positive. All 5 nodes must be distinct.
// Each pentagon is decomposed into 3 sorted triangles.
inline std::vector<std::tuple<int,int,int>> brute_force_pentagons(
    const int num_nodes,
    const std::vector<int>& rep_tails, const std::vector<int>& rep_heads,
    const std::vector<int>& pos_tails, const std::vector<int>& pos_heads)
{
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
        const int v1 = rep_tails[i];
        const int v2 = rep_heads[i];
        // Enumerate all neighbours a of v1, b of v2
        for (int a : adj[v1])
        {
            if (a == v2) continue;
            for (int b : adj[v2])
            {
                if (b == v1 || b == a) continue;
                // Find common neighbours m of (a, b) excluding v1, v2
                for (int m : adj[a])
                {
                    if (m == v1 || m == v2 || m == a || m == b) continue;
                    if (!pos_adj.count({m, b})) continue;
                    // Found pentagon v1-a-m-b-v2
                    result_set.insert(sorted_tri(v1, v2, a));
                    result_set.insert(sorted_tri(v2, a, m));
                    result_set.insert(sorted_tri(v2, m, b));
                }
            }
        }
    }

    std::vector<std::tuple<int,int,int>> result(result_set.begin(), result_set.end());
    std::sort(result.begin(), result.end());
    return result;
}

template<template<typename> class VectorType>
void test_pent_random_graphs()
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

        auto [v1, v2, v3, strength] = find_pentagons<VectorType>(
            rep_tails, rep_heads, offsets, col_ids, csr_costs, rep_costs);

        thrust::host_vector<int> h_v1(v1), h_v2(v2), h_v3(v3);

        std::vector<std::tuple<int,int,int>> found;
        for (size_t i = 0; i < h_v1.size(); ++i)
            found.push_back({h_v1[i], h_v2[i], h_v3[i]});
        std::sort(found.begin(), found.end());

        auto expected = brute_force_pentagons(
            rg.num_nodes, neg_tails, neg_heads, pos_tails, pos_heads);

        const std::string label = "n=" + std::to_string(num_nodes)
            + " p=" + std::to_string(edge_prob) + " seed=" + std::to_string(seed);
        test(found.size() == expected.size(), label + ": pent triangle count mismatch ("
            + std::to_string(found.size()) + " vs " + std::to_string(expected.size()) + ")");
        test(found == expected, label + ": pent triangle content mismatch");
    }
}

template<template<typename> class VectorType>
void run_all_find_pentagons_tests()
{
    test_single_pentagon<VectorType>();
    test_no_pentagons<VectorType>();
    test_pent_empty_input<VectorType>();
    test_two_pentagons_shared_repulsive_edge<VectorType>();
    test_pent_random_graphs<VectorType>();
}
