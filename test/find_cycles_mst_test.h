#pragma once

#include "find_cycles_mst.h"
#include "graph.h"
#include "test.h"
#include <vector>
#include <tuple>
#include <algorithm>

// Build symmetric COO from single-orientation positive edges with costs.
// Returns (tails, heads, costs) in symmetric form via Graph construction.
template<template<typename> class VectorType>
struct PosCOO {
    VectorType<int> tails;
    VectorType<int> heads;
    VectorType<float> costs;
};

template<template<typename> class VectorType>
PosCOO<VectorType> build_pos_coo(int num_nodes,
                                  const std::vector<int>& t,
                                  const std::vector<int>& h,
                                  const std::vector<float>& c)
{
    Graph<VectorType> g(num_nodes,
                        t.begin(), t.end(),
                        h.begin(), h.end(),
                        c.begin(), c.end());
    PosCOO<VectorType> result;
    result.tails = g.get_tails();
    result.heads = g.get_heads();
    result.costs = VectorType<float>(g.get_costs().begin(), g.get_costs().end());
    return result;
}

// 6-node ring with one repulsive edge.
//
//     +1        +1        +1        +1        +1
// 0 -----> 5 -----> 4 -----> 3 -----> 2 -----> 1
// |                                             |
// +-------------------- -1 --------------------+
//
// Positive graph is a path: already a tree (5 edges, 6 nodes).
// MST = all positive edges. Tree path 0->5->4->3->2->1 (length 5).
// Cycle length 6. Fan from 0: 4 triangles.
template<template<typename> class VectorType>
void test_mst_6_cycle()
{
    std::vector<int> pos_t = {1, 2, 3, 4, 0};
    std::vector<int> pos_h = {2, 3, 4, 5, 5};
    std::vector<float> pos_c = {1, 1, 1, 1, 1};

    auto pos = build_pos_coo<VectorType>(6, pos_t, pos_h, pos_c);

    VectorType<int> rep_tails(1); rep_tails[0] = 0;
    VectorType<int> rep_heads(1); rep_heads[0] = 1;
    VectorType<float> rep_costs(1); rep_costs[0] = -1.0f;

    auto [v1, v2, v3, strength] = find_conflicted_cycles_mst<VectorType>(
        rep_tails, rep_heads, rep_costs,
        pos.tails, pos.heads, pos.costs,
        6, 0, false);

    test(v1.size() == 4, "mst 6-cycle: should find 4 triangles, got " + std::to_string(v1.size()));

    thrust::host_vector<int> h1(v1), h2(v2), h3(v3);
    std::vector<std::tuple<int,int,int>> found;
    for (size_t i = 0; i < h1.size(); ++i)
        found.push_back({h1[i], h2[i], h3[i]});
    std::sort(found.begin(), found.end());

    std::vector<std::tuple<int,int,int>> expected = {
        {0, 1, 2}, {0, 2, 3}, {0, 3, 4}, {0, 4, 5}};
    test(found == expected, "mst 6-cycle: triangle content mismatch");
}

// 7-node ring with one repulsive edge.
// Positive path: 0->6->5->4->3->2->1 (length 6). 5 triangles.
template<template<typename> class VectorType>
void test_mst_7_cycle()
{
    std::vector<int> pos_t = {1, 2, 3, 4, 5, 0};
    std::vector<int> pos_h = {2, 3, 4, 5, 6, 6};
    std::vector<float> pos_c = {1, 1, 1, 1, 1, 1};

    auto pos = build_pos_coo<VectorType>(7, pos_t, pos_h, pos_c);

    VectorType<int> rep_tails(1); rep_tails[0] = 0;
    VectorType<int> rep_heads(1); rep_heads[0] = 1;
    VectorType<float> rep_costs(1); rep_costs[0] = -1.0f;

    auto [v1, v2, v3, strength] = find_conflicted_cycles_mst<VectorType>(
        rep_tails, rep_heads, rep_costs,
        pos.tails, pos.heads, pos.costs,
        7, 0, false);

    test(v1.size() == 5, "mst 7-cycle: should find 5 triangles, got " + std::to_string(v1.size()));

    thrust::host_vector<int> h1(v1), h2(v2), h3(v3);
    std::vector<std::tuple<int,int,int>> found;
    for (size_t i = 0; i < h1.size(); ++i)
        found.push_back({h1[i], h2[i], h3[i]});
    std::sort(found.begin(), found.end());

    std::vector<std::tuple<int,int,int>> expected = {
        {0, 1, 2}, {0, 2, 3}, {0, 3, 4}, {0, 4, 5}, {0, 5, 6}};
    test(found == expected, "mst 7-cycle: triangle content mismatch");
}

// Repulsive edge endpoints in different positive-graph components.
// No tree path exists. 0 triangles.
template<template<typename> class VectorType>
void test_mst_no_path()
{
    // Component A: 0-1-2   Component B: 3-4-5
    std::vector<int> pos_t = {0, 1, 3, 4};
    std::vector<int> pos_h = {1, 2, 4, 5};
    std::vector<float> pos_c = {1, 1, 1, 1};

    auto pos = build_pos_coo<VectorType>(6, pos_t, pos_h, pos_c);

    VectorType<int> rep_tails(1); rep_tails[0] = 0;
    VectorType<int> rep_heads(1); rep_heads[0] = 3;
    VectorType<float> rep_costs(1); rep_costs[0] = -1.0f;

    auto [v1, v2, v3, strength] = find_conflicted_cycles_mst<VectorType>(
        rep_tails, rep_heads, rep_costs,
        pos.tails, pos.heads, pos.costs,
        6, 0, false);

    test(v1.size() == 0, "mst no path: should find 0 triangles");
}

// Triangle: repulsive (0,1), positive (0,2) and (1,2).
// MST is already a tree (2 edges). Path from 0 to 1 has length 2
// (cycle length 3 <= 5). Fan-triangulation requires path >= 5. 0 triangles.
template<template<typename> class VectorType>
void test_mst_short_path_filtered()
{
    std::vector<int> pos_t = {0, 1};
    std::vector<int> pos_h = {2, 2};
    std::vector<float> pos_c = {1, 1};

    auto pos = build_pos_coo<VectorType>(3, pos_t, pos_h, pos_c);

    VectorType<int> rep_tails(1); rep_tails[0] = 0;
    VectorType<int> rep_heads(1); rep_heads[0] = 1;
    VectorType<float> rep_costs(1); rep_costs[0] = -1.0f;

    auto [v1, v2, v3, strength] = find_conflicted_cycles_mst<VectorType>(
        rep_tails, rep_heads, rep_costs,
        pos.tails, pos.heads, pos.costs,
        3, 0, false);

    test(v1.size() == 0, "mst short path: should find 0 triangles");
}

// Graph where MST path is longer than BFS shortest path.
//
// 7 nodes. Repulsive: (0,1).
// Positive edges:
//   0-2(+10), 2-3(+10), 3-4(+10), 4-5(+10), 1-5(+10)  [high-weight chain]
//   0-6(+1), 1-6(+1)                                     [low-weight shortcut]
//
// BFS shortest positive path: 0->6->1 (length 2, cycle 3 <= 5) -> 0 triangles.
// MST drops one low-weight edge (e.g. 1-6). Tree path: 0->2->3->4->5->1
// (length 5, cycle 6) -> 4 triangles.
template<template<typename> class VectorType>
void test_mst_longer_than_bfs()
{
    std::vector<int> pos_t = {0, 2, 3, 4, 1, 0, 1};
    std::vector<int> pos_h = {2, 3, 4, 5, 5, 6, 6};
    std::vector<float> pos_c = {10, 10, 10, 10, 10, 1, 1};

    auto pos = build_pos_coo<VectorType>(7, pos_t, pos_h, pos_c);

    VectorType<int> rep_tails(1); rep_tails[0] = 0;
    VectorType<int> rep_heads(1); rep_heads[0] = 1;
    VectorType<float> rep_costs(1); rep_costs[0] = -1.0f;

    auto [v1, v2, v3, strength] = find_conflicted_cycles_mst<VectorType>(
        rep_tails, rep_heads, rep_costs,
        pos.tails, pos.heads, pos.costs,
        7, 0, false);

    test(v1.size() == 4, "mst longer than bfs: should find 4 triangles, got " + std::to_string(v1.size()));

    thrust::host_vector<int> h1(v1), h2(v2), h3(v3);
    std::vector<std::tuple<int,int,int>> found;
    for (size_t i = 0; i < h1.size(); ++i)
        found.push_back({h1[i], h2[i], h3[i]});
    std::sort(found.begin(), found.end());

    std::vector<std::tuple<int,int,int>> expected = {
        {0, 1, 5}, {0, 2, 3}, {0, 3, 4}, {0, 4, 5}};
    test(found == expected, "mst longer than bfs: triangle content mismatch");
}

// 10-node ring with max_cycle_length=7. MST path is 9 edges, exceeds
// the depth bound (7-1=6). 0 triangles.
template<template<typename> class VectorType>
void test_mst_max_length_bound()
{
    // Positive edges: 1-2, 2-3, ..., 8-9, 0-9 (a path = already a tree)
    std::vector<int> pos_t = {1, 2, 3, 4, 5, 6, 7, 8, 0};
    std::vector<int> pos_h = {2, 3, 4, 5, 6, 7, 8, 9, 9};
    std::vector<float> pos_c = {1, 1, 1, 1, 1, 1, 1, 1, 1};

    auto pos = build_pos_coo<VectorType>(10, pos_t, pos_h, pos_c);

    VectorType<int> rep_tails(1); rep_tails[0] = 0;
    VectorType<int> rep_heads(1); rep_heads[0] = 1;
    VectorType<float> rep_costs(1); rep_costs[0] = -1.0f;

    // max_cycle_length=7 -> max_depth=6, but MST path is 9 edges
    auto [v1, v2, v3, strength] = find_conflicted_cycles_mst<VectorType>(
        rep_tails, rep_heads, rep_costs,
        pos.tails, pos.heads, pos.costs,
        10, 7, false);

    test(v1.size() == 0, "mst max length bound: should find 0 triangles");

    // Same with unlimited -> should find 8 triangles (dist=9, count=8)
    auto [u1, u2, u3, u_str] = find_conflicted_cycles_mst<VectorType>(
        rep_tails, rep_heads, rep_costs,
        pos.tails, pos.heads, pos.costs,
        10, 0, false);

    test(u1.size() == 8, "mst max length unlimited: should find 8 triangles, got " + std::to_string(u1.size()));
}

template<template<typename> class VectorType>
void run_all_find_cycles_mst_tests()
{
    test_mst_6_cycle<VectorType>();
    test_mst_7_cycle<VectorType>();
    test_mst_no_path<VectorType>();
    test_mst_short_path_filtered<VectorType>();
    test_mst_longer_than_bfs<VectorType>();
    test_mst_max_length_bound<VectorType>();
}
