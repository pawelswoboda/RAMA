#pragma once

#include "test.h"
#include "graph.h"
#include "lifted_cut_constraints.h"
#include "connected_components.h"
#include "rama_utils.h"
#include "random_graph.h"

#include <iostream>
#include <vector>
#include <algorithm>
#include <numeric>
#include <set>
#include <cmath>
#include <limits>

// Helper: build a symmetric Graph from undirected edge list.
template<template<typename> class VectorType>
Graph<VectorType> make_graph(int num_nodes,
                             const std::vector<int>& i,
                             const std::vector<int>& j,
                             const std::vector<float>& c)
{
    std::vector<int> t, h;
    std::vector<float> costs;
    for (size_t e = 0; e < i.size(); e++)
    {
        t.push_back(i[e]); h.push_back(j[e]); costs.push_back(c[e]);
        t.push_back(j[e]); h.push_back(i[e]); costs.push_back(c[e]);
    }
    VectorType<int> vt(t.begin(), t.end());
    VectorType<int> vh(h.begin(), h.end());
    VectorType<float> vc(costs.begin(), costs.end());
    return Graph<VectorType>(num_nodes, vt.begin(), vt.end(),
                             vh.begin(), vh.end(), vc.begin(), vc.end(),
                             false, true);
}

// Test 1: Single Q-edge -- simplest violation.
//
//         +10
//     .........................
//     :                       v
//   +---+  +3   +---+  -5   +---+
//   | 0 | ----> | 1 | ----> | 2 |
//   +---+       +---+       +---+
//
//   Non-neg subgraph: {0-1}. Components: {0,1}, {2}.
//   Q: 1 edge between the two components.
//   Expect: 1 constraint, cut_size=1, min_abs_base_cost=5.
template<template<typename> class VectorType>
void test_single_q_edge()
{
    std::cout << "test_single_q_edge ... ";

    auto base = make_graph<VectorType>(3, {0, 1}, {1, 2}, {3.0f, -5.0f});
    auto lifted = make_graph<VectorType>(3, {0}, {2}, {10.0f});

    auto constraints = find_lifted_cut_constraints<VectorType>(base, lifted);

    test(constraints.size() == 1, "expected 1 constraint");
    test(constraints[0].cut_size == 1, "expected cut_size=1");
    test(constraints[0].lifted_cost == 10.0f, "expected lifted_cost=10");
    test(constraints[0].min_abs_base_cost == 5.0f, "expected min_abs_base_cost=5");
    test(constraints[0].cut_comp_pairs.size() == 1, "expected 1 cut comp pair");

    std::cout << "passed\n";
}

// Test 2: Two components, two parallel Q-edges.
//
//           +10
//     ...........................
//     :                         v
//   +-----+  +5   +---+  -1   +---+
//   |  0  | ----> | 1 | ----> | 3 |
//   +-----+       +---+       +---+
//     |                         ^
//     | -1                      |
//     v                         |
//   +-----+  +5                 |
//   |  2  | --------------------+
//   +-----+
//
//   Components: {0,1}, {2,3}. Q: 2 parallel edges between the same pair.
//   With only 2 Q-nodes, Karger can never contract (would merge s,t).
//   Cut always includes all inter-component edges. cut_size=2.
template<template<typename> class VectorType>
void test_two_component_parallel()
{
    std::cout << "test_two_component_parallel ... ";

    auto base = make_graph<VectorType>(4,
        {0, 2, 0, 1}, {1, 3, 2, 3}, {5.0f, 5.0f, -1.0f, -1.0f});
    auto lifted = make_graph<VectorType>(4, {0}, {3}, {10.0f});

    auto constraints = find_lifted_cut_constraints<VectorType>(base, lifted);

    test(constraints.size() == 1, "expected 1 constraint");
    test(constraints[0].cut_size == 2, "expected cut_size=2");
    test(constraints[0].min_abs_base_cost == 1.0f, "expected min_abs=1");

    std::cout << "passed\n";
}

// Test 3: Path quotient (3 components) -- min cut = 1.
//
//           +10
//     .........................................
//     :                                       v
//   +-----+  -1   +-----+  -1   +---+  +5   +---+
//   |  0  | ----> |  2  | ----> | 4 | ----> | 5 |
//   +-----+       +-----+       +---+       +---+
//     |             |
//     | +5          | +5
//     v             v
//   +-----+       +-----+
//   |  1  |       |  3  |
//   +-----+       +-----+
//
//   Components: A={0,1}, B={2,3}, C={4,5}.
//   Quotient graph Q (path):
//     +---+  1   +---+  1   +---+
//     | A | ---> | B | ---> | C |
//     +---+      +---+      +---+
//   Karger contracts B into one side, yielding cut_size=1.
template<template<typename> class VectorType>
void test_path_quotient()
{
    std::cout << "test_path_quotient ... ";

    auto base = make_graph<VectorType>(6,
        {0, 2, 4, 0, 2},
        {1, 3, 5, 2, 4},
        {5.0f, 5.0f, 5.0f, -1.0f, -1.0f});
    auto lifted = make_graph<VectorType>(6, {0}, {5}, {10.0f});

    auto constraints = find_lifted_cut_constraints<VectorType>(base, lifted);

    test(constraints.size() == 1, "expected 1 constraint");
    test(constraints[0].cut_size == 1, "expected cut_size=1");
    test(constraints[0].min_abs_base_cost == 1.0f, "expected min_abs=1");

    std::cout << "passed\n";
}

// Test 4: Triangle quotient (3 components) -- min cut = 2.
//
//           -1
//     +---------------------------+
//     |                           v
//   +-----+  -1   +-----+  -1   +---+  +5   +---+
//   |  0  | ----> |  2  | ----> | 4 | ----> | 5 |
//   +-----+       +-----+       +---+       +---+
//     |             |                         ^
//     | +5          |                         :
//     v             |                         :
//   +-----+  +8    |                         :
//   |  1  | ......!...........................
//   +-----+        |
//                   | +5
//                   v
//                 +-----+
//                 |  3  |
//                 +-----+
//
//   Components: A={0,1}, B={2,3}, C={4,5}.
//   Quotient graph Q (triangle):
//           1
//     +---------------------+
//     |                     v
//   +---+  1   +---+  1   +---+
//   | A | ---> | B | ---> | C |
//   +---+      +---+      +---+
//   Min A-C cut = 2: any partition puts B on one side, leaving 2 crossing edges.
template<template<typename> class VectorType>
void test_triangle_quotient()
{
    std::cout << "test_triangle_quotient ... ";

    auto base = make_graph<VectorType>(6,
        {0, 2, 4, 0, 2, 0},
        {1, 3, 5, 2, 4, 4},
        {5.0f, 5.0f, 5.0f, -1.0f, -1.0f, -1.0f});
    auto lifted = make_graph<VectorType>(6, {1}, {5}, {8.0f});

    auto constraints = find_lifted_cut_constraints<VectorType>(base, lifted);

    test(constraints.size() == 1, "expected 1 constraint");
    test(constraints[0].cut_size == 2, "expected cut_size=2");
    test(constraints[0].lifted_cost == 8.0f, "expected lifted_cost=8");

    std::cout << "passed\n";
}

// Test 5: No violations -- single component (all base edges non-negative).
//
//   +---+  +3   +---+  +5   +---+
//   | 0 | ----> | 1 | ----> | 2 |
//   +---+       +---+       +---+
//   Lifted: (0,2, +10). All in one component. No violations.
template<template<typename> class VectorType>
void test_no_violations_single_component()
{
    std::cout << "test_no_violations_single_component ... ";

    auto base = make_graph<VectorType>(3, {0, 1}, {1, 2}, {3.0f, 5.0f});
    auto lifted = make_graph<VectorType>(3, {0}, {2}, {10.0f});

    auto constraints = find_lifted_cut_constraints<VectorType>(base, lifted);
    test(constraints.empty(), "expected no constraints");

    std::cout << "passed\n";
}

// Test 6: Empty quotient graph -- two components but no negative inter-component edges.
//   [0] --+5-- [1]     [2] isolated.     Lifted: (0,2, +10).
//   Components: {0,1}, {2}. Q is empty (no negative edges between them).
template<template<typename> class VectorType>
void test_empty_quotient()
{
    std::cout << "test_empty_quotient ... ";

    auto base = make_graph<VectorType>(3, {0}, {1}, {5.0f});
    auto lifted = make_graph<VectorType>(3, {0}, {2}, {10.0f});

    auto constraints = find_lifted_cut_constraints<VectorType>(base, lifted);
    test(constraints.empty(), "expected no constraints (empty Q)");

    std::cout << "passed\n";
}

// Test 7: Multiple violations sharing the same component pair -- same cut reused.
//
//         +10             +7
//     .............   .........
//     :           v   :       v
//   +---+  +5   +---+  -3   +---+
//   | 0 | ----> | 1 | ----> | 2 |
//   +---+       +---+       +---+
//
//   Both lifted edges cross the same component pair.
//   Expect 2 constraints with identical cuts.
template<template<typename> class VectorType>
void test_multiple_violations_same_pair()
{
    std::cout << "test_multiple_violations_same_pair ... ";

    auto base = make_graph<VectorType>(3, {0, 1}, {1, 2}, {5.0f, -3.0f});
    auto lifted = make_graph<VectorType>(3, {0, 1}, {2, 2}, {10.0f, 7.0f});

    auto constraints = find_lifted_cut_constraints<VectorType>(base, lifted);

    test(constraints.size() == 2, "expected 2 constraints");
    test(constraints[0].cut_size == constraints[1].cut_size,
         "same pair should yield same cut_size");
    test(constraints[0].cut_comp_pairs == constraints[1].cut_comp_pairs,
         "same pair should yield same cut_comp_pairs");
    test(constraints[0].lifted_fwd_idx != constraints[1].lifted_fwd_idx,
         "should reference different lifted edges");

    std::cout << "passed\n";
}

// Test 8: Empty lifted graph.
template<template<typename> class VectorType>
void test_empty_lifted()
{
    std::cout << "test_empty_lifted ... ";

    auto base = make_graph<VectorType>(3, {0, 1}, {1, 2}, {3.0f, -5.0f});
    Graph<VectorType> lifted;

    auto constraints = find_lifted_cut_constraints<VectorType>(base, lifted);
    test(constraints.empty(), "expected no constraints for empty lifted");

    std::cout << "passed\n";
}

// Test 9: No positive lifted edges -- all lifted costs <= 0, no violations.
//
//         -2
//     .........................
//     :                       v
//   +---+  +5   +---+  -3   +---+
//   | 0 | ----> | 1 | ----> | 2 |
//   +---+       +---+       +---+
//
//   Negative lifted edge is not a violation.
template<template<typename> class VectorType>
void test_no_positive_lifted()
{
    std::cout << "test_no_positive_lifted ... ";

    auto base = make_graph<VectorType>(3, {0, 1}, {1, 2}, {5.0f, -3.0f});
    auto lifted = make_graph<VectorType>(3, {0}, {2}, {-2.0f});

    auto constraints = find_lifted_cut_constraints<VectorType>(base, lifted);
    test(constraints.empty(), "expected no constraints for non-positive lifted");

    std::cout << "passed\n";
}

// Test 10: Verify min_abs_base_cost with two parallel negative edges of different cost.
//
//           +10
//     ...........................
//     :                         v
//   +-----+  +5   +---+  -7   +---+
//   |  0  | ----> | 1 | ----> | 3 |
//   +-----+       +---+       +---+
//     |                         ^
//     | -3                      |
//     v                         |
//   +-----+  +5                 |
//   |  2  | --------------------+
//   +-----+
//
//   Components: {0,1}, {2,3}. Q: 2 parallel edges. cut_size=2.
//   min_abs = min(3,7) = 3.
template<template<typename> class VectorType>
void test_min_abs_base_cost()
{
    std::cout << "test_min_abs_base_cost ... ";

    auto base = make_graph<VectorType>(4,
        {0, 2, 0, 1}, {1, 3, 2, 3}, {5.0f, 5.0f, -3.0f, -7.0f});
    auto lifted = make_graph<VectorType>(4, {0}, {2}, {10.0f});

    auto constraints = find_lifted_cut_constraints<VectorType>(base, lifted);

    test(constraints.size() == 1, "expected 1 constraint");
    test(constraints[0].cut_size == 2, "expected cut_size=2");
    test(constraints[0].min_abs_base_cost == 3.0f, "expected min_abs=3");

    std::cout << "passed\n";
}

// ---- Random graph tests ----

namespace lcc_test_detail {

struct UnionFind {
    std::vector<int> parent;
    UnionFind(int n) : parent(n) { std::iota(parent.begin(), parent.end(), 0); }
    int find(int x) {
        while (parent[x] != x) { parent[x] = parent[parent[x]]; x = parent[x]; }
        return x;
    }
    void unite(int a, int b) { parent[find(b)] = find(a); }
    bool connected(int a, int b) { return find(a) == find(b); }
};

} // namespace lcc_test_detail

// Test 11: Random graphs -- verify completeness and cut validity.
//
// For several random instances:
//   (a) Every positive lifted edge with endpoints in different components of
//       the non-negative base subgraph (and reachable in Q) produces a constraint.
//   (b) Each reported cut is a valid s-t cut: merging all non-cut Q-edges
//       leaves s_comp and t_comp disconnected.
//   (c) cut_size and min_abs_base_cost are consistent with the base graph.
template<template<typename> class VectorType>
void test_random_completeness_and_validity()
{
    std::cout << "test_random_completeness_and_validity ... ";

    const int sizes[] = {8, 15, 30, 50};
    const double base_probs[] = {0.15, 0.3, 0.5};
    const double lifted_probs[] = {0.1, 0.2, 0.4};

    unsigned seed = 100;
    for (int n : sizes)
    for (double bp : base_probs)
    for (double lp : lifted_probs)
    {
        auto rg = generate_random_graph(n, bp, lp, seed++);
        if (rg.tails.empty() || rg.lifted_tails.empty())
            continue;

        auto base = make_graph<VectorType>(rg.num_nodes, rg.tails, rg.heads, rg.costs);
        auto lifted = make_graph<VectorType>(rg.num_nodes,
            rg.lifted_tails, rg.lifted_heads, rg.lifted_costs);

        auto constraints = find_lifted_cut_constraints<VectorType>(base, lifted);

        // Recompute CC on non-negative base subgraph (on host for verification)
        const int num_base_dir = (int)base.num_directed_edges();
        const int num_lifted_dir = (int)lifted.num_directed_edges();
        thrust::host_vector<int> h_bt(base.get_tails());
        thrust::host_vector<int> h_bh(base.get_heads());
        thrust::host_vector<float> h_bc(base.get_costs());
        thrust::host_vector<int> h_lt(lifted.get_tails());
        thrust::host_vector<int> h_lh(lifted.get_heads());
        thrust::host_vector<float> h_lc(lifted.get_costs());

        std::vector<int> at, ah;
        for (int e = 0; e < num_base_dir; e++)
            if (h_bc[e] >= 0.0f) { at.push_back(h_bt[e]); ah.push_back(h_bh[e]); }

        thrust::host_vector<int> hv_at(at.begin(), at.end());
        thrust::host_vector<int> hv_ah(ah.begin(), ah.end());
        thrust::host_vector<int> comp =
            connected_components::compute_cc<thrust::host_vector>(rg.num_nodes, hv_at, hv_ah);
        int ml = *std::max_element(comp.begin(), comp.end());
        comp = compress_label_sequence<thrust::host_vector>(comp, ml);
        int num_comp = *std::max_element(comp.begin(), comp.end()) + 1;

        // Build quotient graph Q on host
        std::vector<int> q_src, q_dst;
        for (int e = 0; e < num_base_dir; e++)
        {
            if (h_bc[e] >= 0.0f) continue;
            int ct = comp[h_bt[e]], ch = comp[h_bh[e]];
            if (ct < ch) { q_src.push_back(ct); q_dst.push_back(ch); }
        }

        // Q-reachability via union-find (merge all Q edges)
        lcc_test_detail::UnionFind q_uf(num_comp);
        for (size_t e = 0; e < q_src.size(); e++)
            q_uf.unite(q_src[e], q_dst[e]);

        // (a) Completeness: every expected violation should have a constraint
        std::set<int> found_fwd;
        for (const auto& c : constraints)
            found_fwd.insert(c.lifted_fwd_idx);

        for (int e = 0; e < num_lifted_dir; e++)
        {
            if (h_lt[e] >= h_lh[e]) continue;
            if (h_lc[e] <= 0.0f) continue;
            int ct = comp[h_lt[e]], ch = comp[h_lh[e]];
            if (ct == ch) continue;
            int sc = std::min(ct, ch), tc = std::max(ct, ch);
            if (!q_uf.connected(sc, tc)) continue;
            test(found_fwd.count(e) > 0,
                 "seed " + std::to_string(seed) +
                 ": missing constraint for violated lifted edge " + std::to_string(e));
        }

        // (b) Cut validity and (c) consistency for each constraint
        for (const auto& c : constraints)
        {
            int lt = h_lt[c.lifted_fwd_idx], lh = h_lh[c.lifted_fwd_idx];
            int sc = std::min(comp[lt], comp[lh]);
            int tc = std::max(comp[lt], comp[lh]);

            // (b) Merge all Q-edges NOT in the cut; s and t must stay disconnected.
            lcc_test_detail::UnionFind cut_uf(num_comp);
            for (size_t e = 0; e < q_src.size(); e++)
            {
                auto p = std::make_pair(std::min(q_src[e], q_dst[e]),
                                        std::max(q_src[e], q_dst[e]));
                if (!c.cut_comp_pairs.count(p))
                    cut_uf.unite(q_src[e], q_dst[e]);
            }
            test(!cut_uf.connected(sc, tc),
                 "seed " + std::to_string(seed) +
                 ": cut does not disconnect s from t");

            // (b2) Minimality: adding back any single cut pair must reconnect s-t.
            for (const auto& cp : c.cut_comp_pairs)
            {
                lcc_test_detail::UnionFind min_uf(num_comp);
                for (size_t e = 0; e < q_src.size(); e++)
                {
                    auto p = std::make_pair(std::min(q_src[e], q_dst[e]),
                                            std::max(q_src[e], q_dst[e]));
                    if (!c.cut_comp_pairs.count(p) || p == cp)
                        min_uf.unite(q_src[e], q_dst[e]);
                }
                test(min_uf.connected(sc, tc),
                     "seed " + std::to_string(seed) +
                     ": cut is not minimal, pair (" +
                     std::to_string(cp.first) + "," + std::to_string(cp.second) +
                     ") is redundant");
            }

            // (c) Verify cut_size and min_abs_base_cost
            int expected_cut_size = 0;
            float expected_min_abs = std::numeric_limits<float>::max();
            for (int e = 0; e < num_base_dir; e++)
            {
                if (h_bc[e] >= 0.0f) continue;
                if (h_bt[e] >= h_bh[e]) continue;
                int ct2 = comp[h_bt[e]], ch2 = comp[h_bh[e]];
                auto p = std::make_pair(std::min(ct2, ch2), std::max(ct2, ch2));
                if (c.cut_comp_pairs.count(p))
                {
                    expected_cut_size++;
                    expected_min_abs = std::min(expected_min_abs, std::abs(h_bc[e]));
                }
            }

            test(c.cut_size == expected_cut_size,
                 "seed " + std::to_string(seed) + ": cut_size mismatch");
            test(std::abs(c.min_abs_base_cost - expected_min_abs) < 1e-6f,
                 "seed " + std::to_string(seed) + ": min_abs_base_cost mismatch");
            test(c.cut_size > 0,
                 "seed " + std::to_string(seed) + ": cut_size should be positive");
            test(c.lifted_cost > 0.0f,
                 "seed " + std::to_string(seed) + ": lifted_cost should be positive");
        }
    }

    std::cout << "passed\n";
}

template<template<typename> class VectorType>
void run_all_lifted_cut_constraints_tests()
{
    test_single_q_edge<VectorType>();
    test_two_component_parallel<VectorType>();
    test_path_quotient<VectorType>();
    test_triangle_quotient<VectorType>();
    test_no_violations_single_component<VectorType>();
    test_empty_quotient<VectorType>();
    test_multiple_violations_same_pair<VectorType>();
    test_empty_lifted<VectorType>();
    test_no_positive_lifted<VectorType>();
    test_min_abs_base_cost<VectorType>();
    test_random_completeness_and_validity<VectorType>();
}