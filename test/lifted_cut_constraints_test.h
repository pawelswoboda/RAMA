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

// Validate structural properties of all cut factors:
//   - Base edges are negative forward inter-component base edges
//   - Lifted edges are positive forward inter-component lifted edges
//   - All lifted edges in a factor share the same component pair
//   - Base edges form a valid s-t cut in the quotient graph
template<template<typename> class VectorType>
void validate_factors(
    const LiftedCutFactors<VectorType>& factors,
    const Graph<VectorType>& base_G,
    const Graph<VectorType>& lifted_G,
    const std::string& label = "")
{
    if (factors.num_factors == 0) return;

    thrust::host_vector<int> h_bo(factors.base_offsets);
    thrust::host_vector<int> h_bi(factors.base_edge_idx);
    thrust::host_vector<int> h_lo(factors.lifted_offsets);
    thrust::host_vector<int> h_li(factors.lifted_edge_idx);

    thrust::host_vector<int> h_bt(base_G.get_tails());
    thrust::host_vector<int> h_bh(base_G.get_heads());
    thrust::host_vector<float> h_bc(base_G.get_costs());
    thrust::host_vector<int> h_lt(lifted_G.get_tails());
    thrust::host_vector<int> h_lh(lifted_G.get_heads());
    thrust::host_vector<float> h_lc(lifted_G.get_costs());

    int num_base_dir = (int)base_G.num_directed_edges();
    int num_nodes = base_G.num_nodes();

    // CC on non-negative base subgraph
    std::vector<int> at, ah;
    for (int e = 0; e < num_base_dir; e++)
        if (h_bc[e] >= 0.0f) { at.push_back(h_bt[e]); ah.push_back(h_bh[e]); }

    thrust::host_vector<int> hv_at(at.begin(), at.end());
    thrust::host_vector<int> hv_ah(ah.begin(), ah.end());
    thrust::host_vector<int> comp =
        connected_components::compute_cc<thrust::host_vector>(num_nodes, hv_at, hv_ah);
    int ml = *std::max_element(comp.begin(), comp.end());
    comp = compress_label_sequence<thrust::host_vector>(comp, ml);
    int num_comp = *std::max_element(comp.begin(), comp.end()) + 1;

    // Build quotient edge set
    std::set<std::pair<int,int>> q_edges;
    for (int e = 0; e < num_base_dir; e++)
    {
        if (h_bc[e] >= 0.0f) continue;
        int ct = comp[h_bt[e]], ch = comp[h_bh[e]];
        if (ct != ch) q_edges.insert({std::min(ct, ch), std::max(ct, ch)});
    }

    for (int f = 0; f < factors.num_factors; f++)
    {
        int base_start = h_bo[f], base_end = h_bo[f + 1];
        int lifted_start = h_lo[f], lifted_end = h_lo[f + 1];

        test(base_end > base_start,
             label + ": factor " + std::to_string(f) + " has no base edges");
        test(lifted_end > lifted_start,
             label + ": factor " + std::to_string(f) + " has no lifted edges");

        // Collect cut component pairs from base edges
        std::set<std::pair<int,int>> cut_pairs;
        for (int k = base_start; k < base_end; k++)
        {
            int idx = h_bi[k];
            test(idx >= 0 && idx < num_base_dir,
                 label + ": base edge idx out of range");
            test(h_bt[idx] < h_bh[idx],
                 label + ": base edge not forward");
            test(h_bc[idx] < 0.0f,
                 label + ": base edge not negative");
            int ct = comp[h_bt[idx]], ch = comp[h_bh[idx]];
            test(ct != ch,
                 label + ": base edge endpoints in same component");
            cut_pairs.insert({std::min(ct, ch), std::max(ct, ch)});
        }

        // All lifted edges must be positive, forward, inter-component, same pair
        std::set<std::pair<int,int>> lifted_comp_pairs;
        for (int k = lifted_start; k < lifted_end; k++)
        {
            int idx = h_li[k];
            test(idx >= 0 && idx < (int)lifted_G.num_directed_edges(),
                 label + ": lifted edge idx out of range");
            test(h_lt[idx] < h_lh[idx],
                 label + ": lifted edge not forward");
            test(h_lc[idx] > 0.0f,
                 label + ": lifted edge not positive");
            int ct = comp[h_lt[idx]], ch = comp[h_lh[idx]];
            test(ct != ch,
                 label + ": lifted edge endpoints in same component");
            lifted_comp_pairs.insert({std::min(ct, ch), std::max(ct, ch)});
        }

        test(lifted_comp_pairs.size() == 1,
             label + ": factor " + std::to_string(f) +
             " has lifted edges spanning different comp pairs");

        auto st_pair = *lifted_comp_pairs.begin();
        int sc = st_pair.first, tc = st_pair.second;

        // Cut must disconnect s from t in Q
        std::vector<int> uf(num_comp);
        std::iota(uf.begin(), uf.end(), 0);
        auto uf_find = [&](int x) {
            while (uf[x] != x) { uf[x] = uf[uf[x]]; x = uf[x]; }
            return x;
        };
        for (const auto& qe : q_edges)
        {
            if (!cut_pairs.count(qe))
            {
                int rx = uf_find(qe.first), ry = uf_find(qe.second);
                if (rx != ry) uf[ry] = rx;
            }
        }
        test(uf_find(sc) != uf_find(tc),
             label + ": cut does not disconnect s from t");
    }
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
//   Components: {0,1}, {2}. Q: 1 edge. Expect: 1 factor, 1 base edge, 1 lifted edge.
template<template<typename> class VectorType>
void test_single_q_edge()
{
    std::cout << "test_single_q_edge ... ";

    auto base = make_graph<VectorType>(3, {0, 1}, {1, 2}, {3.0f, -5.0f});
    auto lifted = make_graph<VectorType>(3, {0}, {2}, {10.0f});

    auto factors = find_lifted_cut_constraints<VectorType>(base, lifted);

    test(factors.num_factors == 1, "expected 1 factor");

    thrust::host_vector<int> bo(factors.base_offsets);
    thrust::host_vector<int> lo(factors.lifted_offsets);
    test(bo[1] - bo[0] == 1, "expected 1 base edge in cut");
    test(lo[1] - lo[0] == 1, "expected 1 lifted edge");

    validate_factors(factors, base, lifted, "single_q_edge");

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
//   Components: {0,1}, {2,3}. Q: 2 edges between same pair. cut_size=2.
template<template<typename> class VectorType>
void test_two_component_parallel()
{
    std::cout << "test_two_component_parallel ... ";

    auto base = make_graph<VectorType>(4,
        {0, 2, 0, 1}, {1, 3, 2, 3}, {5.0f, 5.0f, -1.0f, -1.0f});
    auto lifted = make_graph<VectorType>(4, {0}, {3}, {10.0f});

    auto factors = find_lifted_cut_constraints<VectorType>(base, lifted);

    test(factors.num_factors == 1, "expected 1 factor");

    thrust::host_vector<int> bo(factors.base_offsets);
    thrust::host_vector<int> lo(factors.lifted_offsets);
    test(bo[1] - bo[0] == 2, "expected 2 base edges in cut");
    test(lo[1] - lo[0] == 1, "expected 1 lifted edge");

    validate_factors(factors, base, lifted, "two_component_parallel");

    std::cout << "passed\n";
}

// Test 3: Path quotient (3 components) -- min cut = 1.
//
//   Components: A={0,1}, B={2,3}, C={4,5}.
//   Q: A--B--C (path). Min A-C cut = 1.
template<template<typename> class VectorType>
void test_path_quotient()
{
    std::cout << "test_path_quotient ... ";

    auto base = make_graph<VectorType>(6,
        {0, 2, 4, 0, 2},
        {1, 3, 5, 2, 4},
        {5.0f, 5.0f, 5.0f, -1.0f, -1.0f});
    auto lifted = make_graph<VectorType>(6, {0}, {5}, {10.0f});

    auto factors = find_lifted_cut_constraints<VectorType>(base, lifted);

    test(factors.num_factors == 1, "expected 1 factor");

    thrust::host_vector<int> bo(factors.base_offsets);
    thrust::host_vector<int> lo(factors.lifted_offsets);
    test(bo[1] - bo[0] == 1, "expected cut_size=1");
    test(lo[1] - lo[0] == 1, "expected 1 lifted edge");

    validate_factors(factors, base, lifted, "path_quotient");

    std::cout << "passed\n";
}

// Test 4: Triangle quotient (3 components) -- min cut = 2.
//
//   Components: A={0,1}, B={2,3}, C={4,5}.
//   Q: triangle A-B-C (3 edges). Min A-C cut = 2.
template<template<typename> class VectorType>
void test_triangle_quotient()
{
    std::cout << "test_triangle_quotient ... ";

    auto base = make_graph<VectorType>(6,
        {0, 2, 4, 0, 2, 0},
        {1, 3, 5, 2, 4, 4},
        {5.0f, 5.0f, 5.0f, -1.0f, -1.0f, -1.0f});
    auto lifted = make_graph<VectorType>(6, {1}, {5}, {8.0f});

    auto factors = find_lifted_cut_constraints<VectorType>(base, lifted);

    test(factors.num_factors == 1, "expected 1 factor");

    thrust::host_vector<int> bo(factors.base_offsets);
    thrust::host_vector<int> lo(factors.lifted_offsets);
    test(bo[1] - bo[0] == 2, "expected cut_size=2");
    test(lo[1] - lo[0] == 1, "expected 1 lifted edge");

    validate_factors(factors, base, lifted, "triangle_quotient");

    std::cout << "passed\n";
}

// Test 5: No violations -- single component (all base edges non-negative).
template<template<typename> class VectorType>
void test_no_violations_single_component()
{
    std::cout << "test_no_violations_single_component ... ";

    auto base = make_graph<VectorType>(3, {0, 1}, {1, 2}, {3.0f, 5.0f});
    auto lifted = make_graph<VectorType>(3, {0}, {2}, {10.0f});

    auto factors = find_lifted_cut_constraints<VectorType>(base, lifted);
    test(factors.num_factors == 0, "expected no factors");

    std::cout << "passed\n";
}

// Test 6: Empty quotient -- components exist but no negative inter-component edges.
template<template<typename> class VectorType>
void test_empty_quotient()
{
    std::cout << "test_empty_quotient ... ";

    auto base = make_graph<VectorType>(3, {0}, {1}, {5.0f});
    auto lifted = make_graph<VectorType>(3, {0}, {2}, {10.0f});

    auto factors = find_lifted_cut_constraints<VectorType>(base, lifted);
    test(factors.num_factors == 0, "expected no factors (empty Q)");

    std::cout << "passed\n";
}

// Test 7: Multiple lifted edges sharing the same component pair → single factor.
//
//         +10             +7
//     .............   .........
//     :           v   :       v
//   +---+  +5   +---+  -3   +---+
//   | 0 | ----> | 1 | ----> | 2 |
//   +---+       +---+       +---+
//
//   Both lifted edges cross components {0,1} vs {2}. 1 factor, 2 lifted edges.
template<template<typename> class VectorType>
void test_multiple_violations_same_pair()
{
    std::cout << "test_multiple_violations_same_pair ... ";

    auto base = make_graph<VectorType>(3, {0, 1}, {1, 2}, {5.0f, -3.0f});
    auto lifted = make_graph<VectorType>(3, {0, 1}, {2, 2}, {10.0f, 7.0f});

    auto factors = find_lifted_cut_constraints<VectorType>(base, lifted);

    test(factors.num_factors == 1, "expected 1 factor");

    thrust::host_vector<int> bo(factors.base_offsets);
    thrust::host_vector<int> lo(factors.lifted_offsets);
    test(bo[1] - bo[0] == 1, "expected 1 base edge in cut");
    test(lo[1] - lo[0] == 2, "expected 2 lifted edges");

    validate_factors(factors, base, lifted, "multiple_violations_same_pair");

    std::cout << "passed\n";
}

// Test 8: Empty lifted graph.
template<template<typename> class VectorType>
void test_empty_lifted()
{
    std::cout << "test_empty_lifted ... ";

    auto base = make_graph<VectorType>(3, {0, 1}, {1, 2}, {3.0f, -5.0f});
    Graph<VectorType> lifted;

    auto factors = find_lifted_cut_constraints<VectorType>(base, lifted);
    test(factors.num_factors == 0, "expected no factors for empty lifted");

    std::cout << "passed\n";
}

// Test 9: No positive lifted edges -- no violations.
template<template<typename> class VectorType>
void test_no_positive_lifted()
{
    std::cout << "test_no_positive_lifted ... ";

    auto base = make_graph<VectorType>(3, {0, 1}, {1, 2}, {5.0f, -3.0f});
    auto lifted = make_graph<VectorType>(3, {0}, {2}, {-2.0f});

    auto factors = find_lifted_cut_constraints<VectorType>(base, lifted);
    test(factors.num_factors == 0, "expected no factors for non-positive lifted");

    std::cout << "passed\n";
}

// Test 10: Random graphs -- validate structural properties of all factors.
template<template<typename> class VectorType>
void test_random_validity()
{
    std::cout << "test_random_validity ... ";

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

        auto factors = find_lifted_cut_constraints<VectorType>(base, lifted);

        validate_factors(factors, base, lifted,
            "random(n=" + std::to_string(n) + ",seed=" + std::to_string(seed) + ")");
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
    test_random_validity<VectorType>();
}
