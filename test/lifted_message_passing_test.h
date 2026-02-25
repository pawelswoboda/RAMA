#pragma once

#include "multicut_message_passing.h"
#include "lifted_multicut_utils.h"
#include "lifted_cut_constraints.h"
#include "random_graph.h"
#include "test.h"

#include <iostream>
#include <vector>
#include <cmath>
#include <string>
#include <algorithm>

// Helper: build a symmetric Graph from undirected edge list.
template<template<typename> class VectorType>
Graph<VectorType> lmp_make_graph(int num_nodes,
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

// Test 1: All positive base — no constraint violation.
//
//       +10 (lifted)
//   .........................
//   :                       v
// +---+  +3   +---+  +5   +---+
// | 0 | ----> | 1 | ----> | 2 |
// +---+       +---+       +---+
//
//   All base edges non-negative → single component → no violation.
//   find_lifted_cut_constraints returns 0 factors.
template<template<typename> class VectorType>
void test_cf_no_violation()
{
    std::cout << "test_cf_no_violation ... ";

    auto base = lmp_make_graph<VectorType>(3, {0, 1}, {1, 2}, {3.0f, 5.0f});
    auto lifted = lmp_make_graph<VectorType>(3, {0}, {2}, {10.0f});
    auto union_G = create_union_graph(base, lifted);

    multicut_message_passing<VectorType> mp(union_G, false);
    auto factors = find_lifted_cut_constraints<VectorType>(base, lifted);
    test(factors.num_factors == 0, "no violation should produce 0 factors");

    int added = mp.add_cut_factors(factors, base, lifted);
    test(added == 0, "add_cut_factors should return 0");

    double lb = mp.lower_bound();
    test(std::abs(lb) < 1e-6, "all-positive: lb should be 0, got " + std::to_string(lb));

    std::cout << "passed\n";
}

// Test 2: Single cut factor with 1 base edge, 1 lifted edge.
//
//         +10 (lifted)
//     .........................
//     :                       v
//   +---+  +3   +---+  -5   +---+
//   | 0 | ----> | 1 | ----> | 2 |
//   +---+       +---+       +---+
//
//   Components: {0,1}, {2}. Cut: {(1,2)}.
//   Factor: base edge (1,2), lifted edge (0,2).
//   Union graph has edges: (0,1,+3), (0,2,+10), (1,2,-5).
//
//   Without the factor, trivial LB = -5.
//   The cut factor constraint says: if (1,2) is cut, then (0,2) must be cut.
//   Cutting (1,2) costs -5, but forces cutting (0,2) at cost +10 → net +5.
//   Not cutting (1,2) costs 0 → total 0.
//   So the factor should tighten the bound from -5 toward 0.
template<template<typename> class VectorType>
void test_cf_single_base_single_lifted()
{
    std::cout << "test_cf_single_base_single_lifted ... ";

    auto base = lmp_make_graph<VectorType>(3, {0, 1}, {1, 2}, {3.0f, -5.0f});
    auto lifted = lmp_make_graph<VectorType>(3, {0}, {2}, {10.0f});
    auto union_G = create_union_graph(base, lifted);

    multicut_message_passing<VectorType> mp(union_G, false);
    auto factors = find_lifted_cut_constraints<VectorType>(base, lifted);
    test(factors.num_factors == 1, "expected 1 factor");

    mp.add_cut_factors(factors, base, lifted);

    double initial_lb = mp.lower_bound();
    test(initial_lb <= -5.0 + 1e-6,
         "initial lb should be <= -5, got " + std::to_string(initial_lb));

    for (int iter = 0; iter < 20; ++iter)
        mp.iteration();

    double final_lb = mp.lower_bound();
    test(final_lb >= initial_lb - 1e-6,
         "lb should not decrease, initial=" + std::to_string(initial_lb) +
         ", final=" + std::to_string(final_lb));
    test(final_lb > -5.0 + 0.1,
         "lb should improve significantly from -5, got " + std::to_string(final_lb));

    std::cout << "passed (lb: " << initial_lb << " -> " << final_lb << ")\n";
}

// Test 3: Cut factor with 2 base edges, 1 lifted edge.
//
//           +10 (lifted)
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
//   Components: {0,1}, {2,3}. Cut: {(0,2), (1,3)}.
//   Factor: 2 base edges, 1 lifted edge (0,3,+10).
//   Constraint: if both base edges cut → lifted must be cut.
//   Trivial LB = -2 (cutting both base edges). With factor: cutting both
//   forces lifted cut (+10) → net +8. Better to cut only 1 → net -1.
//   Factor should tighten toward -1.
template<template<typename> class VectorType>
void test_cf_two_base_one_lifted()
{
    std::cout << "test_cf_two_base_one_lifted ... ";

    auto base = lmp_make_graph<VectorType>(4,
        {0, 2, 0, 1}, {1, 3, 2, 3}, {5.0f, 5.0f, -1.0f, -1.0f});
    auto lifted = lmp_make_graph<VectorType>(4, {0}, {3}, {10.0f});
    auto union_G = create_union_graph(base, lifted);

    multicut_message_passing<VectorType> mp(union_G, false);
    auto factors = find_lifted_cut_constraints<VectorType>(base, lifted);
    test(factors.num_factors == 1, "expected 1 factor");

    thrust::host_vector<int> bo(factors.base_offsets);
    test(bo[1] - bo[0] == 2, "expected 2 base edges in factor");

    mp.add_cut_factors(factors, base, lifted);

    double initial_lb = mp.lower_bound();

    for (int iter = 0; iter < 20; ++iter)
        mp.iteration();

    double final_lb = mp.lower_bound();
    test(final_lb >= initial_lb - 1e-6,
         "lb should not decrease");
    test(final_lb > initial_lb + 0.01,
         "lb should improve from " + std::to_string(initial_lb) +
         ", got " + std::to_string(final_lb));

    std::cout << "passed (lb: " << initial_lb << " -> " << final_lb << ")\n";
}

// Test 4: Cut factor with 1 base edge, 2 lifted edges.
//
//         +10 (lifted)           +7 (lifted)
//     .........................   .........
//     :                       :           v
//   +---+  +5   +---+  +5   +---+  -3   +---+
//   | 0 | ----> | 1 | ----> | 2 | ----> | 3 |
//   +---+       +---+       +---+       +---+
//
//   Base: (0,1,+5), (1,2,+5), (2,3,-3).
//   Lifted: (0,3,+10), (1,3,+7).
//   Components: {0,1,2}, {3}. Cut: {(2,3)}.
//   Factor: 1 base edge (2,3), 2 lifted edges (0,3) and (1,3).
//   Cutting base (2,3) costs -3 but forces both lifted edges cut (+17) → net +14.
//   Not cutting: 0. Factor should tighten toward 0.
template<template<typename> class VectorType>
void test_cf_one_base_two_lifted()
{
    std::cout << "test_cf_one_base_two_lifted ... ";

    auto base = lmp_make_graph<VectorType>(4,
        {0, 1, 2}, {1, 2, 3}, {5.0f, 5.0f, -3.0f});
    auto lifted = lmp_make_graph<VectorType>(4,
        {0, 1}, {3, 3}, {10.0f, 7.0f});
    auto union_G = create_union_graph(base, lifted);

    multicut_message_passing<VectorType> mp(union_G, false);
    auto factors = find_lifted_cut_constraints<VectorType>(base, lifted);
    test(factors.num_factors == 1, "expected 1 factor");

    thrust::host_vector<int> lo(factors.lifted_offsets);
    test(lo[1] - lo[0] == 2, "expected 2 lifted edges in factor");

    mp.add_cut_factors(factors, base, lifted);

    double initial_lb = mp.lower_bound();

    for (int iter = 0; iter < 20; ++iter)
        mp.iteration();

    double final_lb = mp.lower_bound();
    test(final_lb >= initial_lb - 1e-6, "lb should not decrease");

    // Cutting base edge (2,3) costs -3 but forces both lifted edges to be cut
    // (+10 + 7 = +17) → net +14. Not cutting: 0. Factor should tighten to 0.
    test(final_lb > -3.0 + 0.1,
         "lb should tighten well beyond -3, got " + std::to_string(final_lb));

    std::cout << "passed (lb: " << initial_lb << " -> " << final_lb << ")\n";
}

// Test 5: Lower bound is non-decreasing across iterations with cut factors.
template<template<typename> class VectorType>
void test_cf_lb_non_decreasing()
{
    std::cout << "test_cf_lb_non_decreasing ... ";

    auto base = lmp_make_graph<VectorType>(3, {0, 1}, {1, 2}, {3.0f, -5.0f});
    auto lifted = lmp_make_graph<VectorType>(3, {0}, {2}, {10.0f});
    auto union_G = create_union_graph(base, lifted);

    multicut_message_passing<VectorType> mp(union_G, false);
    auto factors = find_lifted_cut_constraints<VectorType>(base, lifted);
    mp.add_cut_factors(factors, base, lifted);

    double prev_lb = mp.lower_bound();
    for (int iter = 0; iter < 50; ++iter)
    {
        mp.iteration();
        double lb = mp.lower_bound();
        test(lb >= prev_lb - 1e-5,
             "lb must be non-decreasing, iter " + std::to_string(iter) +
             ": " + std::to_string(prev_lb) + " -> " + std::to_string(lb));
        prev_lb = lb;
    }

    std::cout << "passed (final lb: " << prev_lb << ")\n";
}

// Test 6: Cut factors + triangles together.
//
//   Base: (0,1,+3), (1,2,-5), (0,2,+1)  — triangle (0,1,2) exists
//   Lifted: (0,2,+10)  — but (0,2) is already a base edge, so union merges them.
//
//   Let's use 4 nodes to keep base and lifted edges separate:
//
//     +---+  +3   +---+  -5   +---+  +2   +---+
//     | 0 | ----> | 1 | ----> | 2 | ----> | 3 |
//     +---+       +---+       +---+       +---+
//       |                                   ^
//       +...............+10 (lifted)........+
//
//   Plus add base edge (0,2,-1) to create triangle (0,1,2).
//   Components on non-neg: {0,1} (via 0-1), {2,3} (via 2-3).
//   Cut factor: base (1,2) and (0,2), lifted (0,3).
//   Triangle: (0,1,2) with edges (0,1)=+3, (0,2)=-1, (1,2)=-5.
//   Both factors share edge (0,2) and (1,2) → edge_counter reflects both.
template<template<typename> class VectorType>
void test_cf_with_triangles()
{
    std::cout << "test_cf_with_triangles ... ";

    auto base = lmp_make_graph<VectorType>(4,
        {0, 0, 1, 2}, {1, 2, 2, 3}, {3.0f, -1.0f, -5.0f, 2.0f});
    auto lifted = lmp_make_graph<VectorType>(4, {0}, {3}, {10.0f});
    auto union_G = create_union_graph(base, lifted);

    // Add triangle (0,1,2) first
    VectorType<int> t1 = std::vector<int>{0};
    VectorType<int> t2 = std::vector<int>{1};
    VectorType<int> t3 = std::vector<int>{2};
    multicut_message_passing<VectorType> mp(union_G, false);
    mp.add_triangles(std::move(t1), std::move(t2), std::move(t3));

    // Then add cut factors
    auto factors = find_lifted_cut_constraints<VectorType>(base, lifted);
    mp.add_cut_factors(factors, base, lifted);

    double initial_lb = mp.lower_bound();

    double prev_lb = initial_lb;
    for (int iter = 0; iter < 50; ++iter)
    {
        mp.iteration();
        double lb = mp.lower_bound();
        test(lb >= prev_lb - 1e-5,
             "triangles+cf: lb must be non-decreasing, iter " + std::to_string(iter) +
             ": " + std::to_string(prev_lb) + " -> " + std::to_string(lb));
        prev_lb = lb;
    }

    double final_lb = prev_lb;
    test(final_lb > initial_lb + 0.01,
         "combined triangle+cf should improve lb from " + std::to_string(initial_lb));

    std::cout << "passed (lb: " << initial_lb << " -> " << final_lb << ")\n";
}

// Test 7: Cut factor lower bound contribution.
//
//   Same setup as test 2. After distributing costs to the factor once,
//   verify cut_factor_lower_bound is consistent with overall lower bound.
template<template<typename> class VectorType>
void test_cf_lower_bound_decomposition()
{
    std::cout << "test_cf_lower_bound_decomposition ... ";

    auto base = lmp_make_graph<VectorType>(3, {0, 1}, {1, 2}, {3.0f, -5.0f});
    auto lifted = lmp_make_graph<VectorType>(3, {0}, {2}, {10.0f});
    auto union_G = create_union_graph(base, lifted);

    multicut_message_passing<VectorType> mp(union_G, false);
    auto factors = find_lifted_cut_constraints<VectorType>(base, lifted);
    mp.add_cut_factors(factors, base, lifted);

    // Run some iterations
    for (int iter = 0; iter < 10; ++iter)
        mp.iteration();

    double total_lb = mp.lower_bound();
    double edge_lb = mp.edge_lower_bound();
    double tri_lb = mp.triangle_lower_bound();
    double cf_lb = mp.cut_factor_lower_bound();

    test(std::abs(total_lb - (edge_lb + tri_lb + cf_lb)) < 1e-4,
         "total LB should equal edge + triangle + cut factor LB, got " +
         std::to_string(total_lb) + " vs " +
         std::to_string(edge_lb) + " + " + std::to_string(tri_lb) + " + " +
         std::to_string(cf_lb));

    std::cout << "passed (edge=" << edge_lb << ", tri=" << tri_lb
              << ", cf=" << cf_lb << ", total=" << total_lb << ")\n";
}

// Test 8: Empty cut factors don't affect standard message passing.
template<template<typename> class VectorType>
void test_cf_empty_factors_no_effect()
{
    std::cout << "test_cf_empty_factors_no_effect ... ";

    VectorType<int> ei = std::vector<int>{0, 0, 1};
    VectorType<int> ej = std::vector<int>{1, 2, 2};
    VectorType<float> ec = std::vector<float>{1.0f, 1.0f, -1.0f};
    Graph<VectorType> G(ei.begin(), ei.end(), ej.begin(), ej.end(),
                        ec.begin(), ec.end());

    // MP without cut factors
    VectorType<int> t1a = std::vector<int>{0};
    VectorType<int> t2a = std::vector<int>{1};
    VectorType<int> t3a = std::vector<int>{2};
    multicut_message_passing<VectorType> mp_ref(G, std::move(t1a), std::move(t2a),
                                                 std::move(t3a), false);

    // MP with empty cut factors
    VectorType<int> t1b = std::vector<int>{0};
    VectorType<int> t2b = std::vector<int>{1};
    VectorType<int> t3b = std::vector<int>{2};
    multicut_message_passing<VectorType> mp_cf(G, std::move(t1b), std::move(t2b),
                                                std::move(t3b), false);
    LiftedCutFactors<VectorType> empty_factors;
    Graph<VectorType> empty_base, empty_lifted;
    mp_cf.add_cut_factors(empty_factors, empty_base, empty_lifted);

    for (int iter = 0; iter < 10; ++iter)
    {
        mp_ref.iteration();
        mp_cf.iteration();
    }

    double lb_ref = mp_ref.lower_bound();
    double lb_cf = mp_cf.lower_bound();
    test(std::abs(lb_ref - lb_cf) < 1e-4,
         "empty factors should not change lb: ref=" + std::to_string(lb_ref) +
         ", cf=" + std::to_string(lb_cf));

    std::cout << "passed\n";
}

// Test 9: Random graphs with cut factors — verify lb is non-decreasing.
template<template<typename> class VectorType>
void test_cf_random_lb_non_decreasing()
{
    std::cout << "test_cf_random_lb_non_decreasing ... ";

    const int sizes[] = {10, 20, 30};
    const double base_probs[] = {0.2, 0.4};
    const double lifted_probs[] = {0.15, 0.3};

    unsigned seed = 200;
    int num_tested = 0;
    for (int n : sizes)
    for (double bp : base_probs)
    for (double lp : lifted_probs)
    {
        auto rg = generate_random_graph(n, bp, lp, seed++);
        if (rg.tails.empty() || rg.lifted_tails.empty())
            continue;

        auto base = lmp_make_graph<VectorType>(rg.num_nodes,
            rg.tails, rg.heads, rg.costs);
        auto lifted = lmp_make_graph<VectorType>(rg.num_nodes,
            rg.lifted_tails, rg.lifted_heads, rg.lifted_costs);
        auto union_G = create_union_graph(base, lifted);

        multicut_message_passing<VectorType> mp(union_G, false);
        auto factors = find_lifted_cut_constraints<VectorType>(base, lifted);
        if (factors.num_factors == 0)
            continue;

        mp.add_cut_factors(factors, base, lifted);

        double prev_lb = mp.lower_bound();
        for (int iter = 0; iter < 30; ++iter)
        {
            mp.iteration();
            double lb = mp.lower_bound();
            test(lb >= prev_lb - 1e-4,
                 "random cf (n=" + std::to_string(n) + ",seed=" +
                 std::to_string(seed) + "): lb decreased at iter " +
                 std::to_string(iter) + ": " + std::to_string(prev_lb) +
                 " -> " + std::to_string(lb));
            prev_lb = lb;
        }
        num_tested++;
    }

    test(num_tested > 0, "should test at least one random instance");
    std::cout << "passed (" << num_tested << " instances)\n";
}

// Test 10: Random graphs — cut factors should not worsen the lower bound
// compared to triangles-only message passing.
template<template<typename> class VectorType>
void test_cf_random_improves_or_matches_triangles()
{
    std::cout << "test_cf_random_improves_or_matches_triangles ... ";

    const int sizes[] = {15, 25};
    const double base_probs[] = {0.3, 0.5};
    const double lifted_probs[] = {0.2, 0.4};

    unsigned seed = 300;
    int num_tested = 0;
    for (int n : sizes)
    for (double bp : base_probs)
    for (double lp : lifted_probs)
    {
        auto rg = generate_random_graph(n, bp, lp, seed++);
        if (rg.tails.empty() || rg.lifted_tails.empty())
            continue;

        auto base = lmp_make_graph<VectorType>(rg.num_nodes,
            rg.tails, rg.heads, rg.costs);
        auto lifted = lmp_make_graph<VectorType>(rg.num_nodes,
            rg.lifted_tails, rg.lifted_heads, rg.lifted_costs);
        auto union_G = create_union_graph(base, lifted);

        auto factors = find_lifted_cut_constraints<VectorType>(base, lifted);
        if (factors.num_factors == 0)
            continue;

        // Reference: triangles only (no cut factors)
        multicut_message_passing<VectorType> mp_ref(union_G, false);
        for (int iter = 0; iter < 30; ++iter)
            mp_ref.iteration();
        double lb_ref = mp_ref.lower_bound();

        // With cut factors
        multicut_message_passing<VectorType> mp_cf(union_G, false);
        mp_cf.add_cut_factors(factors, base, lifted);
        for (int iter = 0; iter < 30; ++iter)
            mp_cf.iteration();
        double lb_cf = mp_cf.lower_bound();

        // Cut factors should not make things worse
        test(lb_cf >= lb_ref - 1e-3,
             "random (n=" + std::to_string(n) + ",seed=" + std::to_string(seed) +
             "): cf lb " + std::to_string(lb_cf) + " < ref lb " + std::to_string(lb_ref));

        num_tested++;
    }

    test(num_tested > 0, "should test at least one random instance");
    std::cout << "passed (" << num_tested << " instances)\n";
}

template<template<typename> class VectorType>
void run_all_lifted_message_passing_tests()
{
    test_cf_no_violation<VectorType>();
    test_cf_single_base_single_lifted<VectorType>();
    test_cf_two_base_one_lifted<VectorType>();
    test_cf_one_base_two_lifted<VectorType>();
    test_cf_lb_non_decreasing<VectorType>();
    test_cf_with_triangles<VectorType>();
    test_cf_lower_bound_decomposition<VectorType>();
    test_cf_empty_factors_no_effect<VectorType>();
    test_cf_random_lb_non_decreasing<VectorType>();
    test_cf_random_improves_or_matches_triangles<VectorType>();

    std::cout << "\nAll lifted message passing tests passed.\n";
}