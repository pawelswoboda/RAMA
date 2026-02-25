#pragma once

#include "rama_solver.h"
#include "random_graph.h"
#include "test.h"
#include <vector>
#include <set>
#include <cmath>
#include <iostream>

template<template<typename> class VectorType>
void test_lifted_path_with_repulsive_lifted_edge()
{
    //       -10 (lifted)
    //   .........................
    //   :                       v
    // +---+  +3   +---+  +3   +---+
    // | 0 | ----> | 1 | ----> | 2 |
    // +---+       +---+       +---+
    //
    // Strong repulsive lifted edge wants 0 and 2 separated.
    // Cutting one base edge costs +3, cutting lifted gains -10.
    // Optimal: cut one base edge, obj = -7.
    std::vector<int> base_tails = {0, 1};
    std::vector<int> base_heads = {1, 2};
    std::vector<float> base_costs = {3.0f, 3.0f};

    std::vector<int> lifted_tails = {0};
    std::vector<int> lifted_heads = {2};
    std::vector<float> lifted_costs = {-10.0f};

    Graph<VectorType> base_G(base_tails.begin(), base_tails.end(),
                             base_heads.begin(), base_heads.end(),
                             base_costs.begin(), base_costs.end());
    Graph<VectorType> lifted_G(lifted_tails.begin(), lifted_tails.end(),
                               lifted_heads.begin(), lifted_heads.end(),
                               lifted_costs.begin(), lifted_costs.end());

    multicut_solver_options opts("PD");
    opts.verbose = false;

    VectorType<int> node_mapping;
    double lb;
    std::vector<std::vector<int>> timeline;
    std::tie(node_mapping, lb, timeline) = rama_solver<VectorType>(base_G, lifted_G, opts);

    test(node_mapping.size() == 3, "mapping size should be 3");

    std::vector<int> m(3);
    thrust::copy(node_mapping.begin(), node_mapping.end(), m.begin());

    // No cut factors should fire (lifted edge is repulsive, all base edges positive).
    // Verify by comparing LB against standard solver on the union graph.
    {
        std::vector<int> ut = {0, 0, 1}, uh = {1, 2, 2};
        std::vector<float> uc = {3.0f, -10.0f, 3.0f};
        Graph<VectorType> union_G(ut.begin(), ut.end(), uh.begin(), uh.end(), uc.begin(), uc.end());
        auto [std_mapping, std_lb, std_tl] = rama_solver<VectorType>(union_G, opts);
        test(std::abs(lb - std_lb) < 1e-4,
             "no cut factors: lifted LB should equal standard LB");
    }

    // Nodes 0 and 2 must be separated (repulsive lifted edge dominates)
    test(m[0] != m[2], "repulsive lifted edge (0,2): nodes 0 and 2 should be in different clusters");

    // Compute objective
    double obj = 0;
    for (size_t e = 0; e < base_tails.size(); ++e)
        if (m[base_tails[e]] != m[base_heads[e]])
            obj += base_costs[e];
    for (size_t e = 0; e < lifted_tails.size(); ++e)
        if (m[lifted_tails[e]] != m[lifted_heads[e]])
            obj += lifted_costs[e];

    test(std::abs(obj - (-7.0)) < 1e-4, "optimal objective should be -7");
}

template<template<typename> class VectorType>
void test_lifted_attractive_prevents_cuts()
{
    //       +20 (lifted)
    //   .........................
    //   :                       v
    // +---+  -5   +---+  -5   +---+
    // | 0 | ----> | 1 | ----> | 2 |
    // +---+       +---+       +---+
    //
    // Repulsive base edges want to separate, but cutting any base edge
    // always separates 0 and 2 (path graph), incurring +20 lifted cost.
    // Optimal: no cuts, obj = 0.
    //
    // The cut factor (1 base edge + lifted edge) tightens the LP bound
    // from -10 to -5. The solver may not reach the true optimum because
    // each Karger min-cut involves only one of the two base edges, so the
    // primal heuristic can only make one base edge positive at a time.
    std::vector<int> base_tails = {0, 1};
    std::vector<int> base_heads = {1, 2};
    std::vector<float> base_costs = {-5.0f, -5.0f};

    std::vector<int> lifted_tails = {0};
    std::vector<int> lifted_heads = {2};
    std::vector<float> lifted_costs = {20.0f};

    Graph<VectorType> base_G(base_tails.begin(), base_tails.end(),
                             base_heads.begin(), base_heads.end(),
                             base_costs.begin(), base_costs.end());
    Graph<VectorType> lifted_G(lifted_tails.begin(), lifted_tails.end(),
                               lifted_heads.begin(), lifted_heads.end(),
                               lifted_costs.begin(), lifted_costs.end());

    multicut_solver_options opts("PD");
    opts.num_outer_itr_dual = 5;
    opts.verbose = false;

    VectorType<int> node_mapping;
    double lb;
    std::vector<std::vector<int>> timeline;
    std::tie(node_mapping, lb, timeline) = rama_solver<VectorType>(base_G, lifted_G, opts);

    test(node_mapping.size() == 3, "mapping size should be 3");

    std::vector<int> m(3);
    thrust::copy(node_mapping.begin(), node_mapping.end(), m.begin());

    // Cut factors make one base edge positive, contraction merges those nodes.
    // The remaining base+lifted edges sum to +15, so the second contraction
    // merges everything. Optimal: all nodes together, obj = 0.
    test(m[0] == m[1] && m[1] == m[2],
         "attractive lifted edge should keep all nodes together");

    double obj = 0;
    for (size_t e = 0; e < base_tails.size(); ++e)
        if (m[base_tails[e]] != m[base_heads[e]])
            obj += base_costs[e];
    for (size_t e = 0; e < lifted_tails.size(); ++e)
        if (m[lifted_tails[e]] != m[lifted_heads[e]])
            obj += lifted_costs[e];

    test(std::abs(obj) < 1e-4, "optimal objective should be 0");
    test(std::abs(lb) < 1e-4, "lower bound should be 0");
}

template<template<typename> class VectorType>
void test_lifted_4node_attractive_prevents_cuts()
{
    //       +25 (lifted)
    //   .........................
    //   :                       v
    // +---+  -5   +---+  -5   +---+  -5   +---+
    // | 0 | ----> | 1 | ----> | 2 | ----> | 3 |
    // +---+       +---+       +---+       +---+
    //   ^   -5                              |
    //   +-----------------------------------+
    //
    // Base: 4-cycle (0,1), (1,2), (2,3), (0,3), all costs -5.
    // Lifted: diagonal (0,2) = +25.
    // Two triangles in the union graph: (0,1,2) and (0,2,3).
    //
    // Without lifted: optimal separates all nodes, obj = -20.
    // With lifted: separating 0 from 2 costs +25, so the solver
    // keeps 0 and 2 together. Optimal: {0,1,2}|{3} or {0,2,3}|{1},
    // obj = -10.
    std::vector<int> base_tails = {0, 1, 2, 0};
    std::vector<int> base_heads = {1, 2, 3, 3};
    std::vector<float> base_costs = {-5.0f, -5.0f, -5.0f, -5.0f};

    std::vector<int> lifted_tails = {0};
    std::vector<int> lifted_heads = {2};
    std::vector<float> lifted_costs = {25.0f};

    Graph<VectorType> base_G(base_tails.begin(), base_tails.end(),
                             base_heads.begin(), base_heads.end(),
                             base_costs.begin(), base_costs.end());
    Graph<VectorType> lifted_G(lifted_tails.begin(), lifted_tails.end(),
                               lifted_heads.begin(), lifted_heads.end(),
                               lifted_costs.begin(), lifted_costs.end());

    multicut_solver_options opts("PD");
    opts.num_outer_itr_dual = 5;
    opts.verbose = false;

    VectorType<int> node_mapping;
    double lb;
    std::vector<std::vector<int>> timeline;
    std::tie(node_mapping, lb, timeline) = rama_solver<VectorType>(base_G, lifted_G, opts);

    test(node_mapping.size() == 4, "4node: mapping size should be 4");

    std::vector<int> m(4);
    thrust::copy(node_mapping.begin(), node_mapping.end(), m.begin());

    // Attractive lifted edge should keep 0 and 2 in the same cluster
    test(m[0] == m[2],
         "4node: attractive lifted edge (0,2) should keep nodes 0 and 2 together");

    // The mapping must correspond to one of the two optimal partitions:
    //   {0,1,2}|{3}: m[0]==m[1]==m[2], m[3] different
    //   {0,2,3}|{1}: m[0]==m[2]==m[3], m[1] different
    const bool is_opt1 = (m[0] == m[1]) && (m[1] == m[2]) && (m[2] != m[3]);
    const bool is_opt2 = (m[0] == m[2]) && (m[2] == m[3]) && (m[0] != m[1]);
    test(is_opt1 || is_opt2, "4node: mapping should be {0,1,2}|{3} or {0,2,3}|{1}");

    double obj = 0;
    for (size_t e = 0; e < base_tails.size(); ++e)
        if (m[base_tails[e]] != m[base_heads[e]])
            obj += base_costs[e];
    for (size_t e = 0; e < lifted_tails.size(); ++e)
        if (m[lifted_tails[e]] != m[lifted_heads[e]])
            obj += lifted_costs[e];

    test(std::abs(obj - (-10.0)) < 1e-4, "4node: optimal objective should be -10 but is " + std::to_string(obj));
    test(std::abs(lb - (-10.0)) < 1e-4, "4node: lower bound should be -10 but is " + std::to_string(lb));
}

template<template<typename> class VectorType>
void test_lifted_6node_two_lifted_edges()
{
    //   0 .............. +50 (lifted) .............. 4
    //   :                   +---+                   :
    //   :             -5  / | 2 | \  -5             :
    //   :               /   +---+   \               :
    // +---+  -5   +---+                 +---+  -5   +---+
    // | 0 | ----> | 1 |                 | 4 | ----> | 5 |
    // +---+       +---+                 +---+       +---+
    //               :     \   +---+   /                :
    //               :   -5  \ | 3 | /  -5              :
    //               :         +---+                    :
    //               :......... +50 (lifted) ...........:
    //
    // Base: (0,1), (1,2), (1,3), (2,4), (3,4), (4,5), all costs -5.
    // Lifted: (0,4) = +50, (1,5) = +50.
    //
    // Strong lifted edges keep 0 with 4 and 1 with 5.
    // Nodes 2 and 3 can each be isolated (2 edges each) without
    // separating any lifted pair. Optimal: isolate one of {2} or {3},
    // e.g. {0,1,3,4,5}|{2} or {0,1,2,4,5}|{3}, obj = -10.
    std::vector<int> base_tails = {0, 1, 1, 2, 3, 4};
    std::vector<int> base_heads = {1, 2, 3, 4, 4, 5};
    std::vector<float> base_costs = {-5.0f, -5.0f, -5.0f, -5.0f, -5.0f, -5.0f};

    std::vector<int> lifted_tails = {0, 1};
    std::vector<int> lifted_heads = {4, 5};
    std::vector<float> lifted_costs = {50.0f, 50.0f};

    Graph<VectorType> base_G(base_tails.begin(), base_tails.end(),
                             base_heads.begin(), base_heads.end(),
                             base_costs.begin(), base_costs.end());
    Graph<VectorType> lifted_G(lifted_tails.begin(), lifted_tails.end(),
                               lifted_heads.begin(), lifted_heads.end(),
                               lifted_costs.begin(), lifted_costs.end());

    multicut_solver_options opts("PD");
    opts.num_outer_itr_dual = 5;
    opts.verbose = false;

    VectorType<int> node_mapping;
    double lb;
    std::vector<std::vector<int>> timeline;
    std::tie(node_mapping, lb, timeline) = rama_solver<VectorType>(base_G, lifted_G, opts);

    test(node_mapping.size() == 6, "6node: mapping size should be 6");

    std::vector<int> m(6);
    thrust::copy(node_mapping.begin(), node_mapping.end(), m.begin());

    // Lifted edges should keep their endpoints together
    test(m[0] == m[4],
         "6node: lifted edge (0,4) should keep nodes 0 and 4 together");
    test(m[1] == m[5],
         "6node: lifted edge (1,5) should keep nodes 1 and 5 together");

    double obj = 0;
    for (size_t e = 0; e < base_tails.size(); ++e)
        if (m[base_tails[e]] != m[base_heads[e]])
            obj += base_costs[e];
    for (size_t e = 0; e < lifted_tails.size(); ++e)
        if (m[lifted_tails[e]] != m[lifted_heads[e]])
            obj += lifted_costs[e];

    test(std::abs(obj - (-10.0)) < 1e-4, "6node: optimal objective should be -10");
    test(std::abs(lb - (-10.0)) < 1e-4, "6node: optimal lower bound should be -10");
}

template<template<typename> class VectorType>
void test_lifted_random_graph()
{
    const int n = 30;
    RandomGraph rg = generate_random_graph(n, 0.3, 42);
    if (rg.tails.empty())
        return;

    // Build a set of base edges for fast lookup
    std::set<std::pair<int,int>> base_edge_set;
    for (size_t e = 0; e < rg.tails.size(); ++e)
        base_edge_set.insert({rg.tails[e], rg.heads[e]});

    // Generate lifted edges between non-adjacent node pairs
    std::mt19937 lift_rng(99);
    std::uniform_real_distribution<float> cost_dist(-1.0f, 1.0f);
    std::vector<int> lifted_tails, lifted_heads;
    std::vector<float> lifted_costs;
    for (int i = 0; i < n; ++i)
    {
        for (int j = i + 1; j < n; ++j)
        {
            if (base_edge_set.count({i, j}) == 0)
            {
                if (std::uniform_real_distribution<double>(0.0, 1.0)(lift_rng) < 0.1)
                {
                    lifted_tails.push_back(i);
                    lifted_heads.push_back(j);
                    lifted_costs.push_back(cost_dist(lift_rng));
                }
            }
        }
    }

    Graph<VectorType> base_G(rg.num_nodes,
                             rg.tails.begin(), rg.tails.end(),
                             rg.heads.begin(), rg.heads.end(),
                             rg.costs.begin(), rg.costs.end());

    Graph<VectorType> lifted_G;
    if (!lifted_tails.empty())
        lifted_G = Graph<VectorType>(n,
                                     lifted_tails.begin(), lifted_tails.end(),
                                     lifted_heads.begin(), lifted_heads.end(),
                                     lifted_costs.begin(), lifted_costs.end());

    multicut_solver_options opts("PD");
    opts.verbose = false;

    VectorType<int> node_mapping;
    double lb;
    std::vector<std::vector<int>> timeline;
    std::tie(node_mapping, lb, timeline) = rama_solver<VectorType>(base_G, lifted_G, opts);

    std::vector<int> m(n);
    thrust::copy(node_mapping.begin(), node_mapping.end(), m.begin());

    test(m.size() == (size_t)n, "mapping size should equal num_nodes");

    // Compute objective
    double obj = 0;
    for (size_t e = 0; e < rg.tails.size(); ++e)
        if (m[rg.tails[e]] != m[rg.heads[e]])
            obj += rg.costs[e];
    for (size_t e = 0; e < lifted_tails.size(); ++e)
        if (m[lifted_tails[e]] != m[lifted_heads[e]])
            obj += lifted_costs[e];

    test(lb <= obj + 1e-4, "lower bound should be <= objective");
}

template<template<typename> class VectorType>
void test_lifted_empty_equals_standard()
{
    //       +2
    //   +------------------------+
    //   v                        |
    // +---+  +3   +---+  -10   +---+
    // | 0 | ----> | 1 | -----> | 2 |
    // +---+       +---+        +---+
    //
    // Same graph as the standard conflicted triangle test.
    // With empty lifted graph, should produce the same partition.
    std::vector<int> tails = {0, 0, 1};
    std::vector<int> heads = {1, 2, 2};
    std::vector<float> costs = {3.0f, 2.0f, -10.0f};

    Graph<VectorType> G(tails.begin(), tails.end(),
                        heads.begin(), heads.end(),
                        costs.begin(), costs.end());
    Graph<VectorType> empty_lifted;

    multicut_solver_options opts("P");
    opts.verbose = false;

    VectorType<int> node_mapping;
    double lb;
    std::vector<std::vector<int>> timeline;
    std::tie(node_mapping, lb, timeline) = rama_solver<VectorType>(G, empty_lifted, opts);

    test(node_mapping.size() == 3, "mapping size should be 3");

    std::vector<int> m(3);
    thrust::copy(node_mapping.begin(), node_mapping.end(), m.begin());

    // Repulsive edge endpoints must be separated (same as standard test)
    test(m[1] != m[2],
         "empty lifted: repulsive edge (1,2) should separate nodes 1 and 2");
}

template<template<typename> class VectorType>
void run_all_lifted_rama_solver_tests()
{
    test_lifted_path_with_repulsive_lifted_edge<VectorType>();
    std::cout << "test_lifted_path_with_repulsive_lifted_edge passed\n";

    test_lifted_attractive_prevents_cuts<VectorType>();
    std::cout << "test_lifted_attractive_prevents_cuts passed\n";

    test_lifted_4node_attractive_prevents_cuts<VectorType>();
    std::cout << "test_lifted_4node_attractive_prevents_cuts passed\n";

    test_lifted_6node_two_lifted_edges<VectorType>();
    std::cout << "test_lifted_6node_two_lifted_edges passed\n";

    test_lifted_random_graph<VectorType>();
    std::cout << "test_lifted_random_graph passed\n";

    test_lifted_empty_equals_standard<VectorType>();
    std::cout << "test_lifted_empty_equals_standard passed\n";

    std::cout << "All lifted_rama_solver tests passed!\n";
}
