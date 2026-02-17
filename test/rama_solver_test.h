#pragma once

#include "rama_solver.h"
#include "random_graph.h"
#include "test.h"
#include <vector>
#include <set>
#include <cmath>
#include <iostream>

template<template<typename> class VectorType>
void test_solver_conflicted_triangle()
{
    //       +2
    //   +-----------------------+
    //   v                       |
    // +---+  +3   +---+  -10  +---+
    // | 0 | ----> | 1 | ----> | 2 |
    // +---+       +---+       +---+
    //
    // Strong repulsive edge (1,2,-10) dominates: cutting it gives obj -8.
    // Expected partition: {0,1}, {2}
    std::vector<int> tails = {0, 0, 1};
    std::vector<int> heads = {1, 2, 2};
    std::vector<float> costs = {3.0f, 2.0f, -10.0f};

    Graph<VectorType> G(tails.begin(), tails.end(),
                        heads.begin(), heads.end(),
                        costs.begin(), costs.end());

    multicut_solver_options opts("P");
    opts.verbose = false;

    VectorType<int> node_mapping;
    double lb;
    std::vector<std::vector<int>> timeline;
    std::tie(node_mapping, lb, timeline) = rama_solver<VectorType>(G, opts);

    test(node_mapping.size() == 3, "mapping size should be 3");
    // Repulsive edge endpoints must be separated
    test(node_mapping[1] != node_mapping[2],
         "repulsive edge (1,2): nodes 1 and 2 should be in different clusters");
}

template<template<typename> class VectorType>
void test_solver_all_positive()
{
    //       +3
    //   +-----------------------+
    //   v                       |
    // +---+  +1   +---+  +2   +---+
    // | 0 | ----> | 1 | ----> | 2 |
    // +---+       +---+       +---+
    //
    // No repulsive edges => everything contracts to 1 cluster
    std::vector<int> tails = {0, 1, 0};
    std::vector<int> heads = {1, 2, 2};
    std::vector<float> costs = {1.0f, 2.0f, 3.0f};

    Graph<VectorType> G(tails.begin(), tails.end(),
                        heads.begin(), heads.end(),
                        costs.begin(), costs.end());

    multicut_solver_options opts("P");
    opts.verbose = false;

    VectorType<int> node_mapping;
    double lb;
    std::vector<std::vector<int>> timeline;
    std::tie(node_mapping, lb, timeline) = rama_solver<VectorType>(G, opts);

    test(node_mapping.size() == 3, "mapping size should be 3");
    test(node_mapping[0] == node_mapping[1] && node_mapping[1] == node_mapping[2],
         "all-positive: all nodes should be in same cluster");
}

template<template<typename> class VectorType>
void test_solver_random_graph()
{
    RandomGraph rg = generate_random_graph(30, 0.3, 42);
    if (rg.tails.empty())
        return;

    Graph<VectorType> G(rg.num_nodes,
                        rg.tails.begin(), rg.tails.end(),
                        rg.heads.begin(), rg.heads.end(),
                        rg.costs.begin(), rg.costs.end());

    multicut_solver_options opts("PD");
    opts.verbose = false;

    VectorType<int> node_mapping;
    double lb;
    std::vector<std::vector<int>> timeline;
    std::tie(node_mapping, lb, timeline) = rama_solver<VectorType>(G, opts);

    const int n = rg.num_nodes;
    std::vector<int> m(n);
    thrust::copy(node_mapping.begin(), node_mapping.end(), m.begin());

    test(m.size() == (size_t)n, "mapping size should equal num_nodes");

    // Labels should be contiguous [0, num_components)
    int max_label = *std::max_element(m.begin(), m.end());
    int min_label = *std::min_element(m.begin(), m.end());
    test(min_label == 0, "min label should be 0");

    std::set<int> unique_labels(m.begin(), m.end());
    int num_components = unique_labels.size();
    test(max_label == num_components - 1, "labels should be contiguous");

    // Compute objective value
    double obj = 0;
    for (size_t e = 0; e < rg.tails.size(); e++)
    {
        if (m[rg.tails[e]] != m[rg.heads[e]])
            obj += rg.costs[e];
    }

    // Lower bound should be <= objective
    test(lb <= obj + 1e-4,
         "lower bound should be <= objective");
}

template<template<typename> class VectorType>
void test_solver_dual_only()
{
    RandomGraph rg = generate_random_graph(20, 0.3, 123);
    if (rg.tails.empty())
        return;

    Graph<VectorType> G(rg.num_nodes,
                        rg.tails.begin(), rg.tails.end(),
                        rg.heads.begin(), rg.heads.end(),
                        rg.costs.begin(), rg.costs.end());

    multicut_solver_options opts("D");
    opts.verbose = false;

    VectorType<int> node_mapping;
    double lb;
    std::vector<std::vector<int>> timeline;
    std::tie(node_mapping, lb, timeline) = rama_solver<VectorType>(G, opts);

    // D mode returns empty node_mapping
    test(node_mapping.size() == 0, "dual-only mode should return empty mapping");
    // LB should be finite
    test(std::isfinite(lb), "lower bound should be finite");
}

template<template<typename> class VectorType>
void run_all_rama_solver_tests()
{
    test_solver_conflicted_triangle<VectorType>();
    std::cout << "test_solver_conflicted_triangle passed\n";

    test_solver_all_positive<VectorType>();
    std::cout << "test_solver_all_positive passed\n";

    test_solver_random_graph<VectorType>();
    std::cout << "test_solver_random_graph passed\n";

    test_solver_dual_only<VectorType>();
    std::cout << "test_solver_dual_only passed\n";

    std::cout << "All rama_solver tests passed!\n";
}