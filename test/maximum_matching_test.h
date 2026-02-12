#pragma once

#include "maximum_matching.h"
#include "random_graph.h"
#include "test.h"
#include <vector>
#include <set>
#include <iostream>

template<template<typename> class VectorType>
void test_matching_basic()
{
    //       2.0
    //   +-------------------------+
    //   |                         |
    // +---+  1.0   +---+  2.0   +---+  3.0   +---+
    // | 3 | ------ | 0 | ------ | 1 | ------ | 2 |
    // +---+        +---+        +---+        +---+
    //                |   -1.0                  |
    //                +-------------------------+
    //
    // Best matching: {1-2, 0-3} -> 4 matched vertices
    const std::vector<int> i = {0, 1, 0, 3, 3};
    const std::vector<int> j = {1, 2, 2, 0, 1};
    const std::vector<float> costs = {2.f, 3.f, -1.f, 1.f, 2.f};

    Graph<VectorType> A(i.begin(), i.end(), j.begin(), j.end(),
                        costs.begin(), costs.end());

    VectorType<int> node_mapping;
    int nr_matched;
    std::tie(node_mapping, nr_matched) = filter_edges_by_matching(A);

    test(nr_matched == 4, "basic: expected 4 matched vertices, got " + std::to_string(nr_matched));
    test((int)node_mapping.size() == A.num_nodes(), "basic: node_mapping size mismatch");

    thrust::host_vector<int> h_mapping = node_mapping;
    for (size_t v = 0; v < h_mapping.size(); ++v)
        test(h_mapping[v] >= 0 && h_mapping[v] <= (int)v,
             "basic: node_mapping[" + std::to_string(v) + "] = " +
             std::to_string(h_mapping[v]) + " invalid");
}

template<template<typename> class VectorType>
void test_matching_single_edge()
{
    // +---+  5.0   +---+
    // | 0 | ------ | 1 |
    // +---+        +---+
    const std::vector<int> i = {0};
    const std::vector<int> j = {1};
    const std::vector<float> costs = {5.0f};

    Graph<VectorType> A(i.begin(), i.end(), j.begin(), j.end(),
                        costs.begin(), costs.end());

    VectorType<int> node_mapping;
    int nr_matched;
    std::tie(node_mapping, nr_matched) = filter_edges_by_matching(A);

    test(nr_matched == 2, "single edge: expected 2 matched vertices, got " + std::to_string(nr_matched));
    thrust::host_vector<int> h_mapping = node_mapping;
    test(h_mapping[0] == 0, "single edge: node 0 maps to self");
    test(h_mapping[1] == 0, "single edge: node 1 maps to 0");
}

template<template<typename> class VectorType>
void test_matching_chain()
{
    // +---+  1.0   +---+  1.0   +---+  1.0   +---+
    // | 0 | ------ | 1 | ------ | 2 | ------ | 3 |
    // +---+        +---+        +---+        +---+
    //
    // Best matching: {0-1, 2-3} -> 4 matched vertices
    const std::vector<int> i = {0, 1, 2};
    const std::vector<int> j = {1, 2, 3};
    const std::vector<float> costs = {1.0f, 1.0f, 1.0f};

    Graph<VectorType> A(i.begin(), i.end(), j.begin(), j.end(),
                        costs.begin(), costs.end());

    VectorType<int> node_mapping;
    int nr_matched;
    std::tie(node_mapping, nr_matched) = filter_edges_by_matching(A);

    test(nr_matched == 4, "chain: expected 4 matched vertices, got " + std::to_string(nr_matched));
}

template<template<typename> class VectorType>
void test_matching_all_negative()
{
    //       -3.0
    //   +---------------------------+
    //   |                           |
    // +---+  -1.0   +---+  -2.0   +---+
    // | 0 | ------- | 1 | ------- | 2 |
    // +---+         +---+         +---+
    //
    // No positive edges -> nothing matches
    const std::vector<int> i = {0, 1, 0};
    const std::vector<int> j = {1, 2, 2};
    const std::vector<float> costs = {-1.0f, -2.0f, -3.0f};

    Graph<VectorType> A(i.begin(), i.end(), j.begin(), j.end(),
                        costs.begin(), costs.end());

    VectorType<int> node_mapping;
    int nr_matched;
    std::tie(node_mapping, nr_matched) = filter_edges_by_matching(A);

    test(nr_matched == 0, "all negative: expected 0 matched, got " + std::to_string(nr_matched));
}

template<template<typename> class VectorType>
void test_matching_node_mapping_consistency()
{
    // +---+  3.0   +---+  1.0   +---+  5.0   +---+  2.0   +---+
    // | 0 | ------ | 1 | ------ | 2 | ------ | 3 | ------ | 4 |
    // +---+        +---+        +---+        +---+        +---+
    const std::vector<int> i = {0, 1, 2, 3};
    const std::vector<int> j = {1, 2, 3, 4};
    const std::vector<float> costs = {3.0f, 1.0f, 5.0f, 2.0f};

    Graph<VectorType> A(5, i.begin(), i.end(), j.begin(), j.end(),
                        costs.begin(), costs.end());

    VectorType<int> node_mapping;
    int nr_matched;
    std::tie(node_mapping, nr_matched) = filter_edges_by_matching(A);

    thrust::host_vector<int> h_mapping = node_mapping;
    int count_modified = 0;
    for (size_t v = 0; v < h_mapping.size(); ++v)
        if (h_mapping[v] != (int)v)
            count_modified++;

    test(count_modified == nr_matched / 2,
         "consistency: modified entries (" + std::to_string(count_modified) +
         ") != nr_matched/2 (" + std::to_string(nr_matched / 2) + ")");
}

// Verify all matching invariants on the result of filter_edges_by_matching.
template<template<typename> class VectorType>
void verify_matching_invariants(
    const Graph<VectorType>& A,
    const VectorType<int>& node_mapping,
    const int nr_matched,
    const std::string& label)
{
    const int n = A.num_nodes();
    test((int)node_mapping.size() == n,
         label + ": node_mapping size " + std::to_string(node_mapping.size()) +
         " != num_nodes " + std::to_string(n));

    // nr_matched must be even (matched vertices come in pairs)
    test(nr_matched % 2 == 0,
         label + ": nr_matched " + std::to_string(nr_matched) + " is odd");
    test(nr_matched >= 0 && nr_matched <= n,
         label + ": nr_matched " + std::to_string(nr_matched) + " out of range");

    thrust::host_vector<int> h_mapping = node_mapping;

    // Build edge set from graph for edge-existence checks
    thrust::host_vector<int> h_tails = A.get_tails();
    thrust::host_vector<int> h_heads = A.get_heads();
    std::set<std::pair<int,int>> edge_set;
    for (size_t e = 0; e < h_tails.size(); ++e)
        edge_set.insert({h_tails[e], h_heads[e]});

    int count_modified = 0;
    std::vector<int> matched_to(n, -1); // matched_to[u] = v if u is matched to v

    for (int v = 0; v < n; ++v)
    {
        const int m = h_mapping[v];

        // node_mapping[v] <= v (larger always maps to smaller)
        test(m >= 0 && m <= v,
             label + ": node_mapping[" + std::to_string(v) + "] = " +
             std::to_string(m) + " invalid (must be in [0, v])");

        // node_mapping[node_mapping[v]] == node_mapping[v] (root maps to itself)
        test(h_mapping[m] == m,
             label + ": node_mapping[node_mapping[" + std::to_string(v) +
             "]] != node_mapping[" + std::to_string(v) + "]");

        if (m != v)
        {
            count_modified++;

            // Matched edge must exist in the graph
            test(edge_set.count({m, v}) > 0 || edge_set.count({v, m}) > 0,
                 label + ": matched edge (" + std::to_string(m) + ", " +
                 std::to_string(v) + ") not in graph");

            // No two matched edges share a node: check v's partner hasn't
            // been claimed by someone else
            test(matched_to[m] == -1 || matched_to[m] == v,
                 label + ": node " + std::to_string(m) +
                 " matched to both " + std::to_string(matched_to[m]) +
                 " and " + std::to_string(v));
            matched_to[m] = v;
            matched_to[v] = m;
        }
    }

    // nr_matched == 2 * count_modified
    test(nr_matched == 2 * count_modified,
         label + ": nr_matched " + std::to_string(nr_matched) +
         " != 2 * modified " + std::to_string(2 * count_modified));
}

template<template<typename> class VectorType>
void test_matching_random(const int num_nodes, const double density,
                          const unsigned seed = 42)
{
    auto rg = generate_random_graph(num_nodes, density, seed);

    const std::string label = "random(n=" + std::to_string(num_nodes) +
        ", d=" + std::to_string(density) + ", s=" + std::to_string(seed) + ")";

    test(!rg.tails.empty(), label + ": no edges generated");

    bool has_positive = false;
    for (float c : rg.costs)
        if (c > 0.0f) { has_positive = true; break; }

    Graph<VectorType> A(num_nodes, rg.tails.begin(), rg.tails.end(),
                        rg.heads.begin(), rg.heads.end(),
                        rg.costs.begin(), rg.costs.end());

    VectorType<int> node_mapping;
    int nr_matched;
    std::tie(node_mapping, nr_matched) = filter_edges_by_matching(A, 0.0, false);

    verify_matching_invariants<VectorType>(A, node_mapping, nr_matched, label);

    // If there are positive edges, at least one pair should be matched
    if (has_positive)
        test(nr_matched >= 2,
             label + ": has positive edges but nr_matched = " +
             std::to_string(nr_matched));
}

template<template<typename> class VectorType>
void run_all_maximum_matching_tests()
{
    test_matching_basic<VectorType>();
    std::cout << "PASSED: matching basic\n";

    test_matching_single_edge<VectorType>();
    std::cout << "PASSED: matching single edge\n";

    test_matching_chain<VectorType>();
    std::cout << "PASSED: matching chain\n";

    test_matching_all_negative<VectorType>();
    std::cout << "PASSED: matching all negative\n";

    test_matching_node_mapping_consistency<VectorType>();
    std::cout << "PASSED: matching node_mapping consistency\n";

    test_matching_random<VectorType>(30, 0.3, 42);
    std::cout << "PASSED: random graph (n=30, density=0.3)\n";

    test_matching_random<VectorType>(50, 0.2, 123);
    std::cout << "PASSED: random graph (n=50, density=0.2)\n";

    test_matching_random<VectorType>(100, 0.1, 7);
    std::cout << "PASSED: random graph (n=100, density=0.1)\n";

    for (int n : {10, 20, 50})
        for (double d : {0.1, 0.3, 0.5}) {
            test_matching_random<VectorType>(n, d, n * 100 + (int)(d * 10));
            std::cout << "PASSED: random graph (n=" << n << ", density=" << d << ")\n";
        }

    std::cout << "\nAll maximum_matching tests passed.\n";
}