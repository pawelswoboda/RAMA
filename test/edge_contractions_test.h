#pragma once

#include "edge_contractions.h"
#include "test.h"
#include "random_graph.h"
#include <vector>
#include <set>
#include <map>
#include <numeric>
#include <algorithm>

template<template<typename> class VectorType>
void test_conflicted_triangle()
{
    //       +2
    //   +-----------------------+
    //   v                       |
    // +---+  +3   +---+  -1   +---+
    // | 0 | ----> | 1 | ----> | 2 |
    // +---+       +---+       +---+
    //
    // MST of positive = {(0,1,3), (0,2,2)}
    // Repulsive edge (1,2) creates a conflicted cycle
    // Expected: nodes 0,1 stay together; node 2 separates
    //
    // +-----+  -1   +---+
    // | 0,1 | ----> | 2 |
    // +-----+       +---+
    std::vector<int> tails = {0, 0, 1};
    std::vector<int> heads = {1, 2, 2};
    std::vector<float> costs = {3.0f, 2.0f, -1.0f};

    Graph<VectorType> G(tails.begin(), tails.end(),
                        heads.begin(), heads.end(),
                        costs.begin(), costs.end());

    VectorType<int> mapping;
    int mst_size;
    std::tie(mapping, mst_size) = find_contraction_mapping<VectorType>(G, false);

    test(mapping.size() == 3, "mapping size should be 3");
    // Nodes 0 and 1 should be in same component
    test(mapping[0] == mapping[1], "nodes 0 and 1 should be in same component");
    // Node 2 should be separate
    test(mapping[2] != mapping[0], "node 2 should be in different component");
}

template<template<typename> class VectorType>
void test_disconnected_positive_components()
{
    // +---+  +3   +---+  -1   +---+  +2   +---+
    // | 0 | ----> | 1 | ----> | 2 | ----> | 3 |
    // +---+       +---+       +---+       +---+
    //
    // Positive MST: {(0,1,3), (2,3,2)} -- two disconnected trees
    // Repulsive edge (1,2) has endpoints in different CCs, removed immediately
    //
    // +-----+     +-----+
    // | 0,1 |     | 2,3 |
    // +-----+     +-----+
    std::vector<int> tails = {0, 2, 1};
    std::vector<int> heads = {1, 3, 2};
    std::vector<float> costs = {3.0f, 2.0f, -1.0f};

    Graph<VectorType> G(tails.begin(), tails.end(),
                        heads.begin(), heads.end(),
                        costs.begin(), costs.end());

    VectorType<int> mapping;
    int mst_size;
    std::tie(mapping, mst_size) = find_contraction_mapping<VectorType>(G, false);

    test(mapping.size() == 4, "mapping size should be 4");
    test(mapping[0] == mapping[1], "nodes 0,1 in same component");
    test(mapping[2] == mapping[3], "nodes 2,3 in same component");
    test(mapping[0] != mapping[2], "components {0,1} and {2,3} should differ");
}

template<template<typename> class VectorType>
void test_all_positive()
{
    //       +3
    //   +-----------------------+
    //   v                       |
    // +---+  +1   +---+  +2   +---+
    // | 0 | ----> | 1 | ----> | 2 |
    // +---+       +---+       +---+
    //
    // No repulsive edges, everything contracts to 1 node
    //
    // +-------+
    // | 0,1,2 |
    // +-------+
    std::vector<int> tails = {0, 1, 0};
    std::vector<int> heads = {1, 2, 2};
    std::vector<float> costs = {1.0f, 2.0f, 3.0f};

    Graph<VectorType> G(tails.begin(), tails.end(),
                        heads.begin(), heads.end(),
                        costs.begin(), costs.end());

    VectorType<int> mapping;
    int mst_size;
    std::tie(mapping, mst_size) = find_contraction_mapping<VectorType>(G, false);

    test(mapping.size() == 3, "mapping size should be 3");
    test(mapping[0] == mapping[1] && mapping[1] == mapping[2],
         "all nodes should be in same component");
    test(mst_size > 0, "MST should have edges remaining");
}

template<template<typename> class VectorType>
void test_random_graph()
{
    RandomGraph rg = generate_random_graph(30, 0.3, 42);
    if (rg.tails.empty())
        return; // degenerate graph

    Graph<VectorType> G(rg.num_nodes,
                        rg.tails.begin(), rg.tails.end(),
                        rg.heads.begin(), rg.heads.end(),
                        rg.costs.begin(), rg.costs.end());

    VectorType<int> mapping;
    int mst_size;
    std::tie(mapping, mst_size) = find_contraction_mapping<VectorType>(G, false);

    if (mapping.size() == 0)
        return; // no positive edges

    // Copy mapping to host for inspection
    const int n = rg.num_nodes;
    std::vector<int> m(n);
    thrust::copy(mapping.begin(), mapping.end(), m.begin());

    test(m.size() == (size_t)n, "mapping size should equal num_nodes");

    // Labels should be contiguous [0, num_components)
    int max_label = *std::max_element(m.begin(), m.end());
    int min_label = *std::min_element(m.begin(), m.end());
    test(min_label == 0, "min label should be 0");

    std::set<int> unique_labels(m.begin(), m.end());
    int num_components = unique_labels.size();
    test(max_label == num_components - 1, "labels should be contiguous");

    // mst_size is the number of directed (symmetric) surviving MST edges,
    // so single-direction count = mst_size / 2.
    // A forest on n nodes with k edges has n - k components.
    test(mst_size % 2 == 0, "surviving MST edges should be symmetric");
    test(num_components == n - mst_size / 2,
         "num components should equal num_nodes - surviving MST edges");

    // Every repulsive edge must separate its endpoints:
    // the algorithm removes bottleneck MST edges for every conflicted cycle
    for (size_t e = 0; e < rg.tails.size(); e++)
    {
        if (rg.costs[e] < 0)
            test(m[rg.tails[e]] != m[rg.heads[e]],
                 "repulsive edge endpoints should be in different components");
    }

    // Every component must be connected through positive edges.
    // Union-find: only union nodes that share a positive edge AND the same label.
    std::vector<int> uf(n);
    std::iota(uf.begin(), uf.end(), 0);
    auto find = [&uf](int x) {
        while (uf[x] != x) { uf[x] = uf[uf[x]]; x = uf[x]; }
        return x;
    };
    for (size_t e = 0; e < rg.tails.size(); e++)
    {
        if (rg.costs[e] > 0 && m[rg.tails[e]] == m[rg.heads[e]])
        {
            int fu = find(rg.tails[e]), fv = find(rg.heads[e]);
            if (fu != fv) uf[fu] = fv;
        }
    }
    // Each mapping label must correspond to exactly one union-find root
    std::map<int, int> label_to_root;
    for (int i = 0; i < n; i++)
    {
        int root = find(i);
        if (label_to_root.count(m[i]) == 0)
            label_to_root[m[i]] = root;
        else
            test(label_to_root[m[i]] == root,
                 "nodes in same component should be connected by positive edges");
    }
}

template<template<typename> class VectorType>
void test_negative_lifted_edge_prevents_merge()
{
    // All-positive base triangle: nodes 0,1,2 would normally all merge.
    //
    //       +2
    //   +-----------------------+
    //   v                       |
    // +---+  +3   +---+  +1   +---+
    // | 0 | ----> | 1 | ----> | 2 |
    // +---+       +---+       +---+
    //
    // Add a negative lifted edge (0,2) with cost -5.
    // This should prevent nodes 0 and 2 from being merged.
    //
    //   0 ......................... 2    (lifted, -5)
    //
    // Expected: node 0 and 2 in different components.
    std::vector<int> base_tails = {0, 1, 0};
    std::vector<int> base_heads = {1, 2, 2};
    std::vector<float> base_costs = {3.0f, 1.0f, 2.0f};

    Graph<VectorType> base_G(base_tails.begin(), base_tails.end(),
                             base_heads.begin(), base_heads.end(),
                             base_costs.begin(), base_costs.end());

    std::vector<int> lifted_tails = {0};
    std::vector<int> lifted_heads = {2};
    std::vector<float> lifted_costs = {-5.0f};

    Graph<VectorType> lifted_G(3,
                               lifted_tails.begin(), lifted_tails.end(),
                               lifted_heads.begin(), lifted_heads.end(),
                               lifted_costs.begin(), lifted_costs.end());

    VectorType<int> mapping;
    int mst_size;
    std::tie(mapping, mst_size) = find_contraction_mapping<VectorType>(base_G, lifted_G, false);

    test(mapping.size() == 3, "mapping size should be 3");
    test(mapping[0] != mapping[2], "nodes 0 and 2 should be in different components (negative lifted edge)");
}

template<template<typename> class VectorType>
void test_positive_lifted_edge_ignored()
{
    // Conflicted base triangle: repulsive edge (1,2).
    //
    //       +2
    //   +-----------------------+
    //   v                       |
    // +---+  +3   +---+  -1   +---+
    // | 0 | ----> | 1 | ----> | 2 |
    // +---+       +---+       +---+
    //
    // Add a positive lifted edge (0,2) with cost +10.
    // Positive lifted edges should be ignored — same result as without lifted graph.
    //
    // Expected: nodes 0,1 in same component; node 2 separate.
    std::vector<int> base_tails = {0, 0, 1};
    std::vector<int> base_heads = {1, 2, 2};
    std::vector<float> base_costs = {3.0f, 2.0f, -1.0f};

    Graph<VectorType> base_G(base_tails.begin(), base_tails.end(),
                             base_heads.begin(), base_heads.end(),
                             base_costs.begin(), base_costs.end());

    std::vector<int> lifted_tails = {0};
    std::vector<int> lifted_heads = {2};
    std::vector<float> lifted_costs = {10.0f};

    Graph<VectorType> lifted_G(3,
                               lifted_tails.begin(), lifted_tails.end(),
                               lifted_heads.begin(), lifted_heads.end(),
                               lifted_costs.begin(), lifted_costs.end());

    VectorType<int> mapping;
    int mst_size;
    std::tie(mapping, mst_size) = find_contraction_mapping<VectorType>(base_G, lifted_G, false);

    test(mapping.size() == 3, "mapping size should be 3");
    test(mapping[0] == mapping[1], "nodes 0 and 1 should be in same component");
    test(mapping[2] != mapping[0], "node 2 should be in different component");
}

template<template<typename> class VectorType>
void run_all_edge_contractions_tests()
{
    test_conflicted_triangle<VectorType>();
    std::cout << "test_conflicted_triangle passed\n";

    test_disconnected_positive_components<VectorType>();
    std::cout << "test_disconnected_positive_components passed\n";

    test_all_positive<VectorType>();
    std::cout << "test_all_positive passed\n";

    test_random_graph<VectorType>();
    std::cout << "test_random_graph passed\n";

    test_negative_lifted_edge_prevents_merge<VectorType>();
    std::cout << "test_negative_lifted_edge_prevents_merge passed\n";

    test_positive_lifted_edge_ignored<VectorType>();
    std::cout << "test_positive_lifted_edge_ignored passed\n";

    std::cout << "All edge_contractions tests passed!\n";
}
