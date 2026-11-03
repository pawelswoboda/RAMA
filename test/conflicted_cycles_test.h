#pragma once

#include "conflicted_cycles.h"
#include "test.h"
#include <vector>
#include <iostream>

template<template<typename> class VectorType>
void test_mixed_graph()
{
    // -1                                  -3
    //        +-----------------------+           +-------------------------+
    //        |                       v           |                         v
    //      +---+  +2   +---+  +3   +---+  +4   +---+  +1.5   +---+  +2   +---+  -1.5   +---+
    //   +- | 0 | ----> | 1 | ----> | 2 | ----> | 3 | ------> |   | ----> | 5 | ------> | 6 |
    //   |  +---+       +---+       +---+       +---+         |   |       +---+         +---+
    //   |    |                       |   +5      ^           |   |        +0.5           ^
    //   |    +-----------------------+-----------+           | 4 | ----------------------+
    //   |                            |                       |   |
    //   |                            |               -2      |   |
    //   +----------------------------+---------------------> |   |
    //                                |                       +---+
    //                                |   +2                    ^
    //                                +-------------------------+
    const std::vector<int> i =     {0, 1, 0, 2, 3, 0, 2, 0, 3, 4, 5, 4};
    const std::vector<int> j =     {1, 2, 2, 3, 4, 3, 4, 4, 5, 5, 6, 6};
    const std::vector<float> costs = {2., 3., -1., 4., 1.5, 5., 2., -2., -3., 2., -1.5, 0.5};

    Graph<VectorType> G(i.begin(), i.end(), j.begin(), j.end(),
                        costs.begin(), costs.end());

    // 3-cycles: 5 triangles (0,1,2), (0,2,3), (0,3,4), (3,4,5), (4,5,6)
    // 4-cycles: 7 triangles, 2 overlap with 3-cycles → 10 unique combined
    // 5-cycles: 3 triangles, all already present → still 10 unique
    {
        auto [v1, v2, v3, eff] = conflicted_cycles<VectorType>(G, 3);
        test(v1.size() == 5, "mixed graph max_cycle_length=3 should give 5 triangles");
    }
    {
        auto [v1, v2, v3, eff] = conflicted_cycles<VectorType>(G, 4);
        test(v1.size() == 10, "mixed graph max_cycle_length=4 should give 10 triangles");
    }
    {
        auto [v1, v2, v3, eff] = conflicted_cycles<VectorType>(G, 5);
        test(v1.size() == 10, "mixed graph max_cycle_length=5 should give 10 triangles");
    }
}

template<template<typename> class VectorType>
void test_3_cycle()
{
    //       +1
    //   +-----------------------+
    //   v                       |
    // +---+  -1   +---+  +2   +---+
    // | 0 | ----> | 1 | ----> | 2 |
    // +---+       +---+       +---+
    const std::vector<int> i =     {0, 1, 2};
    const std::vector<int> j =     {1, 2, 0};
    const std::vector<float> costs = {-1, 2, 1};

    Graph<VectorType> G(i.begin(), i.end(), j.begin(), j.end(),
                        costs.begin(), costs.end());

    {
        auto [v1, v2, v3, eff] = conflicted_cycles<VectorType>(G, 3);
        test(v1.size() == 1, "3-cycle with max_cycle_length=3 should give 1 triangle");
    }
    {
        auto [v1, v2, v3, eff] = conflicted_cycles<VectorType>(G, 4);
        test(v1.size() == 1, "3-cycle with max_cycle_length=4 should give 1 triangle");
    }
    {
        auto [v1, v2, v3, eff] = conflicted_cycles<VectorType>(G, 5);
        test(v1.size() == 1, "3-cycle with max_cycle_length=5 should give 1 triangle");
    }
}

template<template<typename> class VectorType>
void test_4_cycle()
{
    //       +3
    //   +-----------------------------------+
    //   v                                   |
    // +---+  -1   +---+  +2   +---+  +1   +---+
    // | 0 | ----> | 1 | ----> | 2 | ----> | 3 |
    // +---+       +---+       +---+       +---+
    const std::vector<int> i =     {0, 1, 2, 3};
    const std::vector<int> j =     {1, 2, 3, 0};
    const std::vector<float> costs = {-1, 2, 1, 3};

    Graph<VectorType> G(i.begin(), i.end(), j.begin(), j.end(),
                        costs.begin(), costs.end());

    {
        auto [v1, v2, v3, eff] = conflicted_cycles<VectorType>(G, 3);
        test(v1.size() == 0, "4-cycle with max_cycle_length=3 should give 0 triangles");
    }
    {
        auto [v1, v2, v3, eff] = conflicted_cycles<VectorType>(G, 4);
        test(v1.size() == 2, "4-cycle with max_cycle_length=4 should give 2 triangles");
    }
    {
        auto [v1, v2, v3, eff] = conflicted_cycles<VectorType>(G, 5);
        test(v1.size() == 2, "4-cycle with max_cycle_length=5 should give 2 triangles");
    }
}

template<template<typename> class VectorType>
void test_4_cycle_non_conflicting()
{
    // Two repulsive edges, no positive path connects both endpoints of either.
    //       +3
    //   +-----------------------------------+
    //   v                                   |
    // +---+  -1   +---+  -2   +---+  +1   +---+
    // | 0 | ----> | 1 | ----> | 2 | ----> | 3 |
    // +---+       +---+       +---+       +---+
    const std::vector<int> i =     {0, 1, 2, 3};
    const std::vector<int> j =     {1, 2, 3, 0};
    const std::vector<float> costs = {-1, -2, 1, 3};

    Graph<VectorType> G(i.begin(), i.end(), j.begin(), j.end(),
                        costs.begin(), costs.end());

    {
        auto [v1, v2, v3, eff] = conflicted_cycles<VectorType>(G, 3);
        test(v1.size() == 0, "non-conflicting 4-cycle max_cycle_length=3 should give 0");
    }
    {
        auto [v1, v2, v3, eff] = conflicted_cycles<VectorType>(G, 4);
        test(v1.size() == 0, "non-conflicting 4-cycle max_cycle_length=4 should give 0");
    }
    {
        auto [v1, v2, v3, eff] = conflicted_cycles<VectorType>(G, 5);
        test(v1.size() == 0, "non-conflicting 4-cycle max_cycle_length=5 should give 0");
    }
}

template<template<typename> class VectorType>
void test_5_cycle()
{
    //       +2
    //   +-----------------------------------------------+
    //   v                                               |
    // +---+  -1   +---+  +2   +---+  +1   +---+  +3   +---+
    // | 0 | ----> | 1 | ----> | 2 | ----> | 3 | ----> | 4 |
    // +---+       +---+       +---+       +---+       +---+
    const std::vector<int> i =     {0, 1, 2, 3, 4};
    const std::vector<int> j =     {1, 2, 3, 4, 0};
    const std::vector<float> costs = {-1, 2, 1, 3, 2};

    Graph<VectorType> G(i.begin(), i.end(), j.begin(), j.end(),
                        costs.begin(), costs.end());

    {
        auto [v1, v2, v3, eff] = conflicted_cycles<VectorType>(G, 3);
        test(v1.size() == 0, "5-cycle with max_cycle_length=3 should give 0 triangles");
    }
    {
        auto [v1, v2, v3, eff] = conflicted_cycles<VectorType>(G, 4);
        test(v1.size() == 0, "5-cycle with max_cycle_length=4 should give 0 triangles");
    }
    {
        auto [v1, v2, v3, eff] = conflicted_cycles<VectorType>(G, 5);
        test(v1.size() == 3, "5-cycle with max_cycle_length=5 should give 3 triangles");
    }
}

// A dense graph makes pentagon enumeration produce O(deg^3) raw triangles per
// repulsive edge, which used to overflow the 32-bit write offsets and corrupt
// memory. Check that the search stays within its raw-triangle cap and that
// every triangle it returns is a real triangle of the graph.
template<template<typename> class VectorType>
void test_dense_graph_triangle_cap()
{
    // Complete graph on 40 nodes: mostly attractive, every 7th edge repulsive.
    const int n = 40;
    std::vector<int> i, j;
    std::vector<float> costs;
    int e = 0;
    for (int a = 0; a < n; ++a)
        for (int b = a + 1; b < n; ++b, ++e)
        {
            i.push_back(a);
            j.push_back(b);
            costs.push_back(e % 7 == 0 ? -1.0f : 1.0f);
        }

    Graph<VectorType> G(i.begin(), i.end(), j.begin(), j.end(),
                        costs.begin(), costs.end());

    size_t prev_size = 0;
    for (const long long cap : {1000LL, 100000LL, 0LL})
    {
        auto [v1, v2, v3, eff] =
            conflicted_cycles<VectorType>(G, 5, 1e-4, false, "bfs", 0, 0.5f, cap);

        // The cap bounds raw triangles per finder; dedup only shrinks the
        // result, and there are three finders (3-, 4- and 5-cycles).
        if (cap > 0)
            test((long long)v1.size() <= 3 * cap,
                 "dense graph: triangle count must respect the raw cap");

        test(v1.size() == v2.size() && v2.size() == v3.size(),
             "dense graph: triangle vectors must have equal length");

        // Vertices sorted and in range. Corrupt write offsets showed up here
        // as garbage node ids before the counting was widened to 64 bits.
        std::vector<int> h1(v1.begin(), v1.end());
        std::vector<int> h2(v2.begin(), v2.end());
        std::vector<int> h3(v3.begin(), v3.end());
        for (size_t t = 0; t < h1.size(); ++t)
            test(h1[t] >= 0 && h3[t] < n && h1[t] < h2[t] && h2[t] < h3[t],
                 "dense graph: triangle vertices must be sorted and in range");

        // A looser cap can only find more triangles.
        test(v1.size() >= prev_size,
             "dense graph: raising the cap must not lose triangles");
        prev_size = v1.size();
    }
}

template<template<typename> class VectorType>
void run_all_conflicted_cycles_tests()
{
    test_mixed_graph<VectorType>();
    test_3_cycle<VectorType>();
    test_4_cycle<VectorType>();
    test_4_cycle_non_conflicting<VectorType>();
    test_5_cycle<VectorType>();
    test_dense_graph_triangle_cap<VectorType>();
    std::cout << "All conflicted_cycles tests passed.\n";
}
