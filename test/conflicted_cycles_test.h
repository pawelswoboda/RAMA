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
        auto [v1, v2, v3] = conflicted_cycles<VectorType>(G, 3);
        test(v1.size() == 5, "mixed graph max_cycle_length=3 should give 5 triangles");
    }
    {
        auto [v1, v2, v3] = conflicted_cycles<VectorType>(G, 4);
        test(v1.size() == 10, "mixed graph max_cycle_length=4 should give 10 triangles");
    }
    {
        auto [v1, v2, v3] = conflicted_cycles<VectorType>(G, 5);
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
        auto [v1, v2, v3] = conflicted_cycles<VectorType>(G, 3);
        test(v1.size() == 1, "3-cycle with max_cycle_length=3 should give 1 triangle");
    }
    {
        auto [v1, v2, v3] = conflicted_cycles<VectorType>(G, 4);
        test(v1.size() == 1, "3-cycle with max_cycle_length=4 should give 1 triangle");
    }
    {
        auto [v1, v2, v3] = conflicted_cycles<VectorType>(G, 5);
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
        auto [v1, v2, v3] = conflicted_cycles<VectorType>(G, 3);
        test(v1.size() == 0, "4-cycle with max_cycle_length=3 should give 0 triangles");
    }
    {
        auto [v1, v2, v3] = conflicted_cycles<VectorType>(G, 4);
        test(v1.size() == 2, "4-cycle with max_cycle_length=4 should give 2 triangles");
    }
    {
        auto [v1, v2, v3] = conflicted_cycles<VectorType>(G, 5);
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
        auto [v1, v2, v3] = conflicted_cycles<VectorType>(G, 3);
        test(v1.size() == 0, "non-conflicting 4-cycle max_cycle_length=3 should give 0");
    }
    {
        auto [v1, v2, v3] = conflicted_cycles<VectorType>(G, 4);
        test(v1.size() == 0, "non-conflicting 4-cycle max_cycle_length=4 should give 0");
    }
    {
        auto [v1, v2, v3] = conflicted_cycles<VectorType>(G, 5);
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
        auto [v1, v2, v3] = conflicted_cycles<VectorType>(G, 3);
        test(v1.size() == 0, "5-cycle with max_cycle_length=3 should give 0 triangles");
    }
    {
        auto [v1, v2, v3] = conflicted_cycles<VectorType>(G, 4);
        test(v1.size() == 0, "5-cycle with max_cycle_length=4 should give 0 triangles");
    }
    {
        auto [v1, v2, v3] = conflicted_cycles<VectorType>(G, 5);
        test(v1.size() == 3, "5-cycle with max_cycle_length=5 should give 3 triangles");
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
    std::cout << "All conflicted_cycles tests passed.\n";
}
