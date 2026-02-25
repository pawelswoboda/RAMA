#pragma once

#include "lifted_edge_contraction.h"
#include "test.h"
#include <vector>
#include <set>
#include <algorithm>

template<template<typename> class VectorType>
void test_lec_direct_path()
{
    // +---+  +3   +---+  +3   +---+
    // | 0 | ----> | 1 | ----> | 2 |
    // +---+       +---+       +---+
    //   :                       :
    //   :........ +10 (lifted)..:
    //
    // All base edges non-negative. Lifted (0,2) positive.
    // BFS finds path 0 -> 1 -> 2, all three nodes merge.
    std::vector<int> base_tails = {0, 1};
    std::vector<int> base_heads = {1, 2};
    std::vector<float> base_costs = {3.0f, 3.0f};

    std::vector<int> lifted_tails = {0};
    std::vector<int> lifted_heads = {2};
    std::vector<float> lifted_costs = {10.0f};

    Graph<VectorType> base_G(base_tails.begin(), base_tails.end(),
                             base_heads.begin(), base_heads.end(),
                             base_costs.begin(), base_costs.end());
    Graph<VectorType> lifted_G(lifted_tails.begin(), lifted_tails.end(),
                               lifted_heads.begin(), lifted_heads.end(),
                               lifted_costs.begin(), lifted_costs.end());

    VectorType<int> mapping;
    int nr;
    std::tie(mapping, nr) = find_lifted_contraction_mapping<VectorType>(base_G, lifted_G, false);

    test(mapping.size() == 3, "lec_direct: mapping size should be 3");
    test(nr > 0, "lec_direct: should contract edges");

    std::vector<int> m(3);
    thrust::copy(mapping.begin(), mapping.end(), m.begin());

    // All three nodes should be in the same component
    test(m[0] == m[1] && m[1] == m[2],
         "lec_direct: all nodes on path should merge");
}

template<template<typename> class VectorType>
void test_lec_path_blocked_by_negative()
{
    // +---+  +3   +---+  -1   +---+
    // | 0 | ----> | 1 | ----> | 2 |
    // +---+       +---+       +---+
    //   :                       :
    //   :........ +10 (lifted)..:
    //
    // Edge (1,2) is negative, so BFS through non-negative edges
    // cannot reach node 2. No contraction.
    std::vector<int> base_tails = {0, 1};
    std::vector<int> base_heads = {1, 2};
    std::vector<float> base_costs = {3.0f, -1.0f};

    std::vector<int> lifted_tails = {0};
    std::vector<int> lifted_heads = {2};
    std::vector<float> lifted_costs = {10.0f};

    Graph<VectorType> base_G(base_tails.begin(), base_tails.end(),
                             base_heads.begin(), base_heads.end(),
                             base_costs.begin(), base_costs.end());
    Graph<VectorType> lifted_G(lifted_tails.begin(), lifted_tails.end(),
                               lifted_heads.begin(), lifted_heads.end(),
                               lifted_costs.begin(), lifted_costs.end());

    VectorType<int> mapping;
    int nr;
    std::tie(mapping, nr) = find_lifted_contraction_mapping<VectorType>(base_G, lifted_G, false);

    test(mapping.size() == 3, "lec_blocked: mapping size should be 3");
    test(nr == 0, "lec_blocked: should contract nothing");

    std::vector<int> m(3);
    thrust::copy(mapping.begin(), mapping.end(), m.begin());

    // Identity mapping
    test(m[0] == 0 && m[1] == 1 && m[2] == 2,
         "lec_blocked: mapping should be identity");
}

template<template<typename> class VectorType>
void test_lec_repulsive_lifted_ignored()
{
    // +---+  +3   +---+  +3   +---+
    // | 0 | ----> | 1 | ----> | 2 |
    // +---+       +---+       +---+
    //   :                       :
    //   :........ -5 (lifted)...:
    //
    // Lifted edge has negative cost — should be skipped entirely.
    std::vector<int> base_tails = {0, 1};
    std::vector<int> base_heads = {1, 2};
    std::vector<float> base_costs = {3.0f, 3.0f};

    std::vector<int> lifted_tails = {0};
    std::vector<int> lifted_heads = {2};
    std::vector<float> lifted_costs = {-5.0f};

    Graph<VectorType> base_G(base_tails.begin(), base_tails.end(),
                             base_heads.begin(), base_heads.end(),
                             base_costs.begin(), base_costs.end());
    Graph<VectorType> lifted_G(lifted_tails.begin(), lifted_tails.end(),
                               lifted_heads.begin(), lifted_heads.end(),
                               lifted_costs.begin(), lifted_costs.end());

    VectorType<int> mapping;
    int nr;
    std::tie(mapping, nr) = find_lifted_contraction_mapping<VectorType>(base_G, lifted_G, false);

    test(nr == 0, "lec_repulsive: should contract nothing");

    std::vector<int> m(3);
    thrust::copy(mapping.begin(), mapping.end(), m.begin());

    test(m[0] == 0 && m[1] == 1 && m[2] == 2,
         "lec_repulsive: mapping should be identity");
}

template<template<typename> class VectorType>
void test_lec_empty_lifted()
{
    std::vector<int> base_tails = {0, 1};
    std::vector<int> base_heads = {1, 2};
    std::vector<float> base_costs = {3.0f, 3.0f};

    Graph<VectorType> base_G(base_tails.begin(), base_tails.end(),
                             base_heads.begin(), base_heads.end(),
                             base_costs.begin(), base_costs.end());
    Graph<VectorType> lifted_G;

    VectorType<int> mapping;
    int nr;
    std::tie(mapping, nr) = find_lifted_contraction_mapping<VectorType>(base_G, lifted_G, false);

    test(mapping.size() == 3, "lec_empty: mapping size should be 3");
    test(nr == 0, "lec_empty: should contract nothing");
}

template<template<typename> class VectorType>
void test_lec_4cycle_alternate_path()
{
    //       +10 (lifted)
    //   .........................
    //   :                       v
    // +---+  +3   +---+  -1   +---+  +3   +---+
    // | 0 | ----> | 1 | ----> | 2 | ----> | 3 |
    // +---+       +---+       +---+       +---+
    //   ^   +3                              |
    //   +-----------------------------------+
    //
    // Direct path 0→1→2 blocked by negative edge (1,2).
    // Alternate path 0→3→2: (0,3)=+3 and (3,2)=+3, both non-negative.
    // BFS should find path through node 3, merging {0,3,2}.
    // Node 1 stays separate.
    std::vector<int> base_tails = {0, 1, 2, 0};
    std::vector<int> base_heads = {1, 2, 3, 3};
    std::vector<float> base_costs = {3.0f, -1.0f, 3.0f, 3.0f};

    std::vector<int> lifted_tails = {0};
    std::vector<int> lifted_heads = {2};
    std::vector<float> lifted_costs = {10.0f};

    Graph<VectorType> base_G(base_tails.begin(), base_tails.end(),
                             base_heads.begin(), base_heads.end(),
                             base_costs.begin(), base_costs.end());
    Graph<VectorType> lifted_G(lifted_tails.begin(), lifted_tails.end(),
                               lifted_heads.begin(), lifted_heads.end(),
                               lifted_costs.begin(), lifted_costs.end());

    VectorType<int> mapping;
    int nr;
    std::tie(mapping, nr) = find_lifted_contraction_mapping<VectorType>(base_G, lifted_G, false);

    test(mapping.size() == 4, "lec_4cycle: mapping size should be 4");
    test(nr > 0, "lec_4cycle: should contract edges");

    std::vector<int> m(4);
    thrust::copy(mapping.begin(), mapping.end(), m.begin());

    // Nodes 0 and 2 (lifted endpoints) should be merged
    test(m[0] == m[2], "lec_4cycle: lifted endpoints 0 and 2 should merge");
    // Node 3 is on the path, should also merge
    test(m[0] == m[3], "lec_4cycle: node 3 on path should merge with 0 and 2");
    // Node 1 is NOT on the path (blocked by negative edge)
    test(m[1] != m[0], "lec_4cycle: node 1 should stay separate");
}

template<template<typename> class VectorType>
void test_lec_two_lifted_edges()
{
    // +---+  +1   +---+  +1   +---+  +1   +---+  +1   +---+
    // | 0 | ----> | 1 | ----> | 2 | ----> | 3 | ----> | 4 |
    // +---+       +---+       +---+       +---+       +---+
    //   :           :                       :           :
    //   :.. +10 ....:                       :... +10 ...:
    //      (lifted)                            (lifted)
    //
    // Lifted (0,1) merges {0,1}.  Lifted (3,4) merges {3,4}.
    // Node 2 is not on any lifted path.
    std::vector<int> base_tails = {0, 1, 2, 3};
    std::vector<int> base_heads = {1, 2, 3, 4};
    std::vector<float> base_costs = {1.0f, 1.0f, 1.0f, 1.0f};

    std::vector<int> lifted_tails = {0, 3};
    std::vector<int> lifted_heads = {1, 4};
    std::vector<float> lifted_costs = {10.0f, 10.0f};

    Graph<VectorType> base_G(base_tails.begin(), base_tails.end(),
                             base_heads.begin(), base_heads.end(),
                             base_costs.begin(), base_costs.end());
    Graph<VectorType> lifted_G(lifted_tails.begin(), lifted_tails.end(),
                               lifted_heads.begin(), lifted_heads.end(),
                               lifted_costs.begin(), lifted_costs.end());

    VectorType<int> mapping;
    int nr;
    std::tie(mapping, nr) = find_lifted_contraction_mapping<VectorType>(base_G, lifted_G, false);

    test(mapping.size() == 5, "lec_two: mapping size should be 5");
    test(nr > 0, "lec_two: should contract edges");

    std::vector<int> m(5);
    thrust::copy(mapping.begin(), mapping.end(), m.begin());

    test(m[0] == m[1], "lec_two: lifted (0,1) should merge");
    test(m[3] == m[4], "lec_two: lifted (3,4) should merge");
    test(m[2] != m[0], "lec_two: node 2 should be separate from {0,1}");
    test(m[2] != m[3], "lec_two: node 2 should be separate from {3,4}");
}

template<template<typename> class VectorType>
void test_lec_labels_compressed()
{
    // Same as test_lec_direct_path but verify labels are contiguous.
    std::vector<int> base_tails = {0, 1};
    std::vector<int> base_heads = {1, 2};
    std::vector<float> base_costs = {3.0f, 3.0f};

    std::vector<int> lifted_tails = {0};
    std::vector<int> lifted_heads = {2};
    std::vector<float> lifted_costs = {10.0f};

    Graph<VectorType> base_G(base_tails.begin(), base_tails.end(),
                             base_heads.begin(), base_heads.end(),
                             base_costs.begin(), base_costs.end());
    Graph<VectorType> lifted_G(lifted_tails.begin(), lifted_tails.end(),
                               lifted_heads.begin(), lifted_heads.end(),
                               lifted_costs.begin(), lifted_costs.end());

    VectorType<int> mapping;
    int nr;
    std::tie(mapping, nr) = find_lifted_contraction_mapping<VectorType>(base_G, lifted_G, false);

    std::vector<int> m(3);
    thrust::copy(mapping.begin(), mapping.end(), m.begin());

    // All three merged → single label 0
    std::set<int> labels(m.begin(), m.end());
    test(labels.size() == 1, "lec_compressed: all nodes should have same label");
    test(*labels.begin() == 0, "lec_compressed: single label should be 0");
}

template<template<typename> class VectorType>
void test_lec_longer_path()
{
    // +---+  +1   +---+  +1   +---+  +1   +---+  +1   +---+
    // | 0 | ----> | 1 | ----> | 2 | ----> | 3 | ----> | 4 |
    // +---+       +---+       +---+       +---+       +---+
    //   :                                               :
    //   :.................. +10 (lifted) ................:
    //
    // BFS finds path 0→1→2→3→4, all nodes merge.
    std::vector<int> base_tails = {0, 1, 2, 3};
    std::vector<int> base_heads = {1, 2, 3, 4};
    std::vector<float> base_costs = {1.0f, 1.0f, 1.0f, 1.0f};

    std::vector<int> lifted_tails = {0};
    std::vector<int> lifted_heads = {4};
    std::vector<float> lifted_costs = {10.0f};

    Graph<VectorType> base_G(base_tails.begin(), base_tails.end(),
                             base_heads.begin(), base_heads.end(),
                             base_costs.begin(), base_costs.end());
    Graph<VectorType> lifted_G(lifted_tails.begin(), lifted_tails.end(),
                               lifted_heads.begin(), lifted_heads.end(),
                               lifted_costs.begin(), lifted_costs.end());

    VectorType<int> mapping;
    int nr;
    std::tie(mapping, nr) = find_lifted_contraction_mapping<VectorType>(base_G, lifted_G, false);

    test(mapping.size() == 5, "lec_longer: mapping size should be 5");
    test(nr == 4, "lec_longer: should contract 4 path edges");

    std::vector<int> m(5);
    thrust::copy(mapping.begin(), mapping.end(), m.begin());

    test(m[0] == m[1] && m[1] == m[2] && m[2] == m[3] && m[3] == m[4],
         "lec_longer: all 5 nodes on path should merge");
}

template<template<typename> class VectorType>
void test_lec_diamond_one_path_chosen()
{
    //               +---+
    //         +3  / | 1 | \  +3
    //           /   +---+   \
    // +---+                     +---+
    // | 0 |                     | 3 |
    // +---+                     +---+
    //           \   +---+   /
    //         +3  \ | 2 | /  +3
    //               +---+
    //   :                         :
    //   :........ +10 (lifted) ...:
    //
    // Two base paths from 0 to 3, both through non-negative edges:
    //   0 -> 1 -> 3: edges (0,1)=+3, (1,3)=+3
    //   0 -> 2 -> 3: edges (0,2)=+3, (2,3)=+3
    //
    // BFS should pick exactly one path. The other intermediate node
    // should NOT be merged (it's not on the chosen path).
    std::vector<int> base_tails = {0, 0, 1, 2};
    std::vector<int> base_heads = {1, 2, 3, 3};
    std::vector<float> base_costs = {3.0f, 3.0f, 3.0f, 3.0f};

    std::vector<int> lifted_tails = {0};
    std::vector<int> lifted_heads = {3};
    std::vector<float> lifted_costs = {10.0f};

    Graph<VectorType> base_G(base_tails.begin(), base_tails.end(),
                             base_heads.begin(), base_heads.end(),
                             base_costs.begin(), base_costs.end());
    Graph<VectorType> lifted_G(lifted_tails.begin(), lifted_tails.end(),
                               lifted_heads.begin(), lifted_heads.end(),
                               lifted_costs.begin(), lifted_costs.end());

    VectorType<int> mapping;
    int nr;
    std::tie(mapping, nr) = find_lifted_contraction_mapping<VectorType>(base_G, lifted_G, false);

    test(mapping.size() == 4, "lec_diamond: mapping size should be 4");
    test(nr == 2, "lec_diamond: should contract exactly 2 path edges");

    std::vector<int> m(4);
    thrust::copy(mapping.begin(), mapping.end(), m.begin());

    // Lifted endpoints 0 and 3 must merge
    test(m[0] == m[3], "lec_diamond: lifted endpoints 0 and 3 should merge");

    // Exactly one of {1, 2} is on the chosen path
    bool one_merged = (m[1] == m[0]) != (m[2] == m[0]);
    test(one_merged, "lec_diamond: exactly one intermediate node should merge");
}

template<template<typename> class VectorType>
void run_all_lifted_edge_contraction_tests()
{
    test_lec_direct_path<VectorType>();
    std::cout << "test_lec_direct_path passed\n";

    test_lec_path_blocked_by_negative<VectorType>();
    std::cout << "test_lec_path_blocked_by_negative passed\n";

    test_lec_repulsive_lifted_ignored<VectorType>();
    std::cout << "test_lec_repulsive_lifted_ignored passed\n";

    test_lec_empty_lifted<VectorType>();
    std::cout << "test_lec_empty_lifted passed\n";

    test_lec_4cycle_alternate_path<VectorType>();
    std::cout << "test_lec_4cycle_alternate_path passed\n";

    test_lec_two_lifted_edges<VectorType>();
    std::cout << "test_lec_two_lifted_edges passed\n";

    test_lec_labels_compressed<VectorType>();
    std::cout << "test_lec_labels_compressed passed\n";

    test_lec_longer_path<VectorType>();
    std::cout << "test_lec_longer_path passed\n";

    test_lec_diamond_one_path_chosen<VectorType>();
    std::cout << "test_lec_diamond_one_path_chosen passed\n";

    std::cout << "All lifted_edge_contraction tests passed!\n";
}
