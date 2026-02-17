#pragma once

#include "graph.h"
#include "test.h"
#include <vector>
#include <cmath>

template<template<typename> class VectorType>
void test_basic_construction()
{
    // Triangle: 0-1 (cost 1.0), 1-2 (cost -0.5), 0-2 (cost 2.0)
    std::vector<int> tails = {0, 1, 0};
    std::vector<int> heads = {1, 2, 2};
    std::vector<float> costs = {1.0f, -0.5f, 2.0f};

    Graph<VectorType> g(tails.begin(), tails.end(),
                        heads.begin(), heads.end(),
                        costs.begin(), costs.end());

    test(g.num_nodes() == 3, "num_nodes should be 3");
    test(g.num_edges() == 3, "num_edges should be 3 (undirected)");
    test(g.num_directed_edges() == 6, "num_directed_edges should be 6 (both directions)");
}

template<template<typename> class VectorType>
void test_construction_with_num_nodes()
{
    std::vector<int> tails = {0, 1};
    std::vector<int> heads = {1, 2};
    std::vector<float> costs = {1.0f, 2.0f};

    // Explicitly provide num_nodes larger than inferred
    Graph<VectorType> g(5, tails.begin(), tails.end(),
                        heads.begin(), heads.end(),
                        costs.begin(), costs.end());

    test(g.num_nodes() == 5, "num_nodes should be 5 (explicitly set)");
    test(g.num_edges() == 2, "num_edges should be 2");
}

template<template<typename> class VectorType>
void test_move_construction()
{
    VectorType<int> tails(2);
    VectorType<int> heads(2);
    VectorType<float> costs(2);
    // Single orientation only; ensure_symmetric adds the reverse
    tails[0] = 0; heads[0] = 1; costs[0] = 1.5f;
    tails[1] = 0; heads[1] = 2; costs[1] = -1.0f;

    Graph<VectorType> g(std::move(tails), std::move(heads), std::move(costs));

    test(g.num_nodes() == 3, "move ctor: num_nodes should be 3");
    test(g.num_edges() == 2, "move ctor: num_edges should be 2");
    test(g.num_directed_edges() == 4, "move ctor: num_directed_edges should be 4");
}

template<template<typename> class VectorType>
void test_symmetry()
{
    // Provide only one direction; ensure_symmetric should add the reverse
    std::vector<int> tails = {0};
    std::vector<int> heads = {1};
    std::vector<float> costs = {3.0f};

    Graph<VectorType> g(tails.begin(), tails.end(),
                        heads.begin(), heads.end(),
                        costs.begin(), costs.end());

    test(g.num_directed_edges() == 2, "single edge should become 2 directed edges");
    test(g.num_edges() == 1, "single edge should be 1 undirected edge");

    // Check that both directions are present
    const VectorType<int>& t = g.get_tails();
    const VectorType<int>& h = g.get_heads();
    const VectorType<float>& c = g.get_costs();

    // Sorted order: (0,1) then (1,0)
    test(t[0] == 0 && h[0] == 1, "first directed edge should be 0->1");
    test(t[1] == 1 && h[1] == 0, "second directed edge should be 1->0");
    test(c[0] == 3.0f && c[1] == 3.0f, "both directions should have same cost");
}

template<template<typename> class VectorType>
void test_statistics()
{
    std::vector<int> tails = {0, 1};
    std::vector<int> heads = {1, 2};
    std::vector<float> costs = {1.0f, -2.0f};

    Graph<VectorType> g(tails.begin(), tails.end(),
                        heads.begin(), heads.end(),
                        costs.begin(), costs.end());

    // After symmetry: costs are {-2, 1, -2, 1} (sorted by (tail,head))
    test(g.min() == -2.0f, "min should be -2.0");
    test(g.max() == 1.0f, "max should be 1.0");
    // Sum of all directed edges: 1 + (-2) + 1 + (-2) = -2
    test(std::fabs(g.sum() - (-2.0f)) < 1e-5f, "sum should be -2.0");
}

template<template<typename> class VectorType>
void test_self_loops()
{
    // Edges: 0->0 (self-loop), 0->1, 1->1 (self-loop)
    std::vector<int> tails = {0, 0, 1};
    std::vector<int> heads = {0, 1, 1};
    std::vector<float> costs = {5.0f, 1.0f, 3.0f};

    Graph<VectorType> g(tails.begin(), tails.end(),
                        heads.begin(), heads.end(),
                        costs.begin(), costs.end());

    // self_loop_costs
    VectorType<float> slc = g.self_loop_costs();
    test(slc.size() == 2, "self_loop_costs size should be num_nodes");
    test(slc[0] == 5.0f, "node 0 self-loop cost should be 5.0");
    test(slc[1] == 3.0f, "node 1 self-loop cost should be 3.0");

    // remove_self_loops
    g.remove_self_loops();
    // Only the 0->1 / 1->0 edge pair should remain
    test(g.num_directed_edges() == 2, "after removing self-loops, 2 directed edges remain");
}

template<template<typename> class VectorType>
void test_node_offsets()
{
    // Star graph: 0->1, 0->2, 0->3
    std::vector<int> tails = {0, 0, 0};
    std::vector<int> heads = {1, 2, 3};
    std::vector<float> costs = {1.0f, 2.0f, 3.0f};

    Graph<VectorType> g(tails.begin(), tails.end(),
                        heads.begin(), heads.end(),
                        costs.begin(), costs.end());

    // After symmetry: 6 directed edges
    // Node 0 has edges to 1,2,3 (3 edges)
    // Nodes 1,2,3 each have 1 edge (to 0)
    VectorType<int> offsets = g.compute_node_offsets();
    test(offsets.size() == 5, "offsets size should be num_nodes + 1");
    test(offsets[0] == 0, "offset[0] = 0");
    test(offsets[1] == 3, "offset[1] = 3 (node 0 has 3 neighbors)");
    test(offsets[2] == 4, "offset[2] = 4 (node 1 has 1 neighbor)");
    test(offsets[3] == 5, "offset[3] = 5 (node 2 has 1 neighbor)");
    test(offsets[4] == 6, "offset[4] = 6 (node 3 has 1 neighbor)");
}

template<template<typename> class VectorType>
void test_filter()
{
    // Edges with various costs
    std::vector<int> tails = {0, 0, 1};
    std::vector<int> heads = {1, 2, 2};
    std::vector<float> costs = {1.0f, -3.0f, 5.0f};

    Graph<VectorType> g(tails.begin(), tails.end(),
                        heads.begin(), heads.end(),
                        costs.begin(), costs.end());

    // Filter to keep only edges with cost in [0, 10]
    Graph<VectorType> filtered = g.filter(0.0f, 10.0f);

    // Should keep edges with cost 1.0 and 5.0 (and their reverses)
    test(filtered.num_directed_edges() == 4, "filtered should have 4 directed edges");
    test(filtered.num_edges() == 2, "filtered should have 2 undirected edges");
    test(filtered.num_nodes() == 3, "filtered should preserve num_nodes");
    test(filtered.min() == 1.0f, "filtered min should be 1.0");
    test(filtered.max() == 5.0f, "filtered max should be 5.0");
}

template<template<typename> class VectorType>
void test_is_single_orientation()
{
    // Single directional: each edge given once
    {
        std::vector<int> tails = {0, 1, 0};
        std::vector<int> heads = {1, 2, 2};
        test(Graph<VectorType>::is_single_orientation(
                 tails.begin(), tails.end(), heads.begin(), heads.end()),
             "single-directional edges should return true");
    }

    // Symmetric: both (0,1) and (1,0) present
    {
        std::vector<int> tails = {0, 1};
        std::vector<int> heads = {1, 0};
        test(!Graph<VectorType>::is_single_orientation(
                  tails.begin(), tails.end(), heads.begin(), heads.end()),
             "symmetric edges should return false");
    }

    // Symmetric with multiple edges: (0,1),(1,0),(1,2)
    {
        std::vector<int> tails = {0, 1, 1};
        std::vector<int> heads = {1, 0, 2};
        test(!Graph<VectorType>::is_single_orientation(
                  tails.begin(), tails.end(), heads.begin(), heads.end()),
             "partially symmetric edges should return false");
    }

    // Single edge: trivially single orientation
    {
        std::vector<int> tails = {0};
        std::vector<int> heads = {1};
        test(Graph<VectorType>::is_single_orientation(
                 tails.begin(), tails.end(), heads.begin(), heads.end()),
             "single edge should return true");
    }

    // Empty: trivially single orientation
    {
        std::vector<int> tails = {};
        std::vector<int> heads = {};
        test(Graph<VectorType>::is_single_orientation(
                 tails.begin(), tails.end(), heads.begin(), heads.end()),
             "empty edges should return true");
    }

    // Self-loops are fine (single orientation)
    {
        std::vector<int> tails = {0, 0, 1};
        std::vector<int> heads = {0, 1, 1};
        test(Graph<VectorType>::is_single_orientation(
                 tails.begin(), tails.end(), heads.begin(), heads.end()),
             "self-loops with single-directional edges should return true");
    }
}

template<template<typename> class VectorType>
void test_construct_symmetric()
{
    // Construct with edges already in both directions, is_symmetric=true
    {
        std::vector<int> tails = {0, 1, 1, 2};
        std::vector<int> heads = {1, 0, 2, 1};
        std::vector<float> costs = {1.0f, 1.0f, -0.5f, -0.5f};

        Graph<VectorType> g(tails.begin(), tails.end(),
                            heads.begin(), heads.end(),
                            costs.begin(), costs.end(),
                            false, true);

        test(g.num_nodes() == 3, "symmetric ctor: num_nodes should be 3");
        test(g.num_edges() == 2, "symmetric ctor: num_edges should be 2");
        test(g.num_directed_edges() == 4, "symmetric ctor: num_directed_edges should be 4");
    }

    // Construct with num_nodes and is_symmetric=true
    {
        std::vector<int> tails = {0, 1};
        std::vector<int> heads = {1, 0};
        std::vector<float> costs = {3.0f, 3.0f};

        Graph<VectorType> g(5, tails.begin(), tails.end(),
                            heads.begin(), heads.end(),
                            costs.begin(), costs.end(),
                            false, true);

        test(g.num_nodes() == 5, "symmetric ctor with num_nodes: should be 5");
        test(g.num_edges() == 1, "symmetric ctor with num_nodes: 1 undirected edge");
        test(g.num_directed_edges() == 2, "symmetric ctor with num_nodes: 2 directed edges");
    }

    // Move constructor with is_symmetric=true
    {
        VectorType<int> tails(4), heads(4);
        VectorType<float> costs(4);
        tails[0] = 0; heads[0] = 1; costs[0] = 2.0f;
        tails[1] = 0; heads[1] = 2; costs[1] = -1.0f;
        tails[2] = 1; heads[2] = 0; costs[2] = 2.0f;
        tails[3] = 2; heads[3] = 0; costs[3] = -1.0f;

        Graph<VectorType> g(std::move(tails), std::move(heads), std::move(costs),
                            false, true);

        test(g.num_nodes() == 3, "symmetric move ctor: num_nodes should be 3");
        test(g.num_edges() == 2, "symmetric move ctor: num_edges should be 2");
        test(g.num_directed_edges() == 4, "symmetric move ctor: num_directed_edges should be 4");
        test(g.min() == -1.0f, "symmetric move ctor: min should be -1.0");
        test(g.max() == 2.0f, "symmetric move ctor: max should be 2.0");
    }
}

template<template<typename> class VectorType>
void test_has_duplicate_edges()
{
    // No duplicates
    {
        std::vector<int> tails = {0, 1, 0};
        std::vector<int> heads = {1, 2, 2};
        test(!Graph<VectorType>::has_duplicate_edges(
                 tails.begin(), tails.end(), heads.begin(), heads.end()),
             "distinct edges should not have duplicates");
    }

    // Duplicate directed edge
    {
        std::vector<int> tails = {0, 0};
        std::vector<int> heads = {1, 1};
        test(Graph<VectorType>::has_duplicate_edges(
                 tails.begin(), tails.end(), heads.begin(), heads.end()),
             "identical directed edges should be duplicates");
    }

    // (i,j) and (j,i) are NOT duplicates (different directed edges)
    {
        std::vector<int> tails = {0, 1};
        std::vector<int> heads = {1, 0};
        test(!Graph<VectorType>::has_duplicate_edges(
                 tails.begin(), tails.end(), heads.begin(), heads.end()),
             "reverse edges are not duplicates");
    }

    // Empty and single edge
    {
        std::vector<int> empty_t = {}, empty_h = {};
        test(!Graph<VectorType>::has_duplicate_edges(
                 empty_t.begin(), empty_t.end(), empty_h.begin(), empty_h.end()),
             "empty edges should not have duplicates");

        std::vector<int> t1 = {0}, h1 = {1};
        test(!Graph<VectorType>::has_duplicate_edges(
                 t1.begin(), t1.end(), h1.begin(), h1.end()),
             "single edge should not have duplicates");
    }
}

template<template<typename> class VectorType>
void test_contract()
{
    //       +2
    //   +-----------------------+
    //   v                       |
    // +---+  +3   +---+  -1   +---+
    // | 0 | ----> | 1 | ----> | 2 |
    // +---+       +---+       +---+
    //
    // Mapping: {0->0, 1->0, 2->1} merges nodes 0,1
    //
    // +-----+  +1   +-----+
    // | 0,1 | ----> |  2  |
    // +-----+       +-----+
    //
    // Edges after mapping:
    //   (0,1): was (0,2,+2) -> maps to (0,1,+2)
    //   (0,1): was (1,2,-1) -> maps to (0,1,-1)
    //   Summed: (0,1, +1)
    //   Self-loop on 0: was (0,1,+3) -> maps to (0,0,+3)
    std::vector<int> tails = {0, 0, 1};
    std::vector<int> heads = {1, 2, 2};
    std::vector<float> costs = {3.0f, 2.0f, -1.0f};

    Graph<VectorType> G(tails.begin(), tails.end(),
                        heads.begin(), heads.end(),
                        costs.begin(), costs.end());

    VectorType<int> mapping(3);
    mapping[0] = 0; mapping[1] = 0; mapping[2] = 1;

    Graph<VectorType> contracted = G.contract(mapping);

    test(contracted.num_nodes() == 2, "contracted graph should have 2 nodes");

    // Should have self-loop (0,0,+3) and edge (0,1,+1) with reverse (1,0,+1)
    // directed edges: (0,0), (0,1), (1,0) = 3
    test(contracted.num_directed_edges() == 3, "contracted should have 3 directed edges");

    // Check self-loop cost
    VectorType<float> slc = contracted.self_loop_costs();
    test(slc[0] == 3.0f, "self-loop on node 0 should have cost 3.0 (from merged edge 0-1)");
    test(slc[1] == 0.0f, "node 1 should have no self-loop");

    // After removing self-loops, should have 1 undirected edge with cost +1
    contracted.remove_self_loops();
    test(contracted.num_edges() == 1, "after removing self-loops, 1 undirected edge");
    test(std::fabs(contracted.sum() - 2.0f) < 1e-5f,
         "sum of symmetric costs should be 2.0 (= 2 * +1.0)");
}

template<template<typename> class VectorType>
void test_contract_all_to_one()
{
    // All-positive triangle, contract everything to node 0
    std::vector<int> tails = {0, 1, 0};
    std::vector<int> heads = {1, 2, 2};
    std::vector<float> costs = {1.0f, 2.0f, 3.0f};

    Graph<VectorType> G(tails.begin(), tails.end(),
                        heads.begin(), heads.end(),
                        costs.begin(), costs.end());

    VectorType<int> mapping(3, 0);

    Graph<VectorType> contracted = G.contract(mapping);

    test(contracted.num_nodes() == 1, "all contracted to 1 node");
    // All 3 edges become self-loops on node 0, summing to 1+2+3=6
    VectorType<float> slc = contracted.self_loop_costs();
    test(std::fabs(slc[0] - 6.0f) < 1e-5f, "self-loop should sum all edge costs");

    contracted.remove_self_loops();
    test(contracted.num_directed_edges() == 0, "no edges after removing self-loops");
}

template<template<typename> class VectorType>
void run_all_graph_tests()
{
    test_is_single_orientation<VectorType>();
    test_has_duplicate_edges<VectorType>();
    test_construct_symmetric<VectorType>();
    test_basic_construction<VectorType>();
    test_construction_with_num_nodes<VectorType>();
    test_move_construction<VectorType>();
    test_symmetry<VectorType>();
    test_statistics<VectorType>();
    test_self_loops<VectorType>();
    test_node_offsets<VectorType>();
    test_filter<VectorType>();

    test_contract<VectorType>();
    std::cout << "test_contract passed\n";
    test_contract_all_to_one<VectorType>();
    std::cout << "test_contract_all_to_one passed\n";
}
