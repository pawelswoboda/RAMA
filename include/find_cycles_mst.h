#pragma once

#include <tuple>
#include <iostream>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include "mst_boruvka.h"
#include "graph.h"
#include "find_cycles_bfs.h"

// Find conflicted cycles via MST fundamental cycles.
// For each repulsive edge (u,v), computes the maximum spanning tree of the
// positive subgraph, then finds the unique tree path u->v via BFS on the
// (sparse) MST. Fan-triangulates paths of length >= 5 (cycle length >= 6).
//
// pos_edge_tails, pos_edge_heads, pos_edge_costs: symmetric COO of positive subgraph.
// rep_edge_costs: costs of repulsive edges (parallel to rep_edge_tails/heads).
// Other parameters as in find_conflicted_cycles_bfs.
template<template<typename> class VectorType>
std::tuple<VectorType<int>, VectorType<int>, VectorType<int>, VectorType<float>>
find_conflicted_cycles_mst(
    const VectorType<int>& rep_edge_tails,
    const VectorType<int>& rep_edge_heads,
    const VectorType<float>& rep_edge_costs,
    const VectorType<int>& pos_edge_tails,
    const VectorType<int>& pos_edge_heads,
    const VectorType<float>& pos_edge_costs,
    const int num_nodes,
    const int max_cycle_length,
    const bool verbose = true)
{
    if (pos_edge_tails.size() == 0 || rep_edge_tails.size() == 0)
        return {VectorType<int>(), VectorType<int>(), VectorType<int>(), VectorType<float>()};

    // 1. Compute maximum spanning tree of positive subgraph.
    auto [mst_tails, mst_heads, mst_costs] =
        MST_boruvka::maximum_spanning_tree<VectorType>(
            pos_edge_tails, pos_edge_heads, pos_edge_costs);

    if (mst_tails.size() == 0)
        return {VectorType<int>(), VectorType<int>(), VectorType<int>(), VectorType<float>()};

    if (verbose)
        std::cout << "MST cycles: " << mst_tails.size()
                  << " MST edges from " << pos_edge_tails.size() / 2
                  << " positive edges\n";

    // 2. Build Graph from single-direction MST edges (constructor symmetrizes).
    Graph<VectorType> mst_graph(num_nodes,
        std::move(mst_tails), std::move(mst_heads), std::move(mst_costs));

    // 3. Extract CSR representation.
    VectorType<int> mst_offsets = mst_graph.compute_node_offsets();
    const VectorType<int>& mst_heads_csr = mst_graph.get_heads();
    const VectorType<float>& mst_costs_csr = mst_graph.get_costs();

    // 4. BFS on the sparse MST graph to find tree paths and fan-triangulate.
    return find_conflicted_cycles_bfs<VectorType>(
        rep_edge_tails, rep_edge_heads, mst_offsets, mst_heads_csr,
        mst_costs_csr, rep_edge_costs,
        num_nodes, max_cycle_length, verbose);
}
