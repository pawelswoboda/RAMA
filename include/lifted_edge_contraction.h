#pragma once

#include "graph.h"
#include "rama_utils.h"

#include <iostream>
#include <vector>

#include <thrust/sequence.h>
#include <thrust/fill.h>
#include <thrust/copy.h>
#include <thrust/for_each.h>
#include <thrust/count.h>
#include <thrust/iterator/counting_iterator.h>

#ifdef __CUDACC__
#define LEC_HOST_DEVICE __host__ __device__
#else
#define LEC_HOST_DEVICE
#endif

// For each lifted edge (u,v) with positive reparametrized cost, find the
// shortest base-graph path from u to v using only edges with non-negative
// reparametrized cost.  If such a path exists, merge all nodes along it.
// Returns (node_mapping, num_contracted_edges).
template<template<typename> class VectorType>
std::tuple<VectorType<int>, int>
find_lifted_contraction_mapping(
    const Graph<VectorType>& base_G,
    const Graph<VectorType>& lifted_G,
    bool verbose = true)
{
    const int num_nodes = base_G.num_nodes();

    if (lifted_G.num_directed_edges() == 0 || base_G.num_directed_edges() == 0)
    {
        VectorType<int> node_mapping(num_nodes);
        thrust::sequence(node_mapping.begin(), node_mapping.end());
        return {std::move(node_mapping), 0};
    }

    const int base_num_edges = base_G.num_directed_edges();
    const int* base_t_ptr = base_G.get_tails_ptr();
    const int* base_h_ptr = base_G.get_heads_ptr();
    const float* base_c_ptr = base_G.get_costs_ptr();

    // Node mapping: initially identity
    VectorType<int> node_mapping(num_nodes);
    thrust::sequence(node_mapping.begin(), node_mapping.end());
    int* nm_ptr = thrust::raw_pointer_cast(node_mapping.data());

    // Reusable BFS arrays
    VectorType<int> dist(num_nodes);
    VectorType<int> parent(num_nodes);
    VectorType<int> on_path(num_nodes);

    // Copy lifted edge endpoints to host for the outer loop
    const size_t num_lifted = lifted_G.num_directed_edges();
    std::vector<int> lt_h(num_lifted), lh_h(num_lifted);
    std::vector<float> lc_h(num_lifted);
    thrust::copy(lifted_G.get_tails().begin(), lifted_G.get_tails().end(), lt_h.begin());
    thrust::copy(lifted_G.get_heads().begin(), lifted_G.get_heads().end(), lh_h.begin());
    thrust::copy(lifted_G.get_costs().begin(), lifted_G.get_costs().end(), lc_h.begin());

    int total_path_edges = 0;

    for (size_t e = 0; e < num_lifted; ++e)
    {
        const int u = lt_h[e];
        const int v = lh_h[e];
        if (u >= v) continue;
        if (lc_h[e] <= 0.0f) continue;

        // Level-synchronous BFS from u through non-negative base edges
        thrust::fill(dist.begin(), dist.end(), num_nodes);
        thrust::fill(parent.begin(), parent.end(), -1);
        thrust::fill(dist.begin() + u, dist.begin() + u + 1, 0);
        thrust::fill(parent.begin() + u, parent.begin() + u + 1, u);

        int* dist_ptr = thrust::raw_pointer_cast(dist.data());
        int* parent_ptr = thrust::raw_pointer_cast(parent.data());

        bool found = false;
        for (int level = 0; level < num_nodes && !found; ++level)
        {
            thrust::for_each(
                thrust::make_counting_iterator(0),
                thrust::make_counting_iterator(base_num_edges),
                [base_t_ptr, base_h_ptr, base_c_ptr,
                 dist_ptr, parent_ptr, level] LEC_HOST_DEVICE (int e) {
                    if (base_c_ptr[e] >= 0.0f)
                    {
                        const int t = base_t_ptr[e];
                        const int h = base_h_ptr[e];
                        if (dist_ptr[t] == level && dist_ptr[h] > level + 1)
                        {
                            dist_ptr[h] = level + 1;
                            parent_ptr[h] = t;
                        }
                    }
                });

            int dist_v;
            thrust::copy_n(dist.begin() + v, 1, &dist_v);
            if (dist_v < num_nodes)
                found = true;
        }

        if (!found) continue;

        // Mark path nodes by propagating from v through parent pointers
        thrust::fill(on_path.begin(), on_path.end(), 0);
        thrust::fill(on_path.begin() + v, on_path.begin() + v + 1, 1);

        int path_len;
        thrust::copy_n(dist.begin() + v, 1, &path_len);

        int* on_path_ptr = thrust::raw_pointer_cast(on_path.data());

        for (int iter = 0; iter < path_len; ++iter)
        {
            thrust::for_each(
                thrust::make_counting_iterator(0),
                thrust::make_counting_iterator(num_nodes),
                [on_path_ptr, parent_ptr] LEC_HOST_DEVICE (int i) {
                    if (on_path_ptr[i] == 1)
                        on_path_ptr[parent_ptr[i]] = 1;
                });
        }

        // Set node_mapping[i] = parent[i] for path nodes (except root u)
        thrust::for_each(
            thrust::make_counting_iterator(0),
            thrust::make_counting_iterator(num_nodes),
            [on_path_ptr, parent_ptr, nm_ptr] LEC_HOST_DEVICE (int i) {
                if (on_path_ptr[i] == 1 && parent_ptr[i] != i)
                    nm_ptr[i] = parent_ptr[i];
            });

        total_path_edges += path_len;
    }

    if (total_path_edges == 0)
    {
        thrust::sequence(node_mapping.begin(), node_mapping.end());
        return {std::move(node_mapping), 0};
    }

    // Pointer jumping to flatten the mapping
    for (int iter = 0; iter < 32; ++iter)
    {
        int changed = thrust::count_if(
            thrust::make_counting_iterator(0),
            thrust::make_counting_iterator(num_nodes),
            [nm_ptr] LEC_HOST_DEVICE (int i) {
                return nm_ptr[i] != nm_ptr[nm_ptr[i]];
            });
        if (changed == 0) break;

        thrust::for_each(
            thrust::make_counting_iterator(0),
            thrust::make_counting_iterator(num_nodes),
            [nm_ptr] LEC_HOST_DEVICE (int i) {
                nm_ptr[i] = nm_ptr[nm_ptr[i]];
            });
    }

    if (verbose)
        std::cout << "lifted path contraction: " << total_path_edges
                  << " path edges contracted\n";

    return {compress_label_sequence<VectorType>(node_mapping, num_nodes - 1),
            total_path_edges};
}
