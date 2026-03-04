#pragma once

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <thrust/gather.h>
#include <thrust/scatter.h>
#include <thrust/reduce.h>
#include <thrust/transform.h>
#include <thrust/equal.h>
#include <thrust/functional.h>
#include <thrust/copy.h>
#include <thrust/swap.h>

namespace connected_components {

// Compute connected components of an undirected graph via Thrust label propagation.
// Input: symmetric COO edges (both directions must be present) and num_nodes.
// Output: vector of size num_nodes where labels[i] = component representative for node i.
//
// Algorithm: iterative min-label propagation with pointer jumping.
// Edges are pre-sorted once by source; each iteration gathers neighbor labels,
// reduces min per source, scatters, and does one pointer-jump step for O(log n) convergence.
template<template<typename> class VectorType>
VectorType<int> compute_cc(const int num_nodes,
                           const VectorType<int>& tails,
                           const VectorType<int>& heads)
{
    const int m = tails.size();

    VectorType<int> labels(num_nodes);
    thrust::sequence(labels.begin(), labels.end());

    if (num_nodes == 0 || m == 0)
        return labels;

    // Pre-sort edges by source (once, outside the loop)
    VectorType<int> sorted_src(tails);
    VectorType<int> sorted_dst(heads);
    thrust::sort_by_key(sorted_src.begin(), sorted_src.end(), sorted_dst.begin());

    VectorType<int> neighbor_labels(m);
    VectorType<int> unique_src(m);
    VectorType<int> min_labels(m);
    VectorType<int> labels_next(num_nodes);
    VectorType<int> labels_jumped(num_nodes);

    bool done = false;
    while (!done) {
        // Gather label[dst] for each edge (sorted by src)
        thrust::gather(sorted_dst.begin(), sorted_dst.end(),
                       labels.begin(),
                       neighbor_labels.begin());

        // Reduce min label per source node
        auto end = thrust::reduce_by_key(
            sorted_src.begin(), sorted_src.end(),
            neighbor_labels.begin(),
            unique_src.begin(),
            min_labels.begin(),
            thrust::equal_to<int>(),
            thrust::minimum<int>());
        int k = end.first - unique_src.begin();

        // Start with current labels, scatter minima
        thrust::copy(labels.begin(), labels.end(), labels_next.begin());
        thrust::scatter(min_labels.begin(), min_labels.begin() + k,
                        unique_src.begin(),
                        labels_next.begin());

        // Enforce min(self, neighbor)
        thrust::transform(labels.begin(), labels.end(),
                          labels_next.begin(),
                          labels_next.begin(),
                          thrust::minimum<int>());

        // Pointer jumping (path compression) into separate buffer
        thrust::gather(labels_next.begin(), labels_next.end(),
                       labels_next.begin(),
                       labels_jumped.begin());

        done = thrust::equal(labels.begin(), labels.end(),
                             labels_jumped.begin());

        labels.swap(labels_jumped);
    }

    return labels;
}

} // namespace connected_components

// Explicit instantiation declarations.
extern template
thrust::host_vector<int>
connected_components::compute_cc<thrust::host_vector>(
    int, const thrust::host_vector<int>&, const thrust::host_vector<int>&);

extern template
thrust::device_vector<int>
connected_components::compute_cc<thrust::device_vector>(
    int, const thrust::device_vector<int>&, const thrust::device_vector<int>&);
