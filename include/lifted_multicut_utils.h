#pragma once

#include "graph.h"
#include <thrust/merge.h>
#include <thrust/binary_search.h>
#include <thrust/gather.h>
#include <thrust/set_operations.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/transform.h>
#include <thrust/reduce.h>
#include <thrust/copy.h>
#include <thrust/tuple.h>

#ifdef __CUDACC__
#define LMU_HOST_DEVICE __host__ __device__
#else
#define LMU_HOST_DEVICE
#endif

namespace lmu_detail {

struct encode_edge_func {
    const int* tails;
    const int* heads;
    const long long max_nodes;

    LMU_HOST_DEVICE
    long long operator()(const size_t idx) const {
        return (long long)tails[idx] * max_nodes + (long long)heads[idx];
    }
};

} // namespace lmu_detail

// Create a union graph from base and lifted graphs.
// Both graphs must be sorted and symmetric.
// Edge sets may overlap: duplicate (tail,head) pairs have their costs summed,
// since overlapping base and lifted edges always share the same cut/join status.
template<template<typename> class VectorType>
Graph<VectorType> create_union_graph(
    const Graph<VectorType>& base_graph,
    const Graph<VectorType>& lifted_graph)
{
    if (base_graph.num_directed_edges() == 0)
        throw std::runtime_error("Base graph has no edges. Lifted multicut requires a non-empty base graph.");

    const size_t base_n = base_graph.num_directed_edges();
    const size_t lifted_n = lifted_graph.num_directed_edges();

    if (lifted_n == 0)
    {
        VectorType<int> t(base_graph.get_tails());
        VectorType<int> h(base_graph.get_heads());
        VectorType<float> c(base_graph.get_costs());
        return Graph<VectorType>(base_graph.num_nodes(),
            std::move(t), std::move(h), std::move(c), true, true);
    }

    const int num_nodes = std::max(base_graph.num_nodes(), lifted_graph.num_nodes());

    // Merge sorted edge lists
    VectorType<int> merged_tails(base_n + lifted_n);
    VectorType<int> merged_heads(base_n + lifted_n);
    VectorType<float> merged_costs(base_n + lifted_n);

    auto base_keys_begin = thrust::make_zip_iterator(
        thrust::make_tuple(base_graph.get_tails().begin(), base_graph.get_heads().begin()));
    auto base_keys_end = thrust::make_zip_iterator(
        thrust::make_tuple(base_graph.get_tails().end(), base_graph.get_heads().end()));
    auto lifted_keys_begin = thrust::make_zip_iterator(
        thrust::make_tuple(lifted_graph.get_tails().begin(), lifted_graph.get_heads().begin()));
    auto lifted_keys_end = thrust::make_zip_iterator(
        thrust::make_tuple(lifted_graph.get_tails().end(), lifted_graph.get_heads().end()));

    auto merged_keys_begin = thrust::make_zip_iterator(
        thrust::make_tuple(merged_tails.begin(), merged_heads.begin()));

    thrust::merge_by_key(
        base_keys_begin, base_keys_end,
        lifted_keys_begin, lifted_keys_end,
        base_graph.get_costs().begin(),
        lifted_graph.get_costs().begin(),
        merged_keys_begin,
        merged_costs.begin());

    // Sum costs of duplicate (tail, head) pairs
    VectorType<int> union_tails(base_n + lifted_n);
    VectorType<int> union_heads(base_n + lifted_n);
    VectorType<float> union_costs(base_n + lifted_n);

    auto union_keys_begin = thrust::make_zip_iterator(
        thrust::make_tuple(union_tails.begin(), union_heads.begin()));

    auto reduce_end = thrust::reduce_by_key(
        merged_keys_begin,
        merged_keys_begin + (base_n + lifted_n),
        merged_costs.begin(),
        union_keys_begin,
        union_costs.begin());

    size_t union_n = std::distance(union_keys_begin, reduce_end.first);
    union_tails.resize(union_n);
    union_heads.resize(union_n);
    union_costs.resize(union_n);

    return Graph<VectorType>(num_nodes,
        std::move(union_tails), std::move(union_heads), std::move(union_costs),
        true, true);
}

// Extract reparametrized costs from the union graph back into separate
// base and lifted graphs.
//
// After dual_solver reparametrizes the union graph (potentially adding diagonal
// edges from triangulation), this function:
// 1. Extracts base edge costs by matching edge pairs via binary search
// 2. Returns everything else as lifted (original lifted + new diagonals)
//
// Diagonal edges from triangulation are semantically lifted edges: they connect
// non-adjacent nodes whose cut/join status is determined by base connectivity.
template<template<typename> class VectorType>
std::tuple<Graph<VectorType>, Graph<VectorType>>
extract_costs_from_union(
    const Graph<VectorType>& union_graph,
    const Graph<VectorType>& base_graph)
{
    const size_t union_n = union_graph.num_directed_edges();
    const size_t base_n = base_graph.num_directed_edges();
    const int num_nodes = union_graph.num_nodes();
    const long long MAX = (long long)num_nodes;

    // Encode edges as int64 keys for binary search
    VectorType<long long> union_keys(union_n);
    thrust::transform(
        thrust::make_counting_iterator<size_t>(0),
        thrust::make_counting_iterator<size_t>(union_n),
        union_keys.begin(),
        lmu_detail::encode_edge_func{
            union_graph.get_tails_ptr(),
            union_graph.get_heads_ptr(), MAX});

    VectorType<long long> base_keys(base_n);
    thrust::transform(
        thrust::make_counting_iterator<size_t>(0),
        thrust::make_counting_iterator<size_t>(base_n),
        base_keys.begin(),
        lmu_detail::encode_edge_func{
            base_graph.get_tails_ptr(),
            base_graph.get_heads_ptr(), MAX});

    // 1. Extract base edge costs via binary search
    VectorType<int> positions(base_n);
    thrust::lower_bound(union_keys.begin(), union_keys.end(),
                        base_keys.begin(), base_keys.end(),
                        positions.begin());

    VectorType<float> new_base_costs(base_n);
    thrust::gather(positions.begin(), positions.end(),
                   union_graph.get_costs().begin(), new_base_costs.begin());

    VectorType<int> new_base_tails(base_graph.get_tails());
    VectorType<int> new_base_heads(base_graph.get_heads());

    // 2. Lifted = union \ base
    if (union_n <= base_n)
    {
        return {Graph<VectorType>(num_nodes,
                    std::move(new_base_tails), std::move(new_base_heads),
                    std::move(new_base_costs), true, true),
                Graph<VectorType>()};
    }

    VectorType<long long> lifted_keys(union_n);
    VectorType<float> lifted_costs(union_n);

    auto diff_end = thrust::set_difference_by_key(
        union_keys.begin(), union_keys.end(),
        base_keys.begin(), base_keys.end(),
        union_graph.get_costs().begin(),
        thrust::make_constant_iterator(0.0f),
        lifted_keys.begin(),
        lifted_costs.begin());

    size_t lifted_n = std::distance(lifted_keys.begin(), diff_end.first);

    if (lifted_n == 0)
    {
        return {Graph<VectorType>(num_nodes,
                    std::move(new_base_tails), std::move(new_base_heads),
                    std::move(new_base_costs), true, true),
                Graph<VectorType>()};
    }

    lifted_keys.resize(lifted_n);
    lifted_costs.resize(lifted_n);

    // Decode lifted keys back to (tail, head) pairs
    VectorType<int> lifted_tails(lifted_n);
    VectorType<int> lifted_heads(lifted_n);
    {
        const long long* keys_ptr = thrust::raw_pointer_cast(lifted_keys.data());
        int* t_ptr = thrust::raw_pointer_cast(lifted_tails.data());
        int* h_ptr = thrust::raw_pointer_cast(lifted_heads.data());
        thrust::for_each(
            thrust::make_counting_iterator<size_t>(0),
            thrust::make_counting_iterator<size_t>(lifted_n),
            [keys_ptr, t_ptr, h_ptr, MAX] LMU_HOST_DEVICE (const size_t idx) {
                t_ptr[idx] = (int)(keys_ptr[idx] / MAX);
                h_ptr[idx] = (int)(keys_ptr[idx] % MAX);
            });
    }

    return {Graph<VectorType>(num_nodes,
                std::move(new_base_tails), std::move(new_base_heads),
                std::move(new_base_costs), true, true),
            Graph<VectorType>(num_nodes,
                std::move(lifted_tails), std::move(lifted_heads),
                std::move(lifted_costs), true, true)};
}
