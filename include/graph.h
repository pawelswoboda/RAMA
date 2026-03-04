#pragma once

#include <cassert>
#include <algorithm>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sort.h>
#include <thrust/extrema.h>
#include <thrust/sequence.h>
#include <thrust/gather.h>
#include <thrust/generate.h>
#include <thrust/copy.h>
#include <thrust/reduce.h>
#include <thrust/transform.h>
#include <thrust/tuple.h>
#include <thrust/for_each.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/remove.h>
#include <thrust/unique.h>
#include <thrust/adjacent_difference.h>
#include <thrust/scatter.h>
#include <thrust/scan.h>
#include <thrust/count.h>
#include <thrust/iterator/constant_iterator.h>
#include <iostream>

// Portable host/device annotation for functors.
// When compiled with nvcc (__CUDACC__), expands to __host__ __device__.
// When compiled with a standard C++ compiler, expands to nothing.
#ifdef __CUDACC__
#define GRAPH_HOST_DEVICE __host__ __device__
#else
#define GRAPH_HOST_DEVICE
#endif

// Templatized graph class that stores symmetric edges (i->j and j->i).
// VectorType can be thrust::device_vector or thrust::host_vector.
template<template<typename> class VectorType>
class Graph {
public:
    Graph() {}

    // Constructor from iterators
    template<typename TAIL_ITERATOR, typename HEAD_ITERATOR, typename COST_ITERATOR>
    Graph(const int num_nodes,
          TAIL_ITERATOR tail_begin, TAIL_ITERATOR tail_end,
          HEAD_ITERATOR head_begin, HEAD_ITERATOR head_end,
          COST_ITERATOR cost_begin, COST_ITERATOR cost_end,
          const bool is_sorted = false, const bool is_symmetric = false);

    // Constructor without explicit num_nodes - infers from data
    template<typename TAIL_ITERATOR, typename HEAD_ITERATOR, typename COST_ITERATOR>
    Graph(TAIL_ITERATOR tail_begin, TAIL_ITERATOR tail_end,
          HEAD_ITERATOR head_begin, HEAD_ITERATOR head_end,
          COST_ITERATOR cost_begin, COST_ITERATOR cost_end,
          const bool is_sorted = false, const bool is_symmetric = false);

    // Move constructor from vectors
    Graph(VectorType<int>&& tails, VectorType<int>&& heads,
          VectorType<float>&& costs, const bool is_sorted = false,
          const bool is_symmetric = false);

    Graph(const int num_nodes,
          VectorType<int>&& tails, VectorType<int>&& heads,
          VectorType<float>&& costs, const bool is_sorted = false,
          const bool is_symmetric = false);

    // Accessors
    size_t num_nodes() const { return num_nodes_; }
    size_t num_edges() const { return costs_.size() / 2; }
    size_t num_directed_edges() const { return costs_.size(); }

    // Data statistics
    float sum() const;
    float min() const;
    float max() const;

    // Graph operations
    void remove_self_loops();
    VectorType<float> self_loop_costs() const;
    VectorType<int> compute_node_offsets() const;
    Graph<VectorType> contract(const VectorType<int>& node_mapping) const;

    // Data access
    const int* get_tails_ptr() const { return thrust::raw_pointer_cast(tails_.data()); }
    const int* get_heads_ptr() const { return thrust::raw_pointer_cast(heads_.data()); }
    const float* get_costs_ptr() const { return thrust::raw_pointer_cast(costs_.data()); }
    float* get_writeable_costs_ptr() { return thrust::raw_pointer_cast(costs_.data()); }

    const VectorType<int>& get_tails() const { return tails_; }
    const VectorType<int>& get_heads() const { return heads_; }
    const VectorType<float>& get_costs() const { return costs_; }

    // Return filtered subgraph with costs in [lb, ub]
    Graph<VectorType> filter(const float lb, const float ub) const;

    // Print for debugging
    void print() const;

    // Check that no edge appears in both orientations.
    template<typename TAIL_ITERATOR, typename HEAD_ITERATOR>
    static bool is_single_orientation(TAIL_ITERATOR tail_begin, TAIL_ITERATOR tail_end,
                                      HEAD_ITERATOR head_begin, HEAD_ITERATOR head_end);

    // Check that duplicate (tail,head) pairs exist.
    template<typename TAIL_ITERATOR, typename HEAD_ITERATOR>
    static bool has_duplicate_edges(TAIL_ITERATOR tail_begin, TAIL_ITERATOR tail_end,
                                    HEAD_ITERATOR head_begin, HEAD_ITERATOR head_end);

private:
    void init(const bool is_sorted, const bool is_symmetric = false);
    void ensure_symmetric();

    // Helper for sorting COO format
    void coo_sort(VectorType<int>& i, VectorType<int>& j, VectorType<float>& costs);
    VectorType<int> compute_offsets(const VectorType<int>& i, const int max_value) const;

    int num_nodes_ = 0;
    VectorType<float> costs_;
    VectorType<int> tails_;
    VectorType<int> heads_;
};

// Implementation

template<template<typename> class VectorType>
Graph<VectorType>::Graph(VectorType<int>&& tails, VectorType<int>&& heads,
                         VectorType<float>&& costs, const bool is_sorted,
                         const bool is_symmetric)
    : tails_(std::move(tails)),
      heads_(std::move(heads)),
      costs_(std::move(costs))
{
    init(is_sorted, is_symmetric);
}

template<template<typename> class VectorType>
Graph<VectorType>::Graph(const int num_nodes,
                         VectorType<int>&& tails, VectorType<int>&& heads,
                         VectorType<float>&& costs, const bool is_sorted,
                         const bool is_symmetric)
    : num_nodes_(num_nodes),
      tails_(std::move(tails)),
      heads_(std::move(heads)),
      costs_(std::move(costs))
{
    init(is_sorted, is_symmetric);
}

template<template<typename> class VectorType>
template<typename TAIL_ITERATOR, typename HEAD_ITERATOR, typename COST_ITERATOR>
Graph<VectorType>::Graph(const int num_nodes,
                          TAIL_ITERATOR tail_begin, TAIL_ITERATOR tail_end,
                          HEAD_ITERATOR head_begin, HEAD_ITERATOR head_end,
                          COST_ITERATOR cost_begin, COST_ITERATOR cost_end,
                          const bool is_sorted, const bool is_symmetric)
    : num_nodes_(num_nodes),
      tails_(tail_begin, tail_end),
      heads_(head_begin, head_end),
      costs_(cost_begin, cost_end)
{
    init(is_sorted, is_symmetric);
}

template<template<typename> class VectorType>
template<typename TAIL_ITERATOR, typename HEAD_ITERATOR, typename COST_ITERATOR>
Graph<VectorType>::Graph(TAIL_ITERATOR tail_begin, TAIL_ITERATOR tail_end,
                          HEAD_ITERATOR head_begin, HEAD_ITERATOR head_end,
                          COST_ITERATOR cost_begin, COST_ITERATOR cost_end,
                          const bool is_sorted, const bool is_symmetric)
    : tails_(tail_begin, tail_end),
      heads_(head_begin, head_end),
      costs_(cost_begin, cost_end)
{
    init(is_sorted, is_symmetric);
}

template<template<typename> class VectorType>
void Graph<VectorType>::coo_sort(VectorType<int>& i, VectorType<int>& j, VectorType<float>& costs)
{
    assert(i.size() == j.size());
    assert(i.size() == costs.size());

    auto first = thrust::make_zip_iterator(thrust::make_tuple(i.begin(), j.begin()));
    auto last = thrust::make_zip_iterator(thrust::make_tuple(i.end(), j.end()));
    thrust::sort_by_key(first, last, costs.begin());
}

// Functor that normalizes an edge (i,j) to (min(i,j), max(i,j))
struct normalize_edge_func {
    GRAPH_HOST_DEVICE
    thrust::tuple<int,int> operator()(const thrust::tuple<int,int>& t) const {
        const int a = thrust::get<0>(t);
        const int b = thrust::get<1>(t);
        return (a <= b) ? thrust::make_tuple(a, b) : thrust::make_tuple(b, a);
    }
};

template<template<typename> class VectorType>
template<typename TAIL_ITERATOR, typename HEAD_ITERATOR>
bool Graph<VectorType>::is_single_orientation(
    TAIL_ITERATOR tail_begin, TAIL_ITERATOR tail_end,
    HEAD_ITERATOR head_begin, HEAD_ITERATOR head_end)
{
    const size_t n = std::distance(tail_begin, tail_end);
    assert(n == (size_t)std::distance(head_begin, head_end));
    if (n <= 1)
        return true;

    // Normalize each edge to (min, max) so (i,j) and (j,i) map to the same pair
    VectorType<int> lo(n), hi(n);
    auto edge_begin = thrust::make_zip_iterator(thrust::make_tuple(tail_begin, head_begin));
    auto edge_end   = thrust::make_zip_iterator(thrust::make_tuple(tail_end,   head_end));
    auto norm_begin = thrust::make_zip_iterator(thrust::make_tuple(lo.begin(), hi.begin()));

    thrust::transform(edge_begin, edge_end, norm_begin, normalize_edge_func());

    // Sort the normalized pairs
    auto sort_begin = thrust::make_zip_iterator(thrust::make_tuple(lo.begin(), hi.begin()));
    auto sort_end   = thrust::make_zip_iterator(thrust::make_tuple(lo.end(),   hi.end()));
    thrust::sort(sort_begin, sort_end);

    // If all normalized edges are unique, no edge was provided in both orientations
    auto unique_end = thrust::unique(sort_begin, sort_end);
    size_t num_unique = std::distance(sort_begin, unique_end);

    return num_unique == n;
}

template<template<typename> class VectorType>
template<typename TAIL_ITERATOR, typename HEAD_ITERATOR>
bool Graph<VectorType>::has_duplicate_edges(
    TAIL_ITERATOR tail_begin, TAIL_ITERATOR tail_end,
    HEAD_ITERATOR head_begin, HEAD_ITERATOR head_end)
{
    const size_t n = std::distance(tail_begin, tail_end);
    assert(n == (size_t)std::distance(head_begin, head_end));
    if (n <= 1)
        return false;

    VectorType<int> t(tail_begin, tail_end);
    VectorType<int> h(head_begin, head_end);

    auto sort_begin = thrust::make_zip_iterator(thrust::make_tuple(t.begin(), h.begin()));
    auto sort_end   = thrust::make_zip_iterator(thrust::make_tuple(t.end(),   h.end()));
    thrust::sort(sort_begin, sort_end);

    auto unique_end = thrust::unique(sort_begin, sort_end);
    size_t num_unique = std::distance(sort_begin, unique_end);

    return num_unique != n;
}

template<template<typename> class VectorType>
void Graph<VectorType>::ensure_symmetric()
{
    const size_t nr_edges = costs_.size();

    // Create vectors to hold both directions
    VectorType<int> tails_symm(2 * nr_edges);
    VectorType<int> heads_symm(2 * nr_edges);
    VectorType<float> costs_symm(2 * nr_edges);

    // Copy original edges (i->j)
    thrust::copy(tails_.begin(), tails_.end(), tails_symm.begin());
    thrust::copy(heads_.begin(), heads_.end(), heads_symm.begin());
    thrust::copy(costs_.begin(), costs_.end(), costs_symm.begin());

    // Copy reverse edges (j->i)
    thrust::copy(tails_.begin(), tails_.end(), heads_symm.begin() + nr_edges);
    thrust::copy(heads_.begin(), heads_.end(), tails_symm.begin() + nr_edges);
    thrust::copy(costs_.begin(), costs_.end(), costs_symm.begin() + nr_edges);

    // Replace with symmetric version
    tails_ = std::move(tails_symm);
    heads_ = std::move(heads_symm);
    costs_ = std::move(costs_symm);

    // Sort and remove duplicates
    coo_sort(tails_, heads_, costs_);

    auto first = thrust::make_zip_iterator(thrust::make_tuple(tails_.begin(), heads_.begin()));
    auto last = thrust::make_zip_iterator(thrust::make_tuple(tails_.end(), heads_.end()));
    auto new_end = thrust::unique_by_key(first, last, costs_.begin());

    size_t new_size = std::distance(first, new_end.first);
    tails_.resize(new_size);
    heads_.resize(new_size);
    costs_.resize(new_size);
}

template<template<typename> class VectorType>
void Graph<VectorType>::init(const bool is_sorted, const bool is_symmetric)
{
    assert(tails_.size() == costs_.size());
    assert(heads_.size() == costs_.size());

    assert(!has_duplicate_edges(tails_.begin(), tails_.end(), heads_.begin(), heads_.end()));
    if (!is_symmetric) {
        assert(is_single_orientation(tails_.begin(), tails_.end(), heads_.begin(), heads_.end()));
        ensure_symmetric();
    }

    if (is_sorted) {
        assert(thrust::is_sorted(tails_.begin(), tails_.end()));
        assert(thrust::is_sorted(thrust::make_zip_iterator(thrust::make_tuple(tails_.begin(), heads_.begin())),
                                 thrust::make_zip_iterator(thrust::make_tuple(tails_.end(), heads_.end()))));
    } else {
        coo_sort(tails_, heads_, costs_);
        assert(thrust::is_sorted(tails_.begin(), tails_.end()));
    }

    // Compute num_nodes from data if not provided
    if (num_nodes_ == 0) {
        int max_tail = *thrust::max_element(tails_.begin(), tails_.end());
        int max_head = *thrust::max_element(heads_.begin(), heads_.end());
        num_nodes_ = std::max(max_tail, max_head) + 1;
    }
    assert(num_nodes_ > *thrust::max_element(tails_.begin(), tails_.end()));
    assert(num_nodes_ > *thrust::max_element(heads_.begin(), heads_.end()));
}

template<template<typename> class VectorType>
float Graph<VectorType>::sum() const
{
    return thrust::reduce(costs_.begin(), costs_.end(), (float)0.0, thrust::plus<float>());
}

template<template<typename> class VectorType>
float Graph<VectorType>::min() const
{
    return *thrust::min_element(costs_.begin(), costs_.end());
}

template<template<typename> class VectorType>
float Graph<VectorType>::max() const
{
    return *thrust::max_element(costs_.begin(), costs_.end());
}

// Functor for self-loop check
struct is_self_loop {
    GRAPH_HOST_DEVICE
    bool operator()(const thrust::tuple<int, int, float>& t) const {
        return thrust::get<0>(t) == thrust::get<1>(t);
    }
};

template<template<typename> class VectorType>
void Graph<VectorType>::remove_self_loops()
{
    auto begin = thrust::make_zip_iterator(thrust::make_tuple(tails_.begin(), heads_.begin(), costs_.begin()));
    auto end = thrust::make_zip_iterator(thrust::make_tuple(tails_.end(), heads_.end(), costs_.end()));

    auto new_last = thrust::remove_if(begin, end, is_self_loop());
    size_t new_size = std::distance(begin, new_last);
    tails_.resize(new_size);
    heads_.resize(new_size);
    costs_.resize(new_size);
}

// Functor for extracting self-loop costs
struct extract_self_loop_costs {
    float* d;

    GRAPH_HOST_DEVICE
    void operator()(const thrust::tuple<int, int, float>& t) {
        if (thrust::get<0>(t) == thrust::get<1>(t)) {
            const int idx = thrust::get<0>(t);
            d[idx] = thrust::get<2>(t);
        }
    }
};

template<template<typename> class VectorType>
VectorType<float> Graph<VectorType>::self_loop_costs() const
{
    VectorType<float> d(num_nodes_, 0.0);

    auto begin = thrust::make_zip_iterator(thrust::make_tuple(tails_.begin(), heads_.begin(), costs_.begin()));
    auto end = thrust::make_zip_iterator(thrust::make_tuple(tails_.end(), heads_.end(), costs_.end()));

    extract_self_loop_costs func{thrust::raw_pointer_cast(d.data())};
    thrust::for_each(begin, end, func);

    return d;
}

template<template<typename> class VectorType>
VectorType<int> Graph<VectorType>::compute_offsets(const VectorType<int>& i, const int max_value) const
{
    assert(thrust::is_sorted(i.begin(), i.end()));

    VectorType<int> offsets(max_value + 2, 0);

    // Count occurrences of each index
    VectorType<int> unique_ids(i.size());
    VectorType<int> counts(i.size());

    auto first = i.begin();
    auto last = i.end();

    auto new_end = thrust::unique_by_key_copy(first, last,
                                               thrust::make_counting_iterator(0),
                                               unique_ids.begin(),
                                               counts.begin());

    size_t num_unique = std::distance(unique_ids.begin(), new_end.first);
    unique_ids.resize(num_unique);
    counts.resize(num_unique + 1);
    counts[num_unique] = i.size();

    thrust::adjacent_difference(counts.begin(), counts.end(), counts.begin());
    VectorType<int> final_counts(counts.begin() + 1, counts.end());

    // Scatter counts to appropriate positions
    thrust::transform(unique_ids.begin(), unique_ids.end(),
                     thrust::make_constant_iterator<int>(1),
                     unique_ids.begin(),
                     thrust::plus<int>());
    thrust::scatter(final_counts.begin(), final_counts.end(), unique_ids.begin(), offsets.begin());
    thrust::inclusive_scan(offsets.begin(), offsets.end(), offsets.begin());

    return offsets;
}

template<template<typename> class VectorType>
VectorType<int> Graph<VectorType>::compute_node_offsets() const
{
    return compute_offsets(tails_, num_nodes_ - 1);
}

// Functor for range filtering
struct is_in_range_func {
    const float lb;
    const float ub;

    GRAPH_HOST_DEVICE
    bool operator()(const float x) const {
        return x >= lb && x <= ub;
    }

    GRAPH_HOST_DEVICE
    bool operator()(const thrust::tuple<int, int, float>& t) const {
        return operator()(thrust::get<2>(t));
    }
};

template<template<typename> class VectorType>
Graph<VectorType> Graph<VectorType>::filter(const float lb, const float ub) const
{
    assert(lb <= ub);

    const size_t new_count = thrust::count_if(costs_.begin(), costs_.end(), is_in_range_func{lb, ub});
    VectorType<int> tails_f(new_count), heads_f(new_count);
    VectorType<float> costs_f(new_count);

    auto first = thrust::make_zip_iterator(thrust::make_tuple(tails_.begin(), heads_.begin(), costs_.begin()));
    auto last = thrust::make_zip_iterator(thrust::make_tuple(tails_.end(), heads_.end(), costs_.end()));
    auto first_f = thrust::make_zip_iterator(thrust::make_tuple(tails_f.begin(), heads_f.begin(), costs_f.begin()));

    thrust::copy_if(first, last, first_f, is_in_range_func{lb, ub});

    Graph<VectorType> result;
    result.num_nodes_ = num_nodes_;
    result.tails_ = std::move(tails_f);
    result.heads_ = std::move(heads_f);
    result.costs_ = std::move(costs_f);
    result.init(true, true);
    return result;
}

template<template<typename> class VectorType>
Graph<VectorType> Graph<VectorType>::contract(const VectorType<int>& node_mapping) const
{
    assert(node_mapping.size() >= num_nodes_);

    // Step 1: Keep only forward edges (tail < head) — one per undirected edge
    auto all_begin = thrust::make_zip_iterator(thrust::make_tuple(tails_.begin(), heads_.begin(), costs_.begin()));
    auto all_end = thrust::make_zip_iterator(thrust::make_tuple(tails_.end(), heads_.end(), costs_.end()));

    auto is_fwd = [] GRAPH_HOST_DEVICE (const thrust::tuple<int, int, float>& t) {
        return thrust::get<0>(t) < thrust::get<1>(t);
    };

    const size_t num_forward = thrust::count_if(all_begin, all_end, is_fwd);

    if (num_forward == 0) {
        int new_num_nodes = *thrust::max_element(node_mapping.begin(), node_mapping.end()) + 1;
        Graph<VectorType> result;
        result.num_nodes_ = new_num_nodes;
        return result;
    }

    VectorType<int> fwd_tails(num_forward), fwd_heads(num_forward);
    VectorType<float> fwd_costs(num_forward);

    auto fwd_begin = thrust::make_zip_iterator(thrust::make_tuple(fwd_tails.begin(), fwd_heads.begin(), fwd_costs.begin()));
    thrust::copy_if(all_begin, all_end, fwd_begin, is_fwd);

    // Step 2: Map endpoints through node_mapping
    VectorType<int> mapped_tails(num_forward), mapped_heads(num_forward);
    thrust::gather(fwd_tails.begin(), fwd_tails.end(), node_mapping.begin(), mapped_tails.begin());
    thrust::gather(fwd_heads.begin(), fwd_heads.end(), node_mapping.begin(), mapped_heads.begin());

    // Step 3: Normalize to (min, max) — some edges may become self-loops
    auto edge_begin = thrust::make_zip_iterator(thrust::make_tuple(mapped_tails.begin(), mapped_heads.begin()));
    auto edge_end = thrust::make_zip_iterator(thrust::make_tuple(mapped_tails.end(), mapped_heads.end()));
    auto normalize = [] GRAPH_HOST_DEVICE (const thrust::tuple<int, int>& t) {
        const int a = thrust::get<0>(t);
        const int b = thrust::get<1>(t);
        return (a <= b) ? thrust::make_tuple(a, b) : thrust::make_tuple(b, a);
    };
    thrust::transform(edge_begin, edge_end, edge_begin, normalize);

    // Step 4: Sort by (tail, head)
    thrust::sort_by_key(edge_begin, edge_end, fwd_costs.begin());

    // Step 5: reduce_by_key to sum costs of edges with same mapped endpoints
    VectorType<int> out_tails(num_forward), out_heads(num_forward);
    VectorType<float> out_costs(num_forward);

    auto out_begin = thrust::make_zip_iterator(thrust::make_tuple(out_tails.begin(), out_heads.begin()));
    auto new_end = thrust::reduce_by_key(edge_begin, edge_end, fwd_costs.begin(), out_begin, out_costs.begin());

    size_t new_size = std::distance(out_costs.begin(), new_end.second);
    out_tails.resize(new_size);
    out_heads.resize(new_size);
    out_costs.resize(new_size);

    int new_num_nodes = *thrust::max_element(node_mapping.begin(), node_mapping.end()) + 1;

    // is_symmetric=false triggers ensure_symmetric() to add reverse edges
    return Graph<VectorType>(new_num_nodes, std::move(out_tails), std::move(out_heads),
                             std::move(out_costs), false, false);
}

template<template<typename> class VectorType>
void Graph<VectorType>::print() const
{
    std::cout << "Graph:\n";
    std::cout << "  nodes: " << num_nodes_ << ", edges: " << num_edges()
              << " (" << num_directed_edges() << " directed)\n";

    // Print first few edges
    std::cout << "  First edges (tail, head, cost):\n";
    size_t n = std::min(size_t(10), num_directed_edges());

    // For device vectors, we need to copy to host first
    thrust::host_vector<int> h_tails(tails_.begin(), tails_.begin() + n);
    thrust::host_vector<int> h_heads(heads_.begin(), heads_.begin() + n);
    thrust::host_vector<float> h_costs(costs_.begin(), costs_.begin() + n);

    for (size_t i = 0; i < n; ++i) {
        std::cout << "    (" << h_tails[i] << ", " << h_heads[i] << ", " << h_costs[i] << ")\n";
    }
}

// Explicit instantiation declarations — suppress implicit instantiation of
// non-template member functions. Definitions in src/graph_cpu.cpp and src/graph_gpu.cu.
extern template class Graph<thrust::host_vector>;
extern template class Graph<thrust::device_vector>;

// Type aliases for convenience
using DeviceGraph = Graph<thrust::device_vector>;
using HostGraph = Graph<thrust::host_vector>;
