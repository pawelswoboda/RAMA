#pragma once

#include <iostream>
#include <vector>
#include <set>
#include <algorithm>
#include <numeric>
#include <functional>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/copy.h>
#include "connected_components.h"
#include "test.h"
#include "random_graph.h"

template<typename T, template<typename> class VectorType>
std::vector<T> to_std(const VectorType<T>& v) {
    std::vector<T> result(v.size());
    thrust::copy(v.begin(), v.end(), result.begin());
    return result;
}

// Symmetrize single-direction edges.
inline void symmetrize(
    const std::vector<int>& t_in, const std::vector<int>& h_in,
    std::vector<int>& t_out, std::vector<int>& h_out)
{
    const int n = t_in.size();
    t_out.resize(2 * n);
    h_out.resize(2 * n);
    for (int i = 0; i < n; ++i) {
        t_out[i] = t_in[i];     h_out[i] = h_in[i];
        t_out[i+n] = h_in[i];   h_out[i+n] = t_in[i];
    }
}

// Reference CPU union-find for verification.
inline std::vector<int> reference_cc(int num_nodes,
                                     const std::vector<int>& tails,
                                     const std::vector<int>& heads)
{
    std::vector<int> parent(num_nodes);
    std::iota(parent.begin(), parent.end(), 0);
    std::function<int(int)> find = [&](int x) -> int {
        while (parent[x] != x) {
            parent[x] = parent[parent[x]];
            x = parent[x];
        }
        return x;
    };
    for (size_t i = 0; i < tails.size(); ++i) {
        int u = find(tails[i]);
        int v = find(heads[i]);
        if (u != v) parent[u] = v;
    }
    std::vector<int> labels(num_nodes);
    for (int i = 0; i < num_nodes; i++)
        labels[i] = find(i);
    return labels;
}

// Count distinct component IDs.
inline int count_components(const std::vector<int>& labels) {
    std::set<int> s(labels.begin(), labels.end());
    return s.size();
}

// Verify that all neighbors share the same component label.
inline void verify_cc(const std::vector<int>& labels,
                      const std::vector<int>& tails,
                      const std::vector<int>& heads)
{
    for (size_t i = 0; i < tails.size(); i++) {
        test(labels[tails[i]] == labels[heads[i]],
             "adjacent nodes " + std::to_string(tails[i]) + " and " +
             std::to_string(heads[i]) + " have different component labels");
    }
}

// Test 1: Two disconnected components
//   Component 1: 0 -- 1
//   Component 2: 2 -- 3 -- 4
template<template<typename> class VectorType>
void test_two_components() {
    std::cout << "  test_two_components..." << std::flush;
    std::vector<int> t1 = {0, 2, 2}, h1 = {1, 3, 4};
    std::vector<int> t, h;
    symmetrize(t1, h1, t, h);

    VectorType<int> tails(t.begin(), t.end());
    VectorType<int> heads(h.begin(), h.end());

    auto result = connected_components::compute_cc<VectorType>(5, tails, heads);
    auto labels = to_std<int, VectorType>(result);

    test(count_components(labels) == 2, "expected 2 components");
    verify_cc(labels, t, h);
    test(labels[0] == labels[1], "nodes 0 and 1 should be in same component");
    test(labels[2] == labels[3], "nodes 2 and 3 should be in same component");
    test(labels[2] == labels[4], "nodes 2 and 4 should be in same component");
    test(labels[0] != labels[2], "components should be different");
    std::cout << " passed" << std::endl;
}

// Test 2: Single connected component (triangle)
template<template<typename> class VectorType>
void test_single_component() {
    std::cout << "  test_single_component..." << std::flush;
    std::vector<int> t1 = {0, 1, 0}, h1 = {1, 2, 2};
    std::vector<int> t, h;
    symmetrize(t1, h1, t, h);

    VectorType<int> tails(t.begin(), t.end());
    VectorType<int> heads(h.begin(), h.end());

    auto result = connected_components::compute_cc<VectorType>(3, tails, heads);
    auto labels = to_std<int, VectorType>(result);

    test(count_components(labels) == 1, "expected 1 component");
    verify_cc(labels, t, h);
    std::cout << " passed" << std::endl;
}

// Test 3: Isolated nodes (no edges)
template<template<typename> class VectorType>
void test_isolated_nodes() {
    std::cout << "  test_isolated_nodes..." << std::flush;
    VectorType<int> tails, heads;

    auto result = connected_components::compute_cc<VectorType>(4, tails, heads);
    auto labels = to_std<int, VectorType>(result);

    test(count_components(labels) == 4, "expected 4 components (each node isolated)");
    // Each node should have a unique label
    std::set<int> unique_labels(labels.begin(), labels.end());
    test((int)unique_labels.size() == 4, "all labels should be unique");
    std::cout << " passed" << std::endl;
}

// Test 4: Empty graph (0 nodes)
template<template<typename> class VectorType>
void test_empty_graph() {
    std::cout << "  test_empty_graph..." << std::flush;
    VectorType<int> tails, heads;

    auto result = connected_components::compute_cc<VectorType>(0, tails, heads);
    test(result.size() == 0, "expected empty output for 0-node graph");
    std::cout << " passed" << std::endl;
}

// Test 5: Chain graph (all in one component)
template<template<typename> class VectorType>
void test_chain() {
    std::cout << "  test_chain..." << std::flush;
    // 0 -- 1 -- 2 -- 3 -- 4
    std::vector<int> t1 = {0, 1, 2, 3}, h1 = {1, 2, 3, 4};
    std::vector<int> t, h;
    symmetrize(t1, h1, t, h);

    VectorType<int> tails(t.begin(), t.end());
    VectorType<int> heads(h.begin(), h.end());

    auto result = connected_components::compute_cc<VectorType>(5, tails, heads);
    auto labels = to_std<int, VectorType>(result);

    test(count_components(labels) == 1, "expected 1 component for chain");
    verify_cc(labels, t, h);
    std::cout << " passed" << std::endl;
}

// Test 6: Random graphs verified against reference CPU implementation
template<template<typename> class VectorType>
void test_random_graphs() {
    std::cout << "  test_random_graphs..." << std::flush;

    const std::vector<int> sizes = {5, 10, 20, 30, 50};
    const std::vector<double> probs = {0.1, 0.3, 0.5, 0.8};

    for (int n : sizes) {
        for (double p : probs) {
            for (unsigned seed = 0; seed < 3; ++seed) {
                auto rg = generate_random_graph(n, p, seed * 1000 + n * 100 + (int)(p * 10));

                // Symmetrize edges
                std::vector<int> t, h;
                symmetrize(rg.tails, rg.heads, t, h);

                VectorType<int> tails(t.begin(), t.end());
                VectorType<int> heads(h.begin(), h.end());

                auto result = connected_components::compute_cc<VectorType>(n, tails, heads);
                auto labels = to_std<int, VectorType>(result);

                // Verify neighbors share labels
                verify_cc(labels, t, h);

                // Verify component count matches reference
                auto ref_labels = reference_cc(n, t, h);
                int expected_cc = count_components(ref_labels);
                int actual_cc = count_components(labels);
                test(actual_cc == expected_cc,
                     "random graph (n=" + std::to_string(n) + ", p=" + std::to_string(p) +
                     ", seed=" + std::to_string(seed) + "): expected " +
                     std::to_string(expected_cc) + " components, got " +
                     std::to_string(actual_cc));
            }
        }
    }
    std::cout << " passed" << std::endl;
}

template<template<typename> class VectorType>
void run_all_connected_components_tests() {
    std::cout << "Running connected components tests..." << std::endl;
    test_two_components<VectorType>();
    test_single_component<VectorType>();
    test_isolated_nodes<VectorType>();
    test_empty_graph<VectorType>();
    test_chain<VectorType>();
    test_random_graphs<VectorType>();
    std::cout << "All connected components tests passed." << std::endl;
}
