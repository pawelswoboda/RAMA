#pragma once

#include <iostream>
#include <vector>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <set>
#include <functional>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include "mst_boruvka.h"
#include "test.h"
#include "random_graph.h"

// Helper: copy VectorType to std::vector for easy inspection.
template<typename T, template<typename> class VectorType>
std::vector<T> to_std(const VectorType<T>& v) {
    std::vector<T> result(v.size());
    thrust::copy(v.begin(), v.end(), result.begin());
    return result;
}

// Helper: compute total MST cost.
template<template<typename> class VectorType>
float mst_cost(const VectorType<float>& costs) {
    auto c = to_std<float, VectorType>(costs);
    float s = 0;
    for (float x : c) s += x;
    return s;
}

// Helper: symmetrize single-direction edges for MST input.
// Takes single-direction edges (i < j) and produces both directions.
inline void symmetrize(
    const std::vector<int>& t_in, const std::vector<int>& h_in, const std::vector<float>& c_in,
    std::vector<int>& t_out, std::vector<int>& h_out, std::vector<float>& c_out)
{
    const int n = t_in.size();
    t_out.resize(2 * n);
    h_out.resize(2 * n);
    c_out.resize(2 * n);
    for (int i = 0; i < n; ++i) {
        t_out[i] = t_in[i];     h_out[i] = h_in[i];     c_out[i] = c_in[i];
        t_out[i+n] = h_in[i];   h_out[i+n] = t_in[i];   c_out[i+n] = c_in[i];
    }
}

// Brute-force Kruskal's maximum spanning tree for verification.
// Input: single-direction edges. Returns total MST cost.
inline float kruskal_max_spanning_tree_cost(
    int num_nodes,
    const std::vector<int>& tails,
    const std::vector<int>& heads,
    const std::vector<float>& costs)
{
    if (tails.empty()) return 0.0f;

    // Sort edges by cost descending (maximum spanning tree)
    std::vector<int> order(tails.size());
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](int a, int b) {
        return costs[a] > costs[b];
    });

    // Union-Find
    std::vector<int> parent(num_nodes);
    std::iota(parent.begin(), parent.end(), 0);
    std::function<int(int)> uf_find = [&](int x) -> int {
        while (parent[x] != x) {
            parent[x] = parent[parent[x]];
            x = parent[x];
        }
        return x;
    };

    float total_cost = 0.0f;
    int edges_added = 0;
    for (int idx : order) {
        int ru = uf_find(tails[idx]);
        int rv = uf_find(heads[idx]);
        if (ru != rv) {
            parent[ru] = rv;
            total_cost += costs[idx];
            edges_added++;
            if (edges_added == num_nodes - 1)
                break;
        }
    }
    return total_cost;
}

// Count connected components in an undirected graph.
inline int count_components(int num_nodes, const std::vector<int>& tails, const std::vector<int>& heads) {
    std::vector<int> parent(num_nodes);
    std::iota(parent.begin(), parent.end(), 0);
    std::function<int(int)> uf_find = [&](int x) -> int {
        while (parent[x] != x) {
            parent[x] = parent[parent[x]];
            x = parent[x];
        }
        return x;
    };
    for (size_t i = 0; i < tails.size(); ++i) {
        int ru = uf_find(tails[i]);
        int rv = uf_find(heads[i]);
        if (ru != rv) parent[ru] = rv;
    }
    std::set<int> roots;
    for (int i = 0; i < num_nodes; ++i)
        roots.insert(uf_find(i));
    return roots.size();
}

// Test 1: Single edge
template<template<typename> class VectorType>
void test_single_edge() {
    std::cout << "  test_single_edge..." << std::flush;
    // Symmetric: 0->1 and 1->0
    std::vector<int> t = {0, 1}, h = {1, 0};
    std::vector<float> c = {5.0f, 5.0f};

    VectorType<int> tails(t.begin(), t.end());
    VectorType<int> heads(h.begin(), h.end());
    VectorType<float> costs(c.begin(), c.end());

    auto [mt, mh, mc] = MST_boruvka::maximum_spanning_tree<VectorType>(tails, heads, costs);
    test(mt.size() == 1, "single edge: expected 1 MST edge");
    test(std::abs(mst_cost<VectorType>(mc) - 5.0f) < 1e-5f, "single edge: wrong cost");
    std::cout << " passed" << std::endl;
}

// Test 2: Path graph (all edges must be in MST)
template<template<typename> class VectorType>
void test_path_graph() {
    std::cout << "  test_path_graph..." << std::flush;
    // 0 --3-- 1 --2-- 2 --1-- 3
    std::vector<int> t1 = {0, 1, 2}, h1 = {1, 2, 3};
    std::vector<float> c1 = {3.0f, 2.0f, 1.0f};
    std::vector<int> t, h; std::vector<float> c;
    symmetrize(t1, h1, c1, t, h, c);

    VectorType<int> tails(t.begin(), t.end());
    VectorType<int> heads(h.begin(), h.end());
    VectorType<float> costs(c.begin(), c.end());

    auto [mt, mh, mc] = MST_boruvka::maximum_spanning_tree<VectorType>(tails, heads, costs);
    test(mt.size() == 3, "path graph: expected 3 MST edges (tree on 4 nodes)");
    test(std::abs(mst_cost<VectorType>(mc) - 6.0f) < 1e-5f, "path graph: wrong cost");
    std::cout << " passed" << std::endl;
}

// Test 3: Triangle (MST drops the lightest edge)
template<template<typename> class VectorType>
void test_triangle() {
    std::cout << "  test_triangle..." << std::flush;
    // 0 --5-- 1, 1 --3-- 2, 0 --7-- 2
    // Maximum spanning tree keeps edges with cost 7 and 5, drops cost 3.
    std::vector<int> t1 = {0, 1, 0}, h1 = {1, 2, 2};
    std::vector<float> c1 = {5.0f, 3.0f, 7.0f};
    std::vector<int> t, h; std::vector<float> c;
    symmetrize(t1, h1, c1, t, h, c);

    VectorType<int> tails(t.begin(), t.end());
    VectorType<int> heads(h.begin(), h.end());
    VectorType<float> costs(c.begin(), c.end());

    auto [mt, mh, mc] = MST_boruvka::maximum_spanning_tree<VectorType>(tails, heads, costs);
    test(mt.size() == 2, "triangle: expected 2 MST edges");
    test(std::abs(mst_cost<VectorType>(mc) - 12.0f) < 1e-5f, "triangle: wrong cost (expected 5+7=12)");
    std::cout << " passed" << std::endl;
}

// Test 4: Disconnected graph (spanning forest)
template<template<typename> class VectorType>
void test_disconnected() {
    std::cout << "  test_disconnected..." << std::flush;
    // Component 1: 0 --3-- 1
    // Component 2: 2 --4-- 3
    // Node 4 is isolated (not reachable, but n_vertices inferred from max endpoint)
    std::vector<int> t1 = {0, 2}, h1 = {1, 3};
    std::vector<float> c1 = {3.0f, 4.0f};
    std::vector<int> t, h; std::vector<float> c;
    symmetrize(t1, h1, c1, t, h, c);

    VectorType<int> tails(t.begin(), t.end());
    VectorType<int> heads(h.begin(), h.end());
    VectorType<float> costs(c.begin(), c.end());

    auto [mt, mh, mc] = MST_boruvka::maximum_spanning_tree<VectorType>(tails, heads, costs);
    test(mt.size() == 2, "disconnected: expected 2 MST edges (spanning forest)");
    test(std::abs(mst_cost<VectorType>(mc) - 7.0f) < 1e-5f, "disconnected: wrong cost");
    std::cout << " passed" << std::endl;
}

// Test 5: Empty graph
template<template<typename> class VectorType>
void test_empty() {
    std::cout << "  test_empty..." << std::flush;
    VectorType<int> tails, heads;
    VectorType<float> costs;

    auto [mt, mh, mc] = MST_boruvka::maximum_spanning_tree<VectorType>(tails, heads, costs);
    test(mt.size() == 0, "empty: expected 0 MST edges");
    std::cout << " passed" << std::endl;
}

// Test 6: Random graphs verified against brute-force Kruskal
template<template<typename> class VectorType>
void test_random_graphs() {
    std::cout << "  test_random_graphs..." << std::flush;

    const std::vector<int> sizes = {5, 10, 15, 20, 30, 40};
    const std::vector<double> probs = {0.3, 0.5, 0.8, 1.0};

    for (int n : sizes) {
        for (double p : probs) {
            for (unsigned seed = 0; seed < 3; ++seed) {
                auto rg = generate_random_graph(n, p, seed * 1000 + n * 100 + (int)(p * 10));
                if (rg.tails.empty()) continue;

                // Symmetrize for MST input
                std::vector<int> st, sh; std::vector<float> sc;
                symmetrize(rg.tails, rg.heads, rg.costs, st, sh, sc);

                VectorType<int> tails(st.begin(), st.end());
                VectorType<int> heads(sh.begin(), sh.end());
                VectorType<float> costs(sc.begin(), sc.end());

                auto [mt, mh, mc] = MST_boruvka::maximum_spanning_tree<VectorType>(tails, heads, costs);

                float boruvka_cost = mst_cost<VectorType>(mc);
                float kruskal_cost = kruskal_max_spanning_tree_cost(n, rg.tails, rg.heads, rg.costs);

                // Verify cost matches Kruskal
                test(std::abs(boruvka_cost - kruskal_cost) < 1e-3f,
                    "random graph (n=" + std::to_string(n) + ", p=" + std::to_string(p) +
                    ", seed=" + std::to_string(seed) + "): cost mismatch, boruvka=" +
                    std::to_string(boruvka_cost) + " kruskal=" + std::to_string(kruskal_cost));

                // Verify MST is a tree/forest (correct number of edges)
                auto mt_std = to_std<int, VectorType>(mt);
                auto mh_std = to_std<int, VectorType>(mh);
                int n_components = count_components(n, rg.tails, rg.heads);
                int expected_edges = n - n_components;
                test((int)mt_std.size() == expected_edges,
                    "random graph (n=" + std::to_string(n) + "): expected " +
                    std::to_string(expected_edges) + " edges, got " + std::to_string(mt_std.size()));
            }
        }
    }
    std::cout << " passed" << std::endl;
}

// Test 7: Graph with equal-weight edges (tests tie-breaking)
template<template<typename> class VectorType>
void test_equal_weights() {
    std::cout << "  test_equal_weights..." << std::flush;
    // Complete graph K4 with all edges having cost 1.0
    std::vector<int> t1 = {0, 0, 0, 1, 1, 2};
    std::vector<int> h1 = {1, 2, 3, 2, 3, 3};
    std::vector<float> c1 = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f};
    std::vector<int> t, h; std::vector<float> c;
    symmetrize(t1, h1, c1, t, h, c);

    VectorType<int> tails(t.begin(), t.end());
    VectorType<int> heads(h.begin(), h.end());
    VectorType<float> costs(c.begin(), c.end());

    auto [mt, mh, mc] = MST_boruvka::maximum_spanning_tree<VectorType>(tails, heads, costs);
    test(mt.size() == 3, "equal weights K4: expected 3 MST edges");
    test(std::abs(mst_cost<VectorType>(mc) - 3.0f) < 1e-5f, "equal weights K4: wrong cost");
    std::cout << " passed" << std::endl;
}

template<template<typename> class VectorType>
void run_all_mst_boruvka_tests() {
    std::cout << "Running MST Boruvka tests..." << std::endl;
    test_single_edge<VectorType>();
    test_path_graph<VectorType>();
    test_triangle<VectorType>();
    test_disconnected<VectorType>();
    test_empty<VectorType>();
    test_equal_weights<VectorType>();
    test_random_graphs<VectorType>();
    std::cout << "All MST Boruvka tests passed." << std::endl;
}
