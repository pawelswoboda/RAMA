#pragma once

#include "multicut_message_passing.h"
#include "random_graph.h"
#include "test.h"
#include <iostream>
#include <cmath>
#include <string>
#include <random>
#include <set>
#include <numeric>
#include <algorithm>

template<template<typename> class VectorType>
void test_single_triangle_all_positive()
{
    // Single triangle with all positive costs:
    //
    //       0
    //      / \
    //    +1   +2
    //    /     \
    //   1-------2
    //      +3
    //
    // All positive costs: optimal is no cuts, LB = 0
    VectorType<int> i = std::vector<int>{0,0,1};
    VectorType<int> j = std::vector<int>{1,2,2};
    VectorType<float> costs = std::vector<float>{1.0, 2.0, 3.0};
    VectorType<int> t1 = std::vector<int>{0};
    VectorType<int> t2 = std::vector<int>{1};
    VectorType<int> t3 = std::vector<int>{2};
    Graph<VectorType> A(i.begin(), i.end(), j.begin(), j.end(), costs.begin(), costs.end());
    multicut_message_passing<VectorType> mcp(A, std::move(t1), std::move(t2), std::move(t3));

    double initial_lb = mcp.lower_bound();
    test(std::abs(initial_lb) < 1e-6,
         "all positive: initial lb must be 0, got " + std::to_string(initial_lb));

    mcp.iteration();
    mcp.iteration();

    double final_lb = mcp.lower_bound();
    test(std::abs(final_lb) < 1e-6,
         "all positive: lb stays 0 after iterations, got " + std::to_string(final_lb));
}

template<template<typename> class VectorType>
void test_single_triangle_all_negative()
{
    // Single triangle with all negative costs:
    //
    //       0
    //      / \
    //    -1   -1
    //    /     \
    //   1-------2
    //      -1
    //
    // Can cut all 3 edges (each node in separate cluster)
    // Optimal = -3, LB should stay at -3 (no improvement needed)
    VectorType<int> i = std::vector<int>{0,0,1};
    VectorType<int> j = std::vector<int>{1,2,2};
    VectorType<float> costs = std::vector<float>{-1.0, -1.0, -1.0};
    VectorType<int> t1 = std::vector<int>{0};
    VectorType<int> t2 = std::vector<int>{1};
    VectorType<int> t3 = std::vector<int>{2};
    Graph<VectorType> A(i.begin(), i.end(), j.begin(), j.end(), costs.begin(), costs.end());
    multicut_message_passing<VectorType> mcp(A, std::move(t1), std::move(t2), std::move(t3));

    const double initial_lb = mcp.lower_bound();
    test(std::abs(initial_lb - (-3.0)) < 1e-6,
         "all negative: initial lb must be -3, got " + std::to_string(initial_lb));

    for (int iter = 0; iter < 10; ++iter)
        mcp.iteration();

    const double final_lb = mcp.lower_bound();
    // LB should NOT improve - we can achieve -3 by cutting all edges
    test(std::abs(final_lb - (-3.0)) < 1e-4,
         "all negative: lb should stay at -3, got " + std::to_string(final_lb));
}

template<template<typename> class VectorType>
void test_single_triangle_one_negative()
{
    // Single triangle with one negative edge:
    //
    //       0
    //      / \
    //    +1   +1
    //    /     \
    //   1-------2
    //      -1
    //
    // Triangle inequality: x_12 <= x_01 + x_02
    // Cannot cut only (1,2) without violating constraint
    // Trivial LB = -1, LP optimal = 0 (improvement by ~1)
    VectorType<int> i = std::vector<int>{0,0,1};
    VectorType<int> j = std::vector<int>{1,2,2};
    VectorType<float> costs = std::vector<float>{1.0, 1.0, -1.0};
    VectorType<int> t1 = std::vector<int>{0};
    VectorType<int> t2 = std::vector<int>{1};
    VectorType<int> t3 = std::vector<int>{2};
    Graph<VectorType> A(i.begin(), i.end(), j.begin(), j.end(), costs.begin(), costs.end());
    multicut_message_passing<VectorType> mcp(A, std::move(t1), std::move(t2), std::move(t3));

    const double initial_lb = mcp.lower_bound();
    test(std::abs(initial_lb - (-1.0)) < 1e-6,
         "one negative: initial lb must be -1, got " + std::to_string(initial_lb));

    for (int iter = 0; iter < 10; ++iter)
        mcp.iteration();

    const double final_lb = mcp.lower_bound();
    test(final_lb > initial_lb - 1e-6,
         "one negative: lb must not decrease, initial=" + std::to_string(initial_lb) + ", final=" + std::to_string(final_lb));
    test(std::abs(final_lb - 0.0) < 1e-4,
         "one negative: lb should converge to 0, got " + std::to_string(final_lb));
}

template<template<typename> class VectorType>
void test_lower_bound_non_decreasing()
{
    // With any graph, the lower bound should be monotonically non-decreasing
    VectorType<int> i = std::vector<int>{0,0,0,1,2};
    VectorType<int> j = std::vector<int>{1,2,3,3,3};
    VectorType<float> costs = std::vector<float>{1.0,1.0,-2.0,1.0,1.0};
    VectorType<int> t1 = std::vector<int>{0,0};
    VectorType<int> t2 = std::vector<int>{1,2};
    VectorType<int> t3 = std::vector<int>{3,3};
    Graph<VectorType> A(i.begin(), i.end(), j.begin(), j.end(), costs.begin(), costs.end());
    multicut_message_passing<VectorType> mcp(A, std::move(t1), std::move(t2), std::move(t3));

    double prev_lb = mcp.lower_bound();
    for (int iter = 0; iter < 20; ++iter) {
        mcp.iteration();
        double lb = mcp.lower_bound();
        test(lb >= prev_lb - 1e-6, "lb must be non-decreasing across iterations");
        prev_lb = lb;
    }
}

template<template<typename> class VectorType>
void test_reparametrized_edges_cover_original()
{
    // The reparametrized edge set should include all original edges
    VectorType<int> i = std::vector<int>{0,0,0,1,2};
    VectorType<int> j = std::vector<int>{1,2,3,3,3};
    VectorType<float> costs = std::vector<float>{1.0,1.0,-2.0,1.0,1.0};
    VectorType<int> t1 = std::vector<int>{0,0};
    VectorType<int> t2 = std::vector<int>{1,2};
    VectorType<int> t3 = std::vector<int>{3,3};
    Graph<VectorType> A(i.begin(), i.end(), j.begin(), j.end(), costs.begin(), costs.end());
    multicut_message_passing<VectorType> mcp(A, std::move(t1), std::move(t2), std::move(t3));

    mcp.iteration();

    const auto& [ri, rj, rc] = mcp.reparametrized_edge_costs();
    test(ri.size() == rj.size(), "reparametrized: i and j must have same size");
    test(ri.size() == rc.size(), "reparametrized: i and costs must have same size");
    test(ri.size() >= 5, "reparametrized: must have at least 5 edges (original count)");
}

template<template<typename> class VectorType>
void test_edges_not_in_triangles_preserved()
{
    // Triangle plus one extra edge not in any triangle:
    //
    //       0
    //      / \
    //    +1   -1
    //    /     \
    //   1-------2-------3
    //      +1      -5
    //
    // Triangle: (0,1,2)
    // Extra edge: (2,3) not in any triangle
    // Edge (2,3) should be preserved unchanged by message passing
    VectorType<int> i = std::vector<int>{0,0,1,2};
    VectorType<int> j = std::vector<int>{1,2,2,3};
    VectorType<float> costs = std::vector<float>{1.0, -1.0, 1.0, -5.0};
    VectorType<int> t1 = std::vector<int>{0};
    VectorType<int> t2 = std::vector<int>{1};
    VectorType<int> t3 = std::vector<int>{2};
    Graph<VectorType> A(i.begin(), i.end(), j.begin(), j.end(), costs.begin(), costs.end());
    multicut_message_passing<VectorType> mcp(A, std::move(t1), std::move(t2), std::move(t3));

    const auto& [ri, rj, rc] = mcp.reparametrized_edge_costs();
    test(ri.size() == 4, "non-triangle edges: must have all 4 edges");

    // The non-triangle edge (2,3) cost should be unchanged since it's not in any triangle
    // Find it in the output
    thrust::host_vector<int> h_i = ri;
    thrust::host_vector<int> h_j = rj;
    thrust::host_vector<float> h_c = rc;
    bool found = false;
    for (size_t e = 0; e < h_i.size(); ++e) {
        if (h_i[e] == 2 && h_j[e] == 3) {
            test(std::abs(h_c[e] - (-5.0f)) < 1e-6, "non-triangle edge cost must be unchanged");
            found = true;
        }
    }
    test(found, "non-triangle edge (2,3) must be present");
}

template<template<typename> class VectorType>
void test_two_triangles_shared_negative_edge()
{
    // Two triangles sharing vertical edge (1,2):
    //
    //        1
    //       /|\
    //    +1/ | \+1
    //     /  |  \
    //    0  -1   3
    //     \  |  /
    //    +1\ | /+1
    //       \|/
    //        2
    //
    // Triangles: (0,1,2) and (1,2,3)
    // Edges: (0,1)=+1, (0,2)=+1, (1,2)=-1, (1,3)=+1, (2,3)=+1
    // Shared edge (1,2)=-1 is the vertical edge
    // Trivial LB = -1 (only shared edge negative)
    // After message passing: should improve to ~0
    VectorType<int> i = std::vector<int>{0,0,1,1,2};
    VectorType<int> j = std::vector<int>{1,2,2,3,3};
    VectorType<float> costs = std::vector<float>{1.0, 1.0, -1.0, 1.0, 1.0};
    VectorType<int> t1 = std::vector<int>{0,1};
    VectorType<int> t2 = std::vector<int>{1,2};
    VectorType<int> t3 = std::vector<int>{2,3};
    Graph<VectorType> A(i.begin(), i.end(), j.begin(), j.end(), costs.begin(), costs.end());
    multicut_message_passing<VectorType> mcp(A, std::move(t1), std::move(t2), std::move(t3));

    const double initial_lb = mcp.lower_bound();
    test(std::abs(initial_lb - (-1.0)) < 1e-6,
         "two triangles: initial lb must be -1, got " + std::to_string(initial_lb));

    double prev_lb = initial_lb;
    for (int iter = 0; iter < 50; ++iter) {
        mcp.iteration();
        double lb = mcp.lower_bound();
        test(lb >= prev_lb - 1e-6,
             "two triangles: lb must be non-decreasing, prev=" + std::to_string(prev_lb) + ", current=" + std::to_string(lb));
        prev_lb = lb;
    }

    double final_lb = mcp.lower_bound();
    test(final_lb <= 1e-6,
         "lower bound should be at most 0, got " + std::to_string(final_lb));
    test(final_lb >= -0.004,
         "lower bound should be close to 0 after optimization, got " + std::to_string(final_lb));
}

template<template<typename> class VectorType>
void test_send_messages_to_triplets_only()
{
    // After send_messages_to_triplets only (no send back), lb should not decrease
    VectorType<int> i = std::vector<int>{0,0,1};
    VectorType<int> j = std::vector<int>{1,2,2};
    VectorType<float> costs = std::vector<float>{-1.0, -1.0, -1.0};
    VectorType<int> t1 = std::vector<int>{0};
    VectorType<int> t2 = std::vector<int>{1};
    VectorType<int> t3 = std::vector<int>{2};
    Graph<VectorType> A(i.begin(), i.end(), j.begin(), j.end(), costs.begin(), costs.end());
    multicut_message_passing<VectorType> mcp(A, std::move(t1), std::move(t2), std::move(t3));

    const double lb_before = mcp.lower_bound();
    mcp.send_messages_to_triplets();
    const double lb_after = mcp.lower_bound();

    test(lb_after >= lb_before - 1e-6, "send_to_triplets: lb must not decrease");

    // All 3 edges are covered by the single triangle, so their edge costs must be zero
    const auto& [ri, rj, rc] = mcp.reparametrized_edge_costs();
    thrust::host_vector<float> h_rc = rc;
    for (size_t e = 0; e < h_rc.size(); ++e) {
        test(std::abs(h_rc[e]) < 1e-6,
             "send_to_triplets: edge " + std::to_string(e) + " cost should be 0, got " + std::to_string(h_rc[e]));
    }
}

template<template<typename> class VectorType>
void test_random_graph_message_passing(const int num_nodes, const double density, const unsigned seed = 42)
{
    auto rg = generate_random_graph(num_nodes, density, seed);

    test(!rg.tails.empty(), "random graph (n=" + std::to_string(num_nodes) + ", d=" + std::to_string(density) + "): no edges generated");

    std::set<std::pair<int,int>> edge_set;
    for (size_t e = 0; e < rg.tails.size(); ++e)
        edge_set.insert({rg.tails[e], rg.heads[e]});

    std::mt19937 gen(seed);

    // Find all triangles in the graph
    std::vector<int> h_t1, h_t2, h_t3;
    for (int u = 0; u < num_nodes; ++u) {
        for (int v = u + 1; v < num_nodes; ++v) {
            if (edge_set.count({u, v}) == 0) continue;
            for (int w = v + 1; w < num_nodes; ++w) {
                if (edge_set.count({u, w}) && edge_set.count({v, w})) {
                    h_t1.push_back(u);
                    h_t2.push_back(v);
                    h_t3.push_back(w);
                }
            }
        }
    }

    if(h_t1.empty()) {
        std::cout << "random graph (n=" + std::to_string(num_nodes) + ", d=" + std::to_string(density) + "): no triangles found, returning\n";
        return;
    }

    // Randomly select a subset of triangles (at least 1, up to half)
    std::vector<int> tri_indices(h_t1.size());
    std::iota(tri_indices.begin(), tri_indices.end(), 0);
    std::shuffle(tri_indices.begin(), tri_indices.end(), gen);
    const int num_selected = std::max(1, (int)(h_t1.size() / 2));
    tri_indices.resize(num_selected);
    std::sort(tri_indices.begin(), tri_indices.end());

    std::vector<int> sel_t1, sel_t2, sel_t3;
    for (int idx : tri_indices) {
        sel_t1.push_back(h_t1[idx]);
        sel_t2.push_back(h_t2[idx]);
        sel_t3.push_back(h_t3[idx]);
    }

    // Verify all selected triangles have all 3 edges in the graph
    for (size_t t = 0; t < sel_t1.size(); ++t) {
        const int a = sel_t1[t], b = sel_t2[t], c = sel_t3[t];
        const std::string tri_str = "(" + std::to_string(a) + "," + std::to_string(b) + "," + std::to_string(c) + ")";
        test(a < b && b < c, "triangle vertices not sorted: " + tri_str);
        test(edge_set.count({a, b}) > 0, "triangle " + tri_str + " missing edge (" + std::to_string(a) + "," + std::to_string(b) + ")");
        test(edge_set.count({a, c}) > 0, "triangle " + tri_str + " missing edge (" + std::to_string(a) + "," + std::to_string(c) + ")");
        test(edge_set.count({b, c}) > 0, "triangle " + tri_str + " missing edge (" + std::to_string(b) + "," + std::to_string(c) + ")");
    }

    // Build Graph and run message passing
    VectorType<int> vi(rg.tails.begin(), rg.tails.end());
    VectorType<int> vj(rg.heads.begin(), rg.heads.end());
    VectorType<float> vc(rg.costs.begin(), rg.costs.end());
    VectorType<int> vt1(sel_t1.begin(), sel_t1.end());
    VectorType<int> vt2(sel_t2.begin(), sel_t2.end());
    VectorType<int> vt3(sel_t3.begin(), sel_t3.end());

    Graph<VectorType> A(num_nodes, vi.begin(), vi.end(), vj.begin(), vj.end(), vc.begin(), vc.end());
    multicut_message_passing<VectorType> mcp(A, std::move(vt1), std::move(vt2), std::move(vt3), false);

    // Lower bound must be non-decreasing across iterations
    double prev_lb = mcp.lower_bound();
    bool lb_improved = false;
    for (int iter = 0; iter < 50; ++iter) {
        mcp.iteration();
        const double lb = mcp.lower_bound();
        const double tol = 1e-6 * std::abs(prev_lb) + 1e-6;
        test(lb >= prev_lb - tol,
             "random graph: lb decreased from " + std::to_string(prev_lb) + " to " + std::to_string(lb) + " at iter " + std::to_string(iter));
        if (lb > prev_lb + 1e-6)
            lb_improved = true;
        prev_lb = lb;
    }

    std::cout << "  n=" << num_nodes << " edges=" << rg.tails.size()
              << " triangles=" << sel_t1.size() << "/" << h_t1.size()
              << " lb_improved=" << (lb_improved ? "yes" : "no") << "\n";

    // With random mixed costs and triangle constraints, message passing
    // should improve the lower bound in virtually all cases
    if(!lb_improved)
        std::cout <<  "random graph (n=" + std::to_string(num_nodes) + ", d=" + std::to_string(density) +
            ", seed=" + std::to_string(seed) + "): lb did not improve after 50 iterations, should be very rare\n";
}

template<template<typename> class VectorType>
void test_add_triangles_incremental()
{
    // Two triangles sharing edge (1,2):
    //
    //        1
    //       /|\
    //    +1/ | \+1
    //     /  |  \
    //    0  -1   3
    //     \  |  /
    //    +1\ | /+1
    //       \|/
    //        2
    //
    VectorType<int> ei = std::vector<int>{0,0,1,1,2};
    VectorType<int> ej = std::vector<int>{1,2,2,3,3};
    VectorType<float> ec = std::vector<float>{1.0, 1.0, -1.0, 1.0, 1.0};
    Graph<VectorType> A(ei.begin(), ei.end(), ej.begin(), ej.end(), ec.begin(), ec.end());

    multicut_message_passing<VectorType> mp(A, false);
    double lb0 = mp.lower_bound();
    test(std::abs(lb0 - (-1.0)) < 1e-6,
         "incremental: trivial lb must be -1, got " + std::to_string(lb0));

    // Add first triangle (0,1,2)
    VectorType<int> ta1 = std::vector<int>{0};
    VectorType<int> ta2 = std::vector<int>{1};
    VectorType<int> ta3 = std::vector<int>{2};
    int added1 = mp.add_triangles(std::move(ta1), std::move(ta2), std::move(ta3));
    test(added1 == 1, "incremental: first add should return 1, got " + std::to_string(added1));

    for (int iter = 0; iter < 20; ++iter)
        mp.iteration();
    double lb1 = mp.lower_bound();
    test(lb1 >= lb0 - 1e-6,
         "incremental: lb after first triangle must not decrease");

    // Add second triangle (1,2,3)
    VectorType<int> tb1 = std::vector<int>{1};
    VectorType<int> tb2 = std::vector<int>{2};
    VectorType<int> tb3 = std::vector<int>{3};
    int added2 = mp.add_triangles(std::move(tb1), std::move(tb2), std::move(tb3));
    test(added2 == 1, "incremental: second add should return 1, got " + std::to_string(added2));

    for (int iter = 0; iter < 20; ++iter)
        mp.iteration();
    double lb2 = mp.lower_bound();
    test(lb2 >= lb1 - 1e-6,
         "incremental: lb after second triangle must not decrease");
}

template<template<typename> class VectorType>
void test_add_triangles_deduplication()
{
    VectorType<int> ei = std::vector<int>{0,0,1};
    VectorType<int> ej = std::vector<int>{1,2,2};
    VectorType<float> ec = std::vector<float>{1.0, 1.0, -1.0};
    Graph<VectorType> A(ei.begin(), ei.end(), ej.begin(), ej.end(), ec.begin(), ec.end());

    multicut_message_passing<VectorType> mp(A, false);

    VectorType<int> ta1 = std::vector<int>{0};
    VectorType<int> ta2 = std::vector<int>{1};
    VectorType<int> ta3 = std::vector<int>{2};
    int added1 = mp.add_triangles(std::move(ta1), std::move(ta2), std::move(ta3));
    test(added1 == 1, "dedup: first add should return 1");

    for (int iter = 0; iter < 5; ++iter)
        mp.iteration();
    double lb_before = mp.lower_bound();

    // Add same triangle again
    VectorType<int> tb1 = std::vector<int>{0};
    VectorType<int> tb2 = std::vector<int>{1};
    VectorType<int> tb3 = std::vector<int>{2};
    int added2 = mp.add_triangles(std::move(tb1), std::move(tb2), std::move(tb3));
    test(added2 == 0, "dedup: second add of same triangle should return 0, got " + std::to_string(added2));

    double lb_after = mp.lower_bound();
    test(std::abs(lb_after - lb_before) < 1e-6,
         "dedup: lb should be unchanged after duplicate add");
}

template<template<typename> class VectorType>
void test_reparametrized_graph()
{
    VectorType<int> ei = std::vector<int>{0,0,1,2};
    VectorType<int> ej = std::vector<int>{1,2,2,3};
    VectorType<float> ec = std::vector<float>{1.0, -1.0, 1.0, -5.0};
    VectorType<int> t1 = std::vector<int>{0};
    VectorType<int> t2 = std::vector<int>{1};
    VectorType<int> t3 = std::vector<int>{2};
    Graph<VectorType> A(ei.begin(), ei.end(), ej.begin(), ej.end(), ec.begin(), ec.end());
    multicut_message_passing<VectorType> mp(A, std::move(t1), std::move(t2), std::move(t3), false);

    for (int iter = 0; iter < 10; ++iter)
        mp.iteration();

    Graph<VectorType> G = mp.reparametrized_graph();
    test(G.num_nodes() == 4, "reparam graph: should have 4 nodes, got " + std::to_string(G.num_nodes()));
    test(G.num_edges() == 4, "reparam graph: should have 4 undirected edges, got " + std::to_string(G.num_edges()));

    // Non-triangle edge (2,3) cost should be unchanged
    thrust::host_vector<int> h_tails = G.get_tails();
    thrust::host_vector<int> h_heads = G.get_heads();
    thrust::host_vector<float> h_costs = G.get_costs();
    bool found = false;
    for (size_t e = 0; e < h_tails.size(); ++e) {
        if ((h_tails[e] == 2 && h_heads[e] == 3) || (h_tails[e] == 3 && h_heads[e] == 2)) {
            test(std::abs(h_costs[e] - (-5.0f)) < 1e-6,
                 "reparam graph: non-triangle edge cost must be unchanged");
            found = true;
            break;
        }
    }
    test(found, "reparam graph: edge (2,3) must be present");
}

template<template<typename> class VectorType>
void test_add_triangles_preserves_costs()
{
    // Two triangles sharing edge (1,2):
    VectorType<int> ei = std::vector<int>{0,0,1,1,2};
    VectorType<int> ej = std::vector<int>{1,2,2,3,3};
    VectorType<float> ec = std::vector<float>{1.0, 1.0, -1.0, 1.0, 1.0};
    Graph<VectorType> A(ei.begin(), ei.end(), ej.begin(), ej.end(), ec.begin(), ec.end());

    // Create MP with first triangle only
    VectorType<int> ta1 = std::vector<int>{0};
    VectorType<int> ta2 = std::vector<int>{1};
    VectorType<int> ta3 = std::vector<int>{2};
    multicut_message_passing<VectorType> mp(A, std::move(ta1), std::move(ta2), std::move(ta3), false);

    // Run iterations to reparametrize
    for (int iter = 0; iter < 10; ++iter)
        mp.iteration();
    double lb_before_add = mp.lower_bound();

    // Add second triangle
    VectorType<int> tb1 = std::vector<int>{1};
    VectorType<int> tb2 = std::vector<int>{2};
    VectorType<int> tb3 = std::vector<int>{3};
    mp.add_triangles(std::move(tb1), std::move(tb2), std::move(tb3));

    // Lower bound should be preserved (adding triangles doesn't change costs)
    double lb_after_add = mp.lower_bound();
    test(std::abs(lb_after_add - lb_before_add) < 1e-4,
         "preserves costs: lb should be preserved after adding triangles, before=" +
         std::to_string(lb_before_add) + ", after=" + std::to_string(lb_after_add));

    // Running more iterations should not decrease lb
    for (int iter = 0; iter < 20; ++iter)
        mp.iteration();
    double lb_final = mp.lower_bound();
    test(lb_final >= lb_after_add - 1e-6,
         "preserves costs: lb should not decrease after more iterations");
}

template<template<typename> class VectorType>
void run_all_multicut_message_passing_tests()
{
    test_single_triangle_all_positive<VectorType>();
    std::cout << "PASSED: single triangle all positive\n";

    test_single_triangle_one_negative<VectorType>();
    std::cout << "PASSED: single triangle one negative\n";

    test_single_triangle_all_negative<VectorType>();
    std::cout << "PASSED: single triangle all negative\n";

    test_lower_bound_non_decreasing<VectorType>();
    std::cout << "PASSED: lower bound non-decreasing\n";

    test_reparametrized_edges_cover_original<VectorType>();
    std::cout << "PASSED: reparametrized edges cover original\n";

    test_edges_not_in_triangles_preserved<VectorType>();
    std::cout << "PASSED: edges not in triangles preserved\n";

    test_two_triangles_shared_negative_edge<VectorType>();
    std::cout << "PASSED: two triangles shared negative edge\n";

    test_send_messages_to_triplets_only<VectorType>();
    std::cout << "PASSED: send messages to triplets only\n";

    test_random_graph_message_passing<VectorType>(30, 0.3, 42);
    std::cout << "PASSED: random graph (n=30, density=0.3)\n";

    test_random_graph_message_passing<VectorType>(50, 0.2, 123);
    std::cout << "PASSED: random graph (n=50, density=0.2)\n";

    for(float density : std::vector<float>({0.1, 0.2, 0.4, 0.8})) {
        for(int n=10; n<100; n+=27) {
            test_random_graph_message_passing<VectorType>(n, density, 7);
            std::cout << "PASSED: random graph (n=" << std::to_string(n) << ", density=" << std::to_string(density) << ")\n";
        }
    }

    test_add_triangles_incremental<VectorType>();
    std::cout << "PASSED: add triangles incremental\n";

    test_add_triangles_deduplication<VectorType>();
    std::cout << "PASSED: add triangles deduplication\n";

    test_reparametrized_graph<VectorType>();
    std::cout << "PASSED: reparametrized graph\n";

    test_add_triangles_preserves_costs<VectorType>();
    std::cout << "PASSED: add triangles preserves costs\n";

    std::cout << "\nAll multicut_message_passing tests passed.\n";
}
