#pragma once

#include "dual_solver.h"
#include "random_graph.h"
#include "test.h"
#include <iostream>
#include <cmath>
#include <string>

template<template<typename> class VectorType>
void test_dual_solver_no_repulsive_edges()
{
    // All-positive graph: no conflicted cycles, lb = 0
    VectorType<int> ei = std::vector<int>{0,0,1};
    VectorType<int> ej = std::vector<int>{1,2,2};
    VectorType<float> ec = std::vector<float>{1.0, 2.0, 3.0};
    Graph<VectorType> G(ei.begin(), ei.end(), ej.begin(), ej.end(), ec.begin(), ec.end());

    double lb = dual_solver<VectorType>(G, 5, 10, 1, 1e-4, false);
    test(std::abs(lb) < 1e-6,
         "no repulsive: lb should be 0, got " + std::to_string(lb));
}

template<template<typename> class VectorType>
void test_dual_solver_single_triangle()
{
    // Single conflicted triangle: one negative edge
    //
    //       0
    //      / \
    //    +1   +1
    //    /     \
    //   1-------2
    //      -1
    //
    // Trivial LB = -1, LP optimal = 0
    VectorType<int> ei = std::vector<int>{0,0,1};
    VectorType<int> ej = std::vector<int>{1,2,2};
    VectorType<float> ec = std::vector<float>{1.0, 1.0, -1.0};
    Graph<VectorType> G(ei.begin(), ei.end(), ej.begin(), ej.end(), ec.begin(), ec.end());

    double lb = dual_solver<VectorType>(G, 3, 20, 1, 1e-4, false);
    test(lb >= -1.0 - 1e-6,
         "single triangle: lb must be >= -1, got " + std::to_string(lb));
    test(lb > -0.5,
         "single triangle: lb should improve significantly, got " + std::to_string(lb));
}

template<template<typename> class VectorType>
void test_dual_solver_lb_non_decreasing()
{
    // Random graph: dual solver should return a reasonable lower bound
    auto rg = generate_random_graph(30, 0.3, 42);
    if (rg.tails.empty()) {
        std::cout << "  skipped (no edges)\n";
        return;
    }

    VectorType<int> vi(rg.tails.begin(), rg.tails.end());
    VectorType<int> vj(rg.heads.begin(), rg.heads.end());
    VectorType<float> vc(rg.costs.begin(), rg.costs.end());
    Graph<VectorType> G(rg.num_nodes, vi.begin(), vi.end(), vj.begin(), vj.end(), vc.begin(), vc.end());

    // Compute trivial LB first
    multicut_message_passing<VectorType> mp_trivial(G, false);
    double trivial_lb = mp_trivial.lower_bound();

    // Run dual solver
    double lb = dual_solver<VectorType>(G, 5, 10, 2, 1e-4, false);
    test(lb >= trivial_lb - 1e-6,
         "random: dual solver lb must be >= trivial lb, trivial=" +
         std::to_string(trivial_lb) + ", dual=" + std::to_string(lb));
}

template<template<typename> class VectorType>
void run_all_dual_solver_tests()
{
    test_dual_solver_no_repulsive_edges<VectorType>();
    std::cout << "PASSED: no repulsive edges\n";

    test_dual_solver_single_triangle<VectorType>();
    std::cout << "PASSED: single triangle\n";

    test_dual_solver_lb_non_decreasing<VectorType>();
    std::cout << "PASSED: lb non-decreasing\n";

    std::cout << "\nAll dual_solver tests passed.\n";
}
