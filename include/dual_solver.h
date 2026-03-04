#pragma once

#include "conflicted_cycles.h"
#include "lifted_cut_constraints.h"
#include "lifted_multicut_utils.h"
#include "multicut_message_passing.h"
#include "time_measure_util.h"
#include <iostream>

// Dual solver with optional lifted cut factor support.
// When use_cut_factors is true, iteratively finds cut constraints on the
// reparametrized graph alongside triangle constraints each outer iteration.
template<template<typename> class VectorType>
double dual_solver(Graph<VectorType>& G,
                   const Graph<VectorType>& base_G,
                   const Graph<VectorType>& lifted_G,
                   const bool use_cut_factors,
                   const int max_cycle_length, const int num_iter,
                   const int num_outer_itr = 1, const float tol_ratio = 1e-4,
                   const bool verbose = true,
                   const std::string& long_cycle_method = "bfs",
                   const float triangle_budget_ratio = 0)
{
    MEASURE_CUMULATIVE_FUNCTION_EXECUTION_TIME

    if (num_iter == 0 || num_outer_itr == 0)
    {
        multicut_message_passing<VectorType> mp(G, false);
        return mp.lower_bound();
    }

    if (max_cycle_length < 3 && !use_cut_factors)
    {
        multicut_message_passing<VectorType> mp(G, false);
        return mp.lower_bound();
    }

    multicut_message_passing<VectorType> mp(G, verbose);
    double final_lb = mp.lower_bound();
    int total_cut_factors = 0;

    try {
        double prev_outer_lb = final_lb;
        for (int outer_itr = 0; outer_itr < num_outer_itr; outer_itr++)
        {
            Graph<VectorType> reparam_G = mp.reparametrized_graph();

            // Find new cut factors on reparametrized costs
            int num_new_cuts = 0;
            std::cout << "Use cut factors: " << use_cut_factors << "\n";
            if (use_cut_factors)
            {
                auto [reparam_base, reparam_lifted] =
                    extract_costs_from_union(reparam_G, base_G);
                auto factors = find_lifted_cut_constraints<VectorType>(
                    reparam_base, reparam_lifted, verbose);
                if (factors.num_factors > 0)
                {
                    num_new_cuts = mp.add_cut_factors(factors, reparam_base, reparam_lifted);
                    total_cut_factors += num_new_cuts;
                    if (verbose)
                        std::cout << "outer " << outer_itr << ": added "
                                  << num_new_cuts << " cut factors\n";
                }
                else if (verbose)
                    std::cout << "outer " << outer_itr << ": no cut factors found\n";
            }

            // Find new triangles on reparametrized costs
            int num_new_triangles = 0;
            if (max_cycle_length >= 3)
            {
                auto [v1, v2, v3] = conflicted_cycles<VectorType>(
                    reparam_G, max_cycle_length, tol_ratio, verbose, long_cycle_method,
                    triangle_budget_ratio);
                if (!v1.empty())
                    num_new_triangles = mp.add_triangles(
                        std::move(v1), std::move(v2), std::move(v3));
            }

            if (num_new_triangles == 0 && num_new_cuts == 0)
                break;

            if (verbose)
                std::cout << "outer " << outer_itr << ": added "
                          << num_new_triangles << " new triangles\n";

            double prev_iter_lb = 0;
            for (int iter = 0; iter < num_iter; ++iter)
            {
                const double lb = mp.lower_bound();
                if (verbose)
                    std::cout << "outer " << outer_itr << ", iteration "
                              << iter << ", lower bound: " << lb << "\n";
                if (iter > 0 && (lb - prev_iter_lb) < 1e-3)
                    break;
                mp.iteration();
                prev_iter_lb = lb;
            }

            final_lb = mp.lower_bound();
            if (verbose)
                std::cout << "outer " << outer_itr << " final lower bound: "
                          << final_lb << "\n";

            if (outer_itr > 0 && (final_lb - prev_outer_lb) < 1e-3)
                break;
            prev_outer_lb = final_lb;
        }
    }
    catch (const std::bad_alloc& ex) {
        std::cerr << "Dual solver out of memory, returning current lower bound\n";
    }

    if (verbose && total_cut_factors > 0)
        std::cout << "total cut factors added: " << total_cut_factors << "\n";

    G = mp.reparametrized_graph();
    return final_lb;
}

// Standard multicut dual solver (no lifted edges).
template<template<typename> class VectorType>
double dual_solver(Graph<VectorType>& G, int max_cycle_length, int num_iter,
                   int num_outer_itr = 1, float tol_ratio = 1e-4, bool verbose = true,
                   const std::string& long_cycle_method = "bfs",
                   float triangle_budget_ratio = 0)
{
    Graph<VectorType> empty_base, empty_lifted;
    return dual_solver(G, empty_base, empty_lifted, false,
                       max_cycle_length, num_iter, num_outer_itr, tol_ratio, verbose,
                       long_cycle_method, triangle_budget_ratio);
}

// Explicit instantiation declarations.
extern template
double dual_solver<thrust::host_vector>(
    Graph<thrust::host_vector>&, const Graph<thrust::host_vector>&,
    const Graph<thrust::host_vector>&, bool, int, int, int, float, bool,
    const std::string&, float);

extern template
double dual_solver<thrust::device_vector>(
    Graph<thrust::device_vector>&, const Graph<thrust::device_vector>&,
    const Graph<thrust::device_vector>&, bool, int, int, int, float, bool,
    const std::string&, float);

extern template
double dual_solver<thrust::host_vector>(
    Graph<thrust::host_vector>&, int, int, int, float, bool, const std::string&, float);

extern template
double dual_solver<thrust::device_vector>(
    Graph<thrust::device_vector>&, int, int, int, float, bool, const std::string&, float);
