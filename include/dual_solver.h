#pragma once

#include "conflicted_cycles.h"
#include "lifted_cut_constraints.h"
#include "lifted_multicut_utils.h"
#include "multicut_message_passing.h"
#include "time_measure_util.h"
#include <iomanip>
#include <iostream>
#include <vector>

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
                   const float triangle_budget_ratio = 0,
                   const float cut_factor_budget_ratio = 0,
                   const long long max_raw_triangles = 0)
{
    MEASURE_CUMULATIVE_FUNCTION_EXECUTION_TIME;

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
    int cut_budget = 0;
    if (cut_factor_budget_ratio > 0 && lifted_G.num_directed_edges() > 0)
    {
        const int num_pos_lifted = thrust::count_if(
            lifted_G.get_costs().begin(), lifted_G.get_costs().end(),
            [] CC_HOST_DEVICE (float c) { return c > 0; }) / 2;
        cut_budget = (int)(cut_factor_budget_ratio * num_pos_lifted);
    }

    try {
        double prev_outer_lb = final_lb;
        float cycle_start_ratio = 0.5f;
        float cut_start_frac = 0.5f;
        for (int outer_itr = 0; outer_itr < num_outer_itr; outer_itr++)
        {
            Graph<VectorType> reparam_G = mp.reparametrized_graph();

            // Find new cut factors on reparametrized costs using cascading tau
            int num_new_cuts = 0;
            if (use_cut_factors)
            {
                auto [reparam_base, reparam_lifted] =
                    extract_costs_from_union(reparam_G, base_G);

                const float base_mag = -reparam_base.min();
                const float lifted_mag = reparam_lifted.max();

                if (base_mag > 0 && lifted_mag > 0)
                {
                    const float tau_max = std::min(base_mag, lifted_mag);

                    // Build threshold schedule: start strict (high tau), relax to tau=0.
                    std::vector<float> taus;
                    for (float r = cut_start_frac; r > tol_ratio; r *= 0.5f)
                        taus.push_back(r * tau_max);
                    taus.push_back(0.0f);

                    if (verbose)
                    {
                        std::cout << "cut factor search: tau_max=" << tau_max
                                  << ", schedule=[";
                        for (size_t ti = 0; ti < taus.size(); ++ti)
                            std::cout << (ti ? ", " : "") << taus[ti];
                        std::cout << "], budget="
                                  << (cut_budget > 0 ? std::to_string(cut_budget) : "unlimited")
                                  << "\n";
                    }

                    bool found_cuts = false;
                    int total_base_edges = 0, total_lifted_edges = 0;
                    for (const float tau : taus)
                    {
                        auto factors = find_lifted_cut_constraints<VectorType>(
                            reparam_base, reparam_lifted, verbose, tau);
                        if (factors.num_factors > 0)
                        {
                            num_new_cuts += mp.add_cut_factors(
                                factors, reparam_base, reparam_lifted);
                            total_base_edges += (int)factors.base_edge_idx.size();
                            total_lifted_edges += (int)factors.lifted_edge_idx.size();
                            if (verbose)
                                std::cout << "outer " << outer_itr << " tau=" << tau
                                          << ": " << factors.num_factors
                                          << " cut factors ("
                                          << factors.base_edge_idx.size() << " base, "
                                          << factors.lifted_edge_idx.size() << " lifted edges)\n";
                            if (!found_cuts)
                            {
                                // Persist: start one level stricter next time
                                float effective_frac = (tau_max > 0) ? tau / tau_max : 0.0f;
                                cut_start_frac = std::min(effective_frac * 2.0f, 0.5f);
                                found_cuts = true;
                            }
                            if (cut_budget > 0 && num_new_cuts >= cut_budget)
                            {
                                if (verbose)
                                    std::cout << "outer " << outer_itr
                                              << ": cut factor budget reached ("
                                              << num_new_cuts << "/" << cut_budget << ")\n";
                                break;
                            }
                        }
                    }
                    if (!found_cuts)
                        cut_start_frac = tol_ratio;
                }

                total_cut_factors += num_new_cuts;
                if (num_new_cuts > 0)
                {
                    if (verbose)
                        std::cout << "outer " << outer_itr << ": added "
                                  << num_new_cuts << " cut factors total\n";
                }
                else if (verbose)
                    std::cout << "outer " << outer_itr << ": no cut factors found\n";
            }

            // Find new triangles on reparametrized costs
            int num_new_triangles = 0;
            if (max_cycle_length >= 3)
            {
                auto [v1, v2, v3, eff_ratio] = conflicted_cycles<VectorType>(
                    reparam_G, max_cycle_length, tol_ratio, verbose, long_cycle_method,
                    triangle_budget_ratio, cycle_start_ratio, max_raw_triangles);
                // Persist: start one level stricter next time
                cycle_start_ratio = std::min(eff_ratio * 2.0f, 0.5f);
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
                assert(iter == 0 || lb >= prev_iter_lb - 1e-4
                    || (std::cerr << "lower bound decreased: " << std::fixed << std::setprecision(5) << prev_iter_lb << " -> " << lb << std::defaultfloat << "\n", false));
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
                   float triangle_budget_ratio = 0,
                   float cut_factor_budget_ratio = 0,
                   long long max_raw_triangles = 0)
{
    Graph<VectorType> empty_base, empty_lifted;
    return dual_solver(G, empty_base, empty_lifted, false,
                       max_cycle_length, num_iter, num_outer_itr, tol_ratio, verbose,
                       long_cycle_method, triangle_budget_ratio, cut_factor_budget_ratio,
                       max_raw_triangles);
}

// Explicit instantiation declarations.
extern template
double dual_solver<thrust::host_vector>(
    Graph<thrust::host_vector>&, const Graph<thrust::host_vector>&,
    const Graph<thrust::host_vector>&, bool, int, int, int, float, bool,
    const std::string&, float, float, long long);

extern template
double dual_solver<thrust::device_vector>(
    Graph<thrust::device_vector>&, const Graph<thrust::device_vector>&,
    const Graph<thrust::device_vector>&, bool, int, int, int, float, bool,
    const std::string&, float, float, long long);

extern template
double dual_solver<thrust::host_vector>(
    Graph<thrust::host_vector>&, int, int, int, float, bool, const std::string&, float, float, long long);

extern template
double dual_solver<thrust::device_vector>(
    Graph<thrust::device_vector>&, int, int, int, float, bool, const std::string&, float, float, long long);
