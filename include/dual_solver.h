#pragma once

#include "conflicted_cycles.h"
#include "multicut_message_passing.h"
#include "time_measure_util.h"
#include <iostream>

template<template<typename> class VectorType>
double dual_solver(Graph<VectorType>& G, int max_cycle_length, int num_iter,
                   int num_outer_itr = 1, float tol_ratio = 1e-4, bool verbose = true)
{
    MEASURE_CUMULATIVE_FUNCTION_EXECUTION_TIME

    if (max_cycle_length < 3 || num_iter == 0 || num_outer_itr == 0)
    {
        multicut_message_passing<VectorType> mp(G, false);
        return mp.lower_bound();
    }

    multicut_message_passing<VectorType> mp(G, verbose);
    double final_lb = mp.lower_bound();

    try {
        double prev_outer_lb = final_lb;
        for (int outer_itr = 0; outer_itr < num_outer_itr; outer_itr++)
        {
            Graph<VectorType> reparam_G = mp.reparametrized_graph();
            auto [v1, v2, v3] = conflicted_cycles<VectorType>(reparam_G, max_cycle_length, tol_ratio, verbose);

            int num_new = 0;
            if (!v1.empty())
                num_new = mp.add_triangles(std::move(v1), std::move(v2), std::move(v3));

            if (num_new == 0)
                break;

            if (verbose)
                std::cout << "outer " << outer_itr << ": added " << num_new << " new triangles\n";

            double prev_iter_lb = 0;
            for (int iter = 0; iter < num_iter; ++iter)
            {
                const double lb = mp.lower_bound();
                if (verbose)
                    std::cout << "outer " << outer_itr << ", iteration " << iter << ", lower bound: " << lb << "\n";
                if (iter > 0 && (lb - prev_iter_lb) < 1e-3)
                    break;
                mp.iteration();
                prev_iter_lb = lb;
            }

            final_lb = mp.lower_bound();
            if (verbose)
                std::cout << "outer " << outer_itr << " final lower bound: " << final_lb << "\n";

            if (outer_itr > 0 && (final_lb - prev_outer_lb) < 1e-3)
                break;
            prev_outer_lb = final_lb;
        }
    }
    catch (const std::bad_alloc& ex) {
        std::cerr << "Dual solver out of memory, returning current lower bound\n";
    }

    G = mp.reparametrized_graph();
    return final_lb;
}
