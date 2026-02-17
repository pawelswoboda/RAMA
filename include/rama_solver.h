#pragma once

#include "graph.h"
#include "dual_solver.h"
#include "edge_contractions.h"
#include "maximum_matching.h"
#include "multicut_message_passing.h"
#include "multicut_solver_options.h"
#include "rama_utils.h"
#include "time_measure_util.h"

#include <vector>
#include <tuple>
#include <chrono>
#include <iostream>
#include <stdexcept>

#include <thrust/sequence.h>
#include <thrust/count.h>
#include <thrust/for_each.h>
#include <thrust/copy.h>
#include <thrust/reduce.h>

#ifdef __CUDACC__
#define SOLVER_HOST_DEVICE __host__ __device__
#else
#define SOLVER_HOST_DEVICE
#endif

namespace rama_solver_detail {

template<template<typename> class VectorType>
bool has_bad_contractions(const Graph<VectorType>& G)
{
    VectorType<float> d = G.self_loop_costs();
    auto is_neg = [] SOLVER_HOST_DEVICE (const float x) { return x < 0.0f; };
    return thrust::count_if(d.begin(), d.end(), is_neg) > 0;
}

template<template<typename> class VectorType>
void map_node_labels(const VectorType<int>& cur_node_mapping, VectorType<int>& orig_node_mapping)
{
    const int* cur_ptr = thrust::raw_pointer_cast(cur_node_mapping.data());
    int* orig_ptr = thrust::raw_pointer_cast(orig_node_mapping.data());
    const unsigned long num_nodes_cont = cur_node_mapping.size();

    thrust::for_each(
        thrust::make_counting_iterator<int>(0),
        thrust::make_counting_iterator<int>((int)orig_node_mapping.size()),
        [cur_ptr, orig_ptr, num_nodes_cont] SOLVER_HOST_DEVICE (const int n) {
            const int n_map = orig_ptr[n];
            if (n_map < (int)num_nodes_cont)
                orig_ptr[n] = cur_ptr[n_map];
        });
}

template<template<typename> class VectorType>
std::tuple<VectorType<int>, int> contraction_mapping_by_maximum_matching(
    const Graph<VectorType>& G, const float mean_multiplier_mm, const bool verbose)
{
    VectorType<int> node_mapping;
    int nr_matched;
    std::tie(node_mapping, nr_matched) =
        filter_edges_by_matching<VectorType>(G, mean_multiplier_mm, verbose);
    return {compress_label_sequence<VectorType>(node_mapping, node_mapping.size() - 1), nr_matched};
}

} // namespace rama_solver_detail

// Templatized multicut solver. Works on both CPU (thrust::host_vector) and GPU (thrust::device_vector).
// Returns (node_mapping, lower_bound, timeline).
template<template<typename> class VectorType>
std::tuple<VectorType<int>, double, std::vector<std::vector<int>>>
rama_solver(Graph<VectorType>& G, const multicut_solver_options& opts)
{
    MEASURE_CUMULATIVE_FUNCTION_EXECUTION_TIME;
    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

    // Dual solve for lower bound (reparametrizes G in-place)
    const double final_lb = dual_solver<VectorType>(G,
        opts.max_cycle_length_lb, opts.num_dual_itr_lb,
        opts.num_outer_itr_dual, 1e-4, opts.verbose);

    if (opts.verbose)
        std::cout << "initial energy = " << G.sum() << "\n";

    VectorType<int> node_mapping(G.num_nodes());
    thrust::sequence(node_mapping.begin(), node_mapping.end());

    std::vector<std::vector<int>> timeline;

    if (opts.only_compute_lb)
        return {VectorType<int>(), final_lb, timeline};

    bool try_matching = true;
    if (opts.matching_thresh_crossover_ratio > 1.0)
        try_matching = false;

    for (size_t iter = 0; G.num_directed_edges() > 0; ++iter)
    {
        if (iter > 0)
            dual_solver<VectorType>(G, opts.max_cycle_length_primal,
                                    opts.num_dual_itr_primal, 1, 1e-4, opts.verbose);

        VectorType<int> cur_node_mapping;
        int nr_edges_to_contract;

        if (try_matching)
        {
            std::tie(cur_node_mapping, nr_edges_to_contract) =
                rama_solver_detail::contraction_mapping_by_maximum_matching<VectorType>(
                    G, opts.mean_multiplier_mm, opts.verbose);

            if (nr_edges_to_contract < (int)G.num_nodes() * opts.matching_thresh_crossover_ratio)
            {
                if (opts.verbose)
                {
                    std::cout << "# edges to contract = " << nr_edges_to_contract
                              << ", # vertices = " << G.num_nodes() << "\n";
                    std::cout << "switching to MST based contraction edge selection\n";
                }
                try_matching = false;
            }
        }
        else
        {
            std::tie(cur_node_mapping, nr_edges_to_contract) =
                find_contraction_mapping<VectorType>(G, opts.verbose);
        }

        if (nr_edges_to_contract == 0)
        {
            if (opts.verbose)
                std::cout << "# iterations = " << iter << "\n";
            break;
        }

        Graph<VectorType> new_G = G.contract(cur_node_mapping);
        if (opts.verbose)
        {
            std::cout << "original G size " << G.num_nodes() << "\n";
            std::cout << "contracted G size " << new_G.num_nodes() << "\n";
        }
        assert(new_G.num_nodes() < G.num_nodes());

        if (opts.verbose)
        {
            VectorType<float> slc = new_G.self_loop_costs();
            float energy_reduction = thrust::reduce(slc.begin(), slc.end());
            std::cout << "energy reduction " << energy_reduction << "\n";
        }

        if (rama_solver_detail::has_bad_contractions<VectorType>(new_G))
            throw std::runtime_error("Found bad contractions");

        G = std::move(new_G);
        G.remove_self_loops();

        if (opts.verbose)
            std::cout << "energy after iteration " << iter << ": " << G.sum()
                      << ", #components = " << G.num_nodes() << "\n";

        rama_solver_detail::map_node_labels<VectorType>(cur_node_mapping, node_mapping);

        if (opts.dump_timeline)
        {
            std::vector<int> current_timeline(node_mapping.size());
            thrust::copy(node_mapping.begin(), node_mapping.end(), current_timeline.begin());
            timeline.push_back(current_timeline);
        }

        if (opts.max_time_sec >= 0)
        {
            auto end = std::chrono::steady_clock::now();
            auto time = std::chrono::duration_cast<std::chrono::seconds>(end - begin).count();
            if (time > opts.max_time_sec)
                break;
        }
    }

    if (opts.verbose)
        std::cout << "final energy = " << G.sum() << "\n";

    return {node_mapping, final_lb, timeline};
}