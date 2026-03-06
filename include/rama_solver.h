#pragma once

#include "graph.h"
#include "lifted_multicut_utils.h"
#include "dual_solver.h"
#include "edge_contractions.h"
#include "lifted_edge_contraction.h"
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

// Lifted multicut solver. Works on both CPU (thrust::host_vector) and GPU (thrust::device_vector).
// The standard multicut is the special case with an empty lifted graph.
// Returns (node_mapping, lower_bound, timeline).
template<template<typename> class VectorType>
std::tuple<VectorType<int>, double, std::vector<std::vector<int>>>
rama_solver(Graph<VectorType>& base_G, Graph<VectorType>& lifted_G,
            const multicut_solver_options& opts)
{
    MEASURE_CUMULATIVE_FUNCTION_EXECUTION_TIME;
    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

    // Decide once whether this is a lifted multicut problem.
    // Must be checked before any dual solving, because triangulation
    // introduces diagonal edges that end up in lifted_G via
    // extract_costs_from_union — those are NOT original lifted edges.
    const bool use_cut_factors = opts.use_lifted_cut_constraints
                                 && lifted_G.num_directed_edges() > 0;

    // Save base costs before reparametrization. Union-graph dual solving
    // distributes lifted cost into base edges, making them appear more
    // positive. We keep the original so that lifted path contraction can
    // contract on undistorted costs (see below).
    Graph<VectorType> orig_base_G = base_G;

    // Initial dual solve on union graph
    Graph<VectorType> union_G = create_union_graph(base_G, lifted_G);
    const double final_lb = dual_solver<VectorType>(union_G, base_G, lifted_G,
        use_cut_factors, opts.max_cycle_length_lb, opts.num_dual_itr_lb,
        opts.num_outer_itr_dual, 1e-4, opts.verbose, opts.long_cycle_method,
        opts.triangle_budget_ratio);
    std::tie(base_G, lifted_G) = extract_costs_from_union(union_G, base_G);


    if (opts.verbose)
        std::cout << "initial energy = " << base_G.sum() + lifted_G.sum() << "\n";

    VectorType<int> node_mapping(base_G.num_nodes());
    thrust::sequence(node_mapping.begin(), node_mapping.end());

    std::vector<std::vector<int>> timeline;

    if (opts.only_compute_lb)
        return {VectorType<int>(), final_lb, timeline};

    bool try_matching = true;
    if (opts.matching_thresh_crossover_ratio > 1.0)
        try_matching = false;

    for (size_t iter = 0; base_G.num_directed_edges() > 0; ++iter)
    {
        if (iter > 0)
        {
            orig_base_G = base_G;
            union_G = create_union_graph(base_G, lifted_G);
            dual_solver<VectorType>(union_G, base_G, lifted_G, use_cut_factors,
                                    opts.max_cycle_length_primal,
                                    opts.num_dual_itr_primal, 1, 1e-4, opts.verbose,
                                    opts.long_cycle_method,
                                    opts.triangle_budget_ratio);
            std::tie(base_G, lifted_G) = extract_costs_from_union(union_G, base_G);
        }

        VectorType<int> cur_node_mapping;
        int nr_edges_to_contract = 0;
        bool used_lifted_contraction = false;

        // For lifted problems, first try path contraction: for each
        // attractive lifted edge, find a base-graph path through
        // non-negative edges and merge all nodes along it.
        // Uses reparametrized costs for path finding (non-negative filter).
        if (lifted_G.num_directed_edges() > 0 && false) // TODO: remove that, current debugging
        {
            std::tie(cur_node_mapping, nr_edges_to_contract) =
                find_lifted_contraction_mapping<VectorType>(base_G, lifted_G, opts.verbose);
            if (nr_edges_to_contract > 0)
            {
                used_lifted_contraction = true;
                if (opts.verbose)
                    std::cout << "contraction " << iter << ": lifted path ("
                              << nr_edges_to_contract << " edges)\n";
            }
        }

        // Fall back to regular base-edge contraction (matching or MST)
        if (nr_edges_to_contract == 0)
        {
            if (try_matching && false) // TODO: debugging
            {
                std::tie(cur_node_mapping, nr_edges_to_contract) =
                    rama_solver_detail::contraction_mapping_by_maximum_matching<VectorType>(
                        base_G, opts.mean_multiplier_mm, opts.verbose);

                if (opts.verbose)
                    std::cout << "contraction " << iter << ": matching ("
                              << nr_edges_to_contract << " edges)\n";

                if (nr_edges_to_contract < (int)base_G.num_nodes() * opts.matching_thresh_crossover_ratio)
                {
                    if (opts.verbose)
                    {
                        std::cout << "# edges to contract = " << nr_edges_to_contract
                                  << ", # vertices = " << base_G.num_nodes() << "\n";
                        std::cout << "switching to MST based contraction edge selection\n";
                    }
                    try_matching = false;
                }
            }
            else
            {
                std::tie(cur_node_mapping, nr_edges_to_contract) =
                    find_contraction_mapping<VectorType>(base_G, lifted_G, opts.verbose);

                if (opts.verbose)
                    std::cout << "contraction " << iter << ": MST ("
                              << nr_edges_to_contract << " edges)\n";
            }
        }

        if (nr_edges_to_contract == 0)
        {
            if (opts.verbose)
                std::cout << "# iterations = " << iter << "\n";
            break;
        }

        // When lifted path contraction is used, contract the original
        // (pre-reparametrization) base graph. Union-graph dual solving
        // distributes lifted cost into base edges, inflating their costs.
        // Using original costs ensures remaining edges accurately reflect
        // whether they should be cut or kept.
        Graph<VectorType> new_base_G = used_lifted_contraction
            ? orig_base_G.contract(cur_node_mapping)
            : base_G.contract(cur_node_mapping);

        if (opts.verbose)
        {
            std::cout << "original G size " << base_G.num_nodes() << "\n";
            std::cout << "contracted G size " << new_base_G.num_nodes() << "\n";
        }
        assert(new_base_G.num_nodes() < base_G.num_nodes());

        // Lifted path contraction may contract negative base edges (the
        // positive lifted cost outweighs them), so negative self-loops are
        // expected. Only check for bad contractions on regular contraction.
        if (!used_lifted_contraction &&
            rama_solver_detail::has_bad_contractions<VectorType>(new_base_G))
            throw std::runtime_error("Found bad contractions");


        base_G = std::move(new_base_G);

        Graph<VectorType> new_lifted_G;
        if (lifted_G.num_directed_edges() > 0)
        {
            if (opts.verbose)
                std::cout << "contracting lifted graph: " << lifted_G.num_directed_edges()
                          << " directed edges, " << lifted_G.num_nodes() << " nodes"
                          << ", node_mapping size " << cur_node_mapping.size() << "\n";
            new_lifted_G = lifted_G.contract(cur_node_mapping);

        }

        if (opts.verbose)
        {
            VectorType<float> base_slc = base_G.self_loop_costs();
            float energy_reduction = thrust::reduce(base_slc.begin(), base_slc.end());
            if (new_lifted_G.num_directed_edges() > 0)
            {
                VectorType<float> lifted_slc = new_lifted_G.self_loop_costs();
                energy_reduction += thrust::reduce(lifted_slc.begin(), lifted_slc.end());
            }
            std::cout << "energy reduction " << energy_reduction << "\n";
        }


        base_G.remove_self_loops();


        if (new_lifted_G.num_directed_edges() > 0)
        {
            new_lifted_G.remove_self_loops();

            lifted_G = std::move(new_lifted_G);
        }

        // Absorb lifted edges that became parallel to base edges after contraction.
        if (lifted_G.num_directed_edges() > 0 && base_G.num_directed_edges() > 0)
        {
            Graph<VectorType> u = create_union_graph(base_G, lifted_G);

            std::tie(base_G, lifted_G) = extract_costs_from_union(u, base_G);

        }

        if (opts.verbose)
            std::cout << "energy after iteration " << iter << ": "
                      << base_G.sum() + lifted_G.sum()
                      << ", #components = " << base_G.num_nodes() << "\n";

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
        std::cout << "final energy = " << base_G.sum() + lifted_G.sum() << "\n";

    return {node_mapping, final_lb, timeline};
}

// Standard multicut solver (special case with no lifted edges).
template<template<typename> class VectorType>
std::tuple<VectorType<int>, double, std::vector<std::vector<int>>>
rama_solver(Graph<VectorType>& G, const multicut_solver_options& opts)
{
    Graph<VectorType> empty_lifted;
    return rama_solver(G, empty_lifted, opts);
}

// Explicit instantiation declarations.
extern template
std::tuple<thrust::host_vector<int>, double, std::vector<std::vector<int>>>
rama_solver<thrust::host_vector>(
    Graph<thrust::host_vector>&, Graph<thrust::host_vector>&,
    const multicut_solver_options&);

extern template
std::tuple<thrust::device_vector<int>, double, std::vector<std::vector<int>>>
rama_solver<thrust::device_vector>(
    Graph<thrust::device_vector>&, Graph<thrust::device_vector>&,
    const multicut_solver_options&);

extern template
std::tuple<thrust::host_vector<int>, double, std::vector<std::vector<int>>>
rama_solver<thrust::host_vector>(
    Graph<thrust::host_vector>&, const multicut_solver_options&);

extern template
std::tuple<thrust::device_vector<int>, double, std::vector<std::vector<int>>>
rama_solver<thrust::device_vector>(
    Graph<thrust::device_vector>&, const multicut_solver_options&);
