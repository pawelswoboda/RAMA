#include <cuda_runtime.h>
#include "iterative_rama_cuda.h"
#include "rama_cuda.h"
#include "dual_solver.h"
#include "multicut_solver_options.h"
#include "rama_utils.h"

iterative_rama_cuda::iterative_rama_cuda(thrust::device_vector<int>&& i, thrust::device_vector<int>&& j, thrust::device_vector<float>&& costs, const multicut_solver_options& opts, const int device)
{
    opts_ = opts;
    cudaSetDevice(device);
    if (opts_.sanitize_graph)
        sanitized_node_ids_ = compute_sanitized_graph(i, j, costs);

    if (!thrust::is_sorted(i.begin(), i.end()))
        throw std::runtime_error("Input graph should already be dCOO sorted.");
    A_ = dCOO(std::move(j), std::move(i), std::move(costs), true, true);

    lb_ = dual_solver(A_, opts.max_cycle_length_lb, opts.num_dual_itr_lb, opts.tri_memory_factor, opts.num_outer_itr_dual, 1e-4, opts.verbose);

    if (opts_.verbose) std::cout << "initial energy = " << A_.sum() << "\n";

    node_mapping_ = thrust::device_vector<int>(A_.max_dim());
    thrust::sequence(node_mapping_.begin(), node_mapping_.end());
        
    if (opts.matching_thresh_crossover_ratio > 1.0)
        try_edges_to_contract_by_maximum_matching_ = false;
}

std::tuple<bool, int> iterative_rama_cuda::try_do_primal_step(int* out_node_mapping)
{
    thrust::device_vector<int> cur_node_mapping;
    bool performed_contraction;
    std::tie(performed_contraction, cur_node_mapping) = single_primal_iteration(node_mapping_, A_, try_edges_to_contract_by_maximum_matching_, opts_, max_num_dual_itr_);
    max_num_dual_itr_ = 1;
    const int prev_num_nodes = cur_node_mapping.size();
    thrust::copy(cur_node_mapping.begin(), cur_node_mapping.end(), out_node_mapping);
    return {performed_contraction, prev_num_nodes};
}

void iterative_rama_cuda::set_edge_costs(const float* const in_ptr)
{
    float* existing_ptr = A_.get_writeable_data_ptr();
    thrust::device_ptr<const float> dev_in_ptr = thrust::device_ptr<const float>(in_ptr);
    thrust::copy(dev_in_ptr, dev_in_ptr + A_.nnz(), existing_ptr);
}

void iterative_rama_cuda::get_node_mapping(int* ptr) const
{
    thrust::device_ptr<int> dev_ptr(ptr);
    if (opts_.sanitize_graph)
        return desanitize_node_labels(node_mapping_, sanitized_node_ids_, dev_ptr);
    else
        thrust::copy(node_mapping_.begin(), node_mapping_.end(), dev_ptr);
}

void sort_edge_list_dCOO(thrust::device_ptr<int> i, thrust::device_ptr<int> j, const int num_edges)
{
    sort_edge_nodes(i, j, num_edges);
    coo_sorting(i, j, num_edges);
}