#define THRUST_DEVICE_SYSTEM THRUST_DEVICE_SYSTEM_CPP
#define THRUST_HOST_SYSTEM THRUST_HOST_SYSTEM_CPP

#include <vector>
#include <tuple>
#include <chrono>
#include "rama_py_common.h"
#include "rama_solver.h"
#include "rama_utils.h"

#ifdef WITH_TORCH
#include <torch/extension.h>
#include <torch/torch.h>
#define CHECK_CPU(x) TORCH_CHECK(x.device().is_cpu(), #x " must be a CPU tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CPU(x); CHECK_CONTIGUOUS(x)
#endif

static std::tuple<thrust::host_vector<int>, double, std::vector<std::vector<int>>>
solve_cpu(thrust::host_vector<int>&& i, thrust::host_vector<int>&& j,
          thrust::host_vector<float>&& costs, const multicut_solver_options& opts)
{
    thrust::host_vector<int> sanitized_node_ids;
    if (opts.sanitize_graph)
        sanitized_node_ids = compute_sanitized_graph<thrust::host_vector>(i, j, costs);
    else
        sort_edge_nodes<thrust::host_vector>(i, j);

    HostGraph G(std::move(i), std::move(j), std::move(costs));
    auto [node_mapping, lb, timeline] = rama_solver<thrust::host_vector>(G, opts);

    if (opts.sanitize_graph)
        node_mapping = desanitize_node_labels<thrust::host_vector>(node_mapping, sanitized_node_ids);

    return {std::move(node_mapping), lb, std::move(timeline)};
}

static std::tuple<std::vector<int>, double, int, std::vector<std::vector<int>>>
rama_cpu(const std::vector<int>& i, const std::vector<int>& j,
         const std::vector<float>& costs, const multicut_solver_options& opts)
{
    thrust::host_vector<int> i_host(i.begin(), i.end());
    thrust::host_vector<int> j_host(j.begin(), j.end());
    thrust::host_vector<float> costs_host(costs.begin(), costs.end());

    auto start = std::chrono::steady_clock::now();
    auto [node_mapping, lb, timeline] = solve_cpu(std::move(i_host), std::move(j_host),
                                                  std::move(costs_host), opts);
    auto end = std::chrono::steady_clock::now();
    int dur = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    std::vector<int> h_node_mapping(node_mapping.begin(), node_mapping.end());
    return {h_node_mapping, lb, dur, timeline};
}

#ifdef WITH_TORCH
std::vector<torch::Tensor> rama_torch(
    const torch::Tensor& _i,
    const torch::Tensor& _j,
    const torch::Tensor& _costs,
    const multicut_solver_options& opts)
{
    CHECK_INPUT(_i);
    CHECK_INPUT(_j);
    CHECK_INPUT(_costs);
    if (_i.size(0) != _j.size(0) || _i.size(0) != _costs.size(0))
        throw std::runtime_error("Input shapes must match");
    if (_i.scalar_type() != _j.scalar_type())
        throw std::runtime_error("Node indices i, j should be of same type");

    TORCH_CHECK(_i.dim() == 1, "i should be one-dimensional");
    TORCH_CHECK(_j.dim() == 1, "j should be one-dimensional");
    TORCH_CHECK(_costs.dim() == 1, "costs should be one-dimensional");

    thrust::host_vector<int> i(_i.data_ptr<int32_t>(), _i.data_ptr<int32_t>() + _i.size(0));
    thrust::host_vector<int> j(_j.data_ptr<int32_t>(), _j.data_ptr<int32_t>() + _j.size(0));
    thrust::host_vector<float> costs(_costs.data_ptr<float>(), _costs.data_ptr<float>() + _costs.size(0));

    auto [node_mapping, lb, timeline] = solve_cpu(std::move(i), std::move(j), std::move(costs), opts);

    torch::Tensor node_mapping_torch = at::empty({long(node_mapping.size())}, _i.options());
    thrust::copy(node_mapping.begin(), node_mapping.end(), node_mapping_torch.data_ptr<int32_t>());

    torch::Tensor lb_torch = torch::full({1}, lb, _costs.options().dtype(torch::kFloat64));
    return {node_mapping_torch, lb_torch};
}
#endif

std::vector<std::vector<int>> rama_cpu_pointers(const int* const i, const int* const j, const float* const edge_costs,
                        int* const node_labels, const int num_nodes, const int num_edges, const multicut_solver_options& opts)
{
    thrust::host_vector<int> i_host(i, i + num_edges);
    thrust::host_vector<int> j_host(j, j + num_edges);
    thrust::host_vector<float> costs_host(edge_costs, edge_costs + num_edges);

    auto [node_mapping, lb, timeline] = solve_cpu(std::move(i_host), std::move(j_host),
                                                  std::move(costs_host), opts);
    thrust::copy(node_mapping.begin(), node_mapping.end(), node_labels);
    return timeline;
}

PYBIND11_MODULE(rama_py_cpu, m) {
    m.doc() = "Bindings for RAMA (CPU solver): Rapid algorithm for multicut. "
                "For running purely primal algorithm initialize multicut_solver_options with \"P\". "
                "For algorithm with best quality call with \"PD+\" where \"PD\" is default algorithm. "
                "For only computing the lower bound call with \"D\". ";
    bind_multicut_solver_options(m);

    m.def("rama_cpu", [](const std::vector<int>& i, const std::vector<int>& j, const std::vector<float>& edge_costs, const multicut_solver_options& opts) {
            return rama_cpu(i, j, edge_costs, opts);
            });

    m.def("rama_cpu_pointers", [](const long i_ptr, const long j_ptr, const long edge_costs_ptr, const long node_labels_out_ptr,
                                  const int num_nodes, const int num_edges, const multicut_solver_options& opts) {
            const int* const i = reinterpret_cast<const int* const>(i_ptr);
            const int* const j = reinterpret_cast<const int* const>(j_ptr);
            const float* const edge_costs = reinterpret_cast<const float* const>(edge_costs_ptr);
            int* const node_labels = reinterpret_cast<int* const>(node_labels_out_ptr);
            return rama_cpu_pointers(i, j, edge_costs, node_labels, num_nodes, num_edges, opts);
            });

    bind_read_multicut_file(m);

    #ifdef WITH_TORCH
        m.def("rama_torch", &rama_torch, "RAMA CPU solver with torch interface.");
    #endif
}
