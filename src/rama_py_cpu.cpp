#define THRUST_DEVICE_SYSTEM THRUST_DEVICE_SYSTEM_CPP
#define THRUST_HOST_SYSTEM THRUST_HOST_SYSTEM_CPP

#include <chrono>
#include "rama_solver.h"
#include "rama_utils.h"
#include "rama_py_backend.h"

rama_py_result rama_solve_cpu(const std::vector<int>& i, const std::vector<int>& j,
                              const std::vector<float>& costs, const multicut_solver_options& opts)
{
    thrust::host_vector<int> i_h(i.begin(), i.end());
    thrust::host_vector<int> j_h(j.begin(), j.end());
    thrust::host_vector<float> costs_h(costs.begin(), costs.end());

    auto start = std::chrono::steady_clock::now();
    thrust::host_vector<int> sanitized_node_ids;
    if (opts.sanitize_graph)
        sanitized_node_ids = compute_sanitized_graph<thrust::host_vector>(i_h, j_h, costs_h);
    else
        sort_edge_nodes<thrust::host_vector>(i_h, j_h);

    HostGraph G(std::move(i_h), std::move(j_h), std::move(costs_h));
    auto [node_mapping, lb, timeline] = rama_solver<thrust::host_vector>(G, opts);

    if (opts.sanitize_graph)
        node_mapping = desanitize_node_labels<thrust::host_vector>(node_mapping, sanitized_node_ids);
    auto end = std::chrono::steady_clock::now();
    int dur = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    return {std::vector<int>(node_mapping.begin(), node_mapping.end()), lb, dur, std::move(timeline)};
}
